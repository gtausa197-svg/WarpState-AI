"""CPU-only checks for transfer integrity, interruption, and safe publication."""

import concurrent.futures
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    'transfer_artifacts', Path(__file__).parent / 'ops' / 'transfer_artifacts.py')
transfer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transfer)
RECEIVER = inspect.getsource(transfer.remote_main) + '\nremote_main()\n'
LOCAL_SSH = [sys.executable, '-c',
             'import os, shlex, sys; command = shlex.split(sys.argv[1]); '
             'os.execvp(command[0], command)']


def receive(action, target, manifest, data=b'', index=None, success=True):
    command = [sys.executable, '-c', RECEIVER, action, str(target), manifest['sha256']]
    if index is not None:
        command.append(str(index))
    result = subprocess.run(command, input=data, capture_output=True)
    if not success:
        assert result.returncode != 0
        return result.stderr.decode()
    assert result.returncode == 0, result.stderr.decode()
    return json.loads(result.stdout)


@pytest.fixture
def artifact(tmp_path):
    # Tests must be invoked with --basetemp below the workspace.
    transfer.workspace_path(tmp_path)
    data = bytes(range(251)) * 100
    source = tmp_path / 'source.bin'
    source.write_bytes(data)
    manifest, _ = transfer.source_manifest(source, 8192)
    return source, tmp_path / 'remote' / 'target.bin', data, manifest


def prepare(target, manifest):
    return receive('prepare', target, manifest, json.dumps(manifest).encode())


def upload(index, target, data, manifest):
    size = manifest['chunk_size']
    return receive('chunk', target, manifest, data[index * size:(index + 1) * size], index)


def test_parallel_upload_verifies_before_atomic_publication(artifact):
    source, target, data, manifest = artifact
    assert manifest['sha256'] == hashlib.sha256(data).hexdigest()
    assert prepare(target, manifest)['completed'] == []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda i: upload(i, target, data, manifest),
                                range(len(manifest['chunk_sha256']))))
    assert sum(row['bytes'] for row in results) == len(data)
    assert not target.exists()
    result = receive('finalize', target, manifest)
    assert result['complete'] and result['sha256'] == manifest['sha256']
    assert target.read_bytes() == source.read_bytes()
    assert not Path(str(target) + '.uploading').exists()
    assert prepare(target, manifest)['complete']


def test_resume_rechecks_receipts_and_retransmits_corrupt_chunks(artifact):
    _, target, data, manifest = artifact
    prepare(target, manifest)
    upload(0, target, data, manifest)
    assert prepare(target, manifest)['completed'] == [0]
    partial = Path(str(target) + '.uploading')
    with partial.open('r+b') as stream:
        stream.write(b'corrupt')
    assert prepare(target, manifest)['completed'] == []
    error = receive('chunk', target, manifest, b'interrupted', 0, success=False)
    assert 'ended early' in error
    assert prepare(target, manifest)['completed'] == []
    for i in range(len(manifest['chunk_sha256'])):
        upload(i, target, data, manifest)
    receive('finalize', target, manifest)
    assert target.read_bytes() == data


def test_wrong_chunk_hash_and_final_corruption_never_publish(artifact):
    _, target, data, manifest = artifact
    prepare(target, manifest)
    error = receive('chunk', target, manifest, b'x' * manifest['chunk_size'], 0, success=False)
    assert 'Chunk SHA256 mismatch' in error
    for i in range(len(manifest['chunk_sha256'])):
        upload(i, target, data, manifest)
    with Path(str(target) + '.uploading').open('r+b') as stream:
        stream.write(b'corrupt after receiving')
    error = receive('finalize', target, manifest, success=False)
    assert 'Full destination SHA256 mismatch' in error
    assert not target.exists()


def test_existing_destination_and_foreign_partial_are_protected(artifact):
    _, target, _, manifest = artifact
    target.parent.mkdir()
    target.write_bytes(b'existing content')
    error = receive('prepare', target, manifest, json.dumps(manifest).encode(), success=False)
    assert 'refusing overwrite' in error
    assert target.read_bytes() == b'existing content'
    target.unlink()
    partial = Path(str(target) + '.uploading')
    partial.write_bytes(b'foreign partial')
    error = receive('prepare', target, manifest, json.dumps(manifest).encode(), success=False)
    assert 'Unclaimed upload sidecar' in error
    assert partial.read_bytes() == b'foreign partial'


def test_paths_and_symlinks_are_rejected(artifact):
    source, target, _, manifest = artifact
    with pytest.raises(ValueError, match='below'):
        transfer.workspace_path('/tmp/outside.bin')
    target.parent.symlink_to(source.parent, target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        transfer.workspace_path(target)
    error = receive('prepare', target, manifest, json.dumps(manifest).encode(), success=False)
    assert 'Symlink is not allowed' in error


def test_destination_created_before_finalize_is_preserved(artifact):
    _, target, data, manifest = artifact
    prepare(target, manifest)
    for i in range(len(manifest['chunk_sha256'])):
        upload(i, target, data, manifest)
    target.write_bytes(b'created independently')
    error = receive('finalize', target, manifest, success=False)
    assert 'refusing overwrite' in error
    assert target.read_bytes() == b'created independently'


def test_client_streaming_without_an_ssh_server(artifact):
    source, target, data, manifest = artifact
    args = SimpleNamespace(port=22, key=source, known_hosts=source,
                           host='unused', compress=False)
    receiver = transfer.SSHReceiver(args, target, manifest)
    # Replace just the SSH transport with a local command dispatcher. The client
    # still serializes the remote command and streams the actual binary input.
    receiver.ssh = [sys.executable, '-c',
                    'import os, shlex, sys; command = shlex.split(sys.argv[1]); '
                    'os.execvp(command[0], command)']
    assert receiver.call('prepare', json.dumps(manifest).encode())['completed'] == []
    for index in range(len(manifest['chunk_sha256'])):
        assert receiver.call('chunk', source=source, index=index)['index'] == index
    assert receiver.call('finalize')['sha256'] == hashlib.sha256(data).hexdigest()
    assert target.read_bytes() == data


def test_expected_source_hash_rejected_before_ssh(artifact):
    source, target, _, _ = artifact
    command = [sys.executable, str(Path(transfer.__file__)), str(source), str(target),
               '--host', 'unused', '--key', str(source), '--known-hosts', str(source),
               '--expected-sha256', '0' * 64]
    result = subprocess.run(command, capture_output=True)
    assert result.returncode != 0
    assert b'Source SHA256 does not match --expected-sha256' in result.stderr
    assert not target.exists()


def pull_client(source, target, chunk_size=8192):
    args = SimpleNamespace(port=22, key=source, known_hosts=source,
                           host='unused', compress=False)
    remote = transfer.SSHSource(args, source)
    remote.ssh = LOCAL_SSH
    manifest, initial = remote.manifest(chunk_size)
    receiver = transfer.PullReceiver(args, target, manifest, remote, initial)
    return remote, receiver, manifest, initial


def test_parallel_pull_resumes_with_verified_receipts(artifact):
    source, target, data, expected = artifact
    remote, receiver, manifest, initial = pull_client(source, target)
    assert manifest == expected
    assert receiver.call('prepare', json.dumps(manifest).encode())['completed'] == []
    receiver.call('chunk', index=0)
    assert receiver.call('prepare', json.dumps(manifest).encode())['completed'] == [0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda i: receiver.call('chunk', index=i),
                                range(1, len(manifest['chunk_sha256']))))
    assert len(results) == len(manifest['chunk_sha256']) - 1
    assert not target.exists()
    remote.check_identity(initial)
    assert receiver.call('finalize')['sha256'] == manifest['sha256']
    assert target.read_bytes() == data
    assert receiver.processes == set()


def test_pull_changed_source_is_rejected_before_streaming(artifact):
    source, target, data, _ = artifact
    remote, receiver, manifest, initial = pull_client(source, target)
    receiver.call('prepare', json.dumps(manifest).encode())
    source.write_bytes(data + b'changed')
    with pytest.raises(RuntimeError, match='Source changed during transfer'):
        receiver.call('chunk', index=0)
    with pytest.raises(RuntimeError, match='Source changed during transfer'):
        remote.check_identity(initial)
    assert not target.exists()
    assert receiver.call('prepare', json.dumps(manifest).encode())['completed'] == []
    assert receiver.processes == set()


@pytest.mark.parametrize('failure', ['corrupt', 'exit'])
def test_pull_transport_corruption_or_failure_never_publishes(artifact, failure):
    source, target, _, _ = artifact
    remote, receiver, manifest, _ = pull_client(source, target)
    receiver.call('prepare', json.dumps(manifest).encode())
    # The fake transport either corrupts a byte or reports failure after sending
    # all bytes. Both endpoints' exit status must succeed before acknowledgment.
    code = ('import shlex, subprocess, sys; '
            'result = subprocess.run(shlex.split(sys.argv[1]), capture_output=True); '
            'data = result.stdout; '
            + ('data = bytes([data[0] ^ 1]) + data[1:]; ' if failure == 'corrupt' else '')
            + 'sys.stdout.buffer.write(data); sys.stderr.buffer.write(result.stderr); '
            + ('sys.exit(result.returncode)' if failure == 'corrupt' else 'sys.exit(42)'))
    remote.ssh = [sys.executable, '-c', code]
    expected = 'Chunk SHA256 mismatch' if failure == 'corrupt' else 'SSH source failed'
    with pytest.raises(RuntimeError, match=expected):
        receiver.call('chunk', index=0)
    assert not target.exists()
    assert receiver.processes == set()


def test_pull_source_symlinks_are_rejected_remotely(artifact):
    source, target, _, _ = artifact
    link = source.parent / 'linked.bin'
    link.symlink_to(source)
    args = SimpleNamespace(port=22, key=source, known_hosts=source,
                           host='unused', compress=False)
    remote = transfer.SSHSource(args, link)
    remote.ssh = LOCAL_SSH
    with pytest.raises(RuntimeError, match='Symlink is not allowed'):
        remote.manifest(8192)
    assert not target.exists()


@pytest.mark.parametrize('failure', [None, 'expected_hash', 'changed_before_finalize'])
def test_pull_cli_verifies_remote_source_and_final_publication(artifact, monkeypatch, failure):
    source, target, data, _ = artifact
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(transfer, 'ssh_command', lambda args: LOCAL_SSH)
    argv = [str(Path(transfer.__file__)), str(source), str(target), '--pull',
            '--host', 'unused', '--key', str(source), '--known-hosts', str(source),
            '--chunk-mib', '1', '--connections', '3', '--retries', '1',
            '--manifest-dir', str(source.parent / 'manifests'),
            '--expected-sha256', '0' * 64 if failure == 'expected_hash' else digest]
    monkeypatch.setattr(sys, 'argv', argv)
    if failure == 'changed_before_finalize':
        original = transfer.SSHSource.check_identity

        def change_then_check(remote, expected):
            source.write_bytes(data + b'changed after the last chunk')
            original(remote, expected)

        monkeypatch.setattr(transfer.SSHSource, 'check_identity', change_then_check)
    if failure:
        error = ValueError if failure == 'expected_hash' else RuntimeError
        expected = '--expected-sha256' if failure == 'expected_hash' else 'Source changed'
        with pytest.raises(error, match=expected):
            transfer.main()
        assert not target.exists()
        if failure == 'expected_hash':
            assert not target.parent.exists()
    else:
        transfer.main()
        assert target.read_bytes() == data
        # An already complete destination is independently hashed on a rerun.
        transfer.main()
