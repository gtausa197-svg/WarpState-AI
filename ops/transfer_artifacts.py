#!/usr/bin/env python3
"""Resume large artifact uploads or downloads over independent SSH connections.

Both endpoints stay inside /workspace/WarpState. Receivers need only Python 3
on Linux. No daemon, additional port, password file, or third-party package is
required. Completed chunks are read and checked on resume; the final file is
published only after full SHA256 verification and never replaces another file.
"""

import argparse
import concurrent.futures
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import threading
import time


ROOT = Path('/workspace/WarpState')
BLOCK = 1024 * 1024


def workspace_path(value):
    """Reject lexical escapes and symlinks, including intermediate components."""
    path = Path(os.path.abspath(value))
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError(f'Path must be below {ROOT}: {path}')
    current = Path('/')
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f'Symlink is not allowed: {current}')
    return path


def remote_main():
    # This function is sent to Python over SSH; keep it self-contained.
    import ctypes
    import fcntl
    import hashlib
    import json
    import os
    from pathlib import Path
    import stat
    import sys
    import uuid

    root = Path('/workspace/WarpState')
    block = 1024 * 1024
    action, destination, expected_sha = sys.argv[1:4]

    def checked(value):
        path = Path(os.path.abspath(value))
        if not path.is_relative_to(root):
            raise ValueError(f'Path must be below {root}: {path}')
        current = Path('/')
        for part in path.parts[1:]:
            current /= part
            if current.is_symlink():
                raise ValueError(f'Symlink is not allowed: {current}')
        return path

    def open_regular(path, flags, mode=0o600):
        checked(path)
        fd = os.open(path, flags | os.O_NOFOLLOW, mode)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError(f'Not a regular file: {path}')
        return fd

    def sync_directory(path):
        fd = os.open(path, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def atomic_json(path, value):
        checked(path)
        temporary = checked(str(path) + '.' + uuid.uuid4().hex + '.tmp')
        fd = open_regular(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(value, stream, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            sync_directory(path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def read_json(path):
        fd = open_regular(path, os.O_RDONLY)
        with os.fdopen(fd, 'r') as stream:
            return json.load(stream)

    def digest(path, offset=0, count=None):
        fd = open_regular(path, os.O_RDONLY)
        checksum = hashlib.sha256()
        with os.fdopen(fd, 'rb') as stream:
            stream.seek(offset)
            while count is None or count > 0:
                data = stream.read(block if count is None else min(block, count))
                if not data:
                    if count:
                        raise ValueError('Unexpected end of file during verification')
                    break
                checksum.update(data)
                if count is not None:
                    count -= len(data)
        return checksum.hexdigest()

    target = checked(destination)
    if target == root:
        raise ValueError('Destination must be a file below the workspace root')
    target.parent.mkdir(parents=True, exist_ok=True)
    checked(target.parent)
    partial = checked(str(target) + '.uploading')
    state_path = checked(str(target) + '.upload-state.json')
    receipts = checked(str(target) + '.upload-chunks')
    lock_fd = open_regular(str(target) + '.upload.lock', os.O_RDWR | os.O_CREAT)
    fcntl.flock(lock_fd, fcntl.LOCK_SH if action == 'chunk' else fcntl.LOCK_EX)

    def target_matches(size):
        if not target.exists():
            return False
        if target.stat().st_size != size or digest(target) != expected_sha:
            raise ValueError(f'Existing destination differs; refusing overwrite: {target}')
        return True

    if action == 'prepare':
        incoming = json.load(sys.stdin)
        if incoming['sha256'] != expected_sha:
            raise ValueError('Source identity mismatch')
        if target_matches(incoming['size']):
            print(json.dumps({'complete': True, 'sha256': expected_sha}))
            return
        if state_path.exists():
            state = read_json(state_path)
            if state != incoming:
                raise ValueError('Partial upload belongs to a different source or chunk size')
        else:
            if partial.exists() or receipts.exists():
                raise ValueError('Unclaimed upload sidecar exists; refusing to replace it')
            # Persist identity before creating the sparse file, so interruption
            # between these operations can safely resume.
            atomic_json(state_path, incoming)
            state = incoming
        if not partial.exists():
            fd = open_regular(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            try:
                os.ftruncate(fd, state['size'])
                os.fsync(fd)
            finally:
                os.close(fd)
        if partial.stat().st_size != state['size']:
            raise ValueError('Partial upload has an unexpected size')
        receipts.mkdir(exist_ok=True)
        completed = []
        for index, expected in enumerate(state['chunk_sha256']):
            receipt = checked(receipts / f'{index:08d}.json')
            if not receipt.exists():
                continue
            offset = index * state['chunk_size']
            length = min(state['chunk_size'], state['size'] - offset)
            try:
                valid = read_json(receipt) == {'sha256': expected, 'size': length}
                valid = valid and digest(partial, offset, length) == expected
            except (ValueError, OSError):
                valid = False
            if valid:
                completed.append(index)
            else:
                receipt.unlink()
        sync_directory(receipts)
        print(json.dumps({'complete': False, 'completed': completed}))
        return

    state = read_json(state_path)
    if state['sha256'] != expected_sha:
        raise ValueError('Upload source identity mismatch')
    if action == 'chunk':
        index = int(sys.argv[4])
        if not 0 <= index < len(state['chunk_sha256']):
            raise ValueError('Invalid chunk index')
        offset = index * state['chunk_size']
        length = min(state['chunk_size'], state['size'] - offset)
        checksum = hashlib.sha256()
        fd = open_regular(partial, os.O_WRONLY)
        try:
            remaining = length
            position = offset
            while remaining:
                data = sys.stdin.buffer.read(min(block, remaining))
                if not data:
                    raise ValueError('Upload stream ended early')
                checksum.update(data)
                view = memoryview(data)
                while view:
                    count = os.pwrite(fd, view, position)
                    if count <= 0:
                        raise OSError('pwrite made no progress')
                    view = view[count:]
                    position += count
                remaining -= len(data)
            if sys.stdin.buffer.read(1):
                raise ValueError('Upload stream exceeded its chunk length')
            actual = checksum.hexdigest()
            if actual != state['chunk_sha256'][index]:
                raise ValueError('Chunk SHA256 mismatch')
            os.fsync(fd)
        finally:
            os.close(fd)
        atomic_json(checked(receipts / f'{index:08d}.json'),
                    {'sha256': actual, 'size': length})
        print(json.dumps({'index': index, 'sha256': actual, 'bytes': length}))
        return

    if action != 'finalize':
        raise ValueError('Unknown receiver action')
    if target_matches(state['size']):
        print(json.dumps({'complete': True, 'sha256': expected_sha, 'bytes': state['size']}))
        return
    for index, expected in enumerate(state['chunk_sha256']):
        length = min(state['chunk_size'], state['size'] - index * state['chunk_size'])
        if read_json(checked(receipts / f'{index:08d}.json')) != {
                'sha256': expected, 'size': length}:
            raise ValueError(f'Missing or invalid receipt for chunk {index}')
    if partial.stat().st_size != state['size'] or digest(partial) != expected_sha:
        raise ValueError('Full destination SHA256 mismatch; partial upload retained')
    # Linux renameat2(RENAME_NOREPLACE) atomically publishes the verified file
    # while protecting a destination created by another process during hashing.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                       ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(partial), -100, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))
    sync_directory(target.parent)
    print(json.dumps({'complete': True, 'sha256': expected_sha, 'bytes': state['size']}))


def identity(path):
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f'Not a regular file: {path}')
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def source_manifest(path, chunk_size):
    before = identity(path)
    whole = hashlib.sha256()
    chunks = []
    with path.open('rb') as stream:
        while True:
            checksum = hashlib.sha256()
            consumed = 0
            while consumed < chunk_size:
                data = stream.read(min(BLOCK, chunk_size - consumed))
                if not data:
                    break
                whole.update(data)
                checksum.update(data)
                consumed += len(data)
            if not consumed:
                break
            chunks.append(checksum.hexdigest())
    if identity(path) != before:
        raise ValueError('Source changed while computing its SHA256')
    return {'version': 1, 'size': before[2], 'sha256': whole.hexdigest(),
            'chunk_size': chunk_size, 'chunk_sha256': chunks}, before


def remote_source_main():
    # Sent over SSH without installing or changing any remote source files.
    import hashlib
    import json
    import os
    from pathlib import Path
    import stat
    import sys

    action, name = sys.argv[1:3]
    root = Path('/workspace/WarpState')
    path = Path(os.path.abspath(name))
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Path must be below {root}: {path}')
    current = Path('/')
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f'Symlink is not allowed: {current}')

    def identity(info):
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Source is not a regular file')
        return [info.st_dev, info.st_ino, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns]

    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = identity(os.fstat(stream.fileno()))

        def unchanged():
            if (identity(os.fstat(stream.fileno())) != before or
                    identity(path.stat(follow_symlinks=False)) != before):
                raise ValueError('Source changed during transfer')

        unchanged()
        if action == 'manifest':
            chunk_size = int(sys.argv[3])
            if chunk_size < 1:
                raise ValueError('Chunk size must be positive')
            whole = hashlib.sha256()
            chunks = []
            while True:
                checksum = hashlib.sha256()
                consumed = 0
                while consumed < chunk_size:
                    data = stream.read(min(1024 * 1024, chunk_size - consumed))
                    if not data:
                        break
                    whole.update(data)
                    checksum.update(data)
                    consumed += len(data)
                if not consumed:
                    break
                chunks.append(checksum.hexdigest())
            unchanged()
            print(json.dumps({'manifest': {
                'version': 1, 'size': before[2], 'sha256': whole.hexdigest(),
                'chunk_size': chunk_size, 'chunk_sha256': chunks},
                'identity': before}))
            return

        if before != json.loads(sys.argv[3]):
            raise ValueError('Source changed during transfer')
        if action == 'identity':
            unchanged()
            print(json.dumps({'identity': before}))
            return
        if action != 'chunk':
            raise ValueError('Unknown source action')
        offset, remaining = map(int, sys.argv[4:6])
        if offset < 0 or remaining < 0 or offset + remaining > before[2]:
            raise ValueError('Invalid source chunk range')
        stream.seek(offset)
        while remaining:
            data = stream.read(min(1024 * 1024, remaining))
            if not data:
                raise ValueError('Source ended early')
            sys.stdout.buffer.write(data)
            remaining -= len(data)
        sys.stdout.buffer.flush()
        unchanged()


def ssh_command(args):
    command = ['ssh', '-F', '/dev/null', '-p', str(args.port),
               '-i', str(args.key), '-o', 'BatchMode=yes',
               '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
               '-o', f'UserKnownHostsFile={args.known_hosts}',
               '-o', 'GlobalKnownHostsFile=/dev/null',
               '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15',
               '-o', 'ServerAliveCountMax=4', '-o', 'ControlMaster=no',
               '-o', 'ControlPath=none']
    if args.compress:
        command.append('-C')
    return command + [args.host]


class SSHSource:
    def __init__(self, args, source):
        self.source = source
        self.ssh = ssh_command(args)
        self.code = inspect.getsource(remote_source_main) + '\nremote_source_main()\n'

    def command(self, action, *parameters):
        remote = ['python3', '-c', self.code, action, str(self.source),
                  *(str(value) for value in parameters)]
        return self.ssh + [shlex.join(remote)]

    def call(self, action, *parameters):
        result = subprocess.run(self.command(action, *parameters),
                                stdin=subprocess.DEVNULL, capture_output=True)
        if result.returncode:
            raise RuntimeError(f'SSH source failed ({result.returncode}): '
                               + result.stderr.decode(errors='replace').strip())
        return json.loads(result.stdout)

    def manifest(self, chunk_size):
        result = self.call('manifest', chunk_size)
        return result['manifest'], result['identity']

    def check_identity(self, expected):
        if self.call('identity', json.dumps(expected))['identity'] != expected:
            raise ValueError('Source changed during transfer')


class SSHReceiver:
    def __init__(self, args, destination, manifest):
        self.destination = destination
        self.manifest = manifest
        self.ssh = ssh_command(args)
        self.code = inspect.getsource(remote_main) + '\nremote_main()\n'
        self.processes = set()
        self.lock = threading.Lock()
        self.cancelled = threading.Event()

    def command(self, action, index):
        remote = ['python3', '-c', self.code, action, str(self.destination),
                  self.manifest['sha256']]
        if index is not None:
            remote.append(str(index))
        return self.ssh + [shlex.join(remote)]

    def start_process(self, command, **kwargs):
        if self.cancelled.is_set():
            raise RuntimeError('Transfer cancelled')
        process = subprocess.Popen(command, **kwargs)
        with self.lock:
            self.processes.add(process)
            if self.cancelled.is_set():
                process.terminate()
        return process

    def call(self, action, payload=None, source=None, index=None):
        process = self.start_process(self.command(action, index),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        try:
            if source is None:
                out, err = process.communicate(payload)
            else:
                offset = index * self.manifest['chunk_size']
                remaining = min(self.manifest['chunk_size'], self.manifest['size'] - offset)
                with source.open('rb') as stream:
                    stream.seek(offset)
                    while remaining:
                        data = stream.read(min(BLOCK, remaining))
                        if not data:
                            raise ValueError('Source ended early')
                        process.stdin.write(data)
                        remaining -= len(data)
                process.stdin.close()
                process.stdin = None
                out, err = process.communicate()
            if process.returncode:
                raise RuntimeError(f'SSH receiver failed ({process.returncode}): '
                                   + err.decode(errors='replace').strip())
            return json.loads(out)
        except BaseException:
            if process.poll() is None:
                process.terminate()
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
                process.stdin = None
            process.communicate()
            raise
        finally:
            with self.lock:
                self.processes.discard(process)

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            for process in self.processes:
                if process.poll() is None:
                    process.terminate()


class PullReceiver(SSHReceiver):
    """Run the same receiver locally, with each SSH stdout piped to its stdin."""

    def __init__(self, args, destination, manifest, remote_source, initial_identity):
        super().__init__(args, destination, manifest)
        self.remote_source = remote_source
        self.initial_identity = initial_identity

    def command(self, action, index):
        command = [sys.executable, '-c', self.code, action, str(self.destination),
                   self.manifest['sha256']]
        return command + ([] if index is None else [str(index)])

    def call(self, action, payload=None, source=None, index=None):
        if action != 'chunk':
            return super().call(action, payload, index=index)
        offset = index * self.manifest['chunk_size']
        length = min(self.manifest['chunk_size'], self.manifest['size'] - offset)
        processes = []
        communicated = set()
        try:
            sender = self.start_process(self.remote_source.command(
                'chunk', json.dumps(self.initial_identity), offset, length),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            processes.append(sender)
            receiver = self.start_process(self.command('chunk', index),
                                          stdin=sender.stdout, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
            processes.append(receiver)
            # Only the receiver owns the pipe's read end; data never accumulates
            # in this process, and receiver failure breaks the sender's pipe.
            sender.stdout.close()
            sender.stdout = None
            out, receiver_error = receiver.communicate()
            communicated.add(receiver)
            _, sender_error = sender.communicate()
            communicated.add(sender)
            if sender.returncode:
                raise RuntimeError(f'SSH source failed ({sender.returncode}): '
                                   + sender_error.decode(errors='replace').strip())
            if receiver.returncode:
                raise RuntimeError(f'Local receiver failed ({receiver.returncode}): '
                                   + receiver_error.decode(errors='replace').strip())
            return json.loads(out)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                if process not in communicated:
                    process.communicate()
                with self.lock:
                    self.processes.discard(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=workspace_path)
    parser.add_argument('destination', type=workspace_path)
    parser.add_argument('--host', required=True, help='user@hostname')
    parser.add_argument('--port', type=int, default=22)
    parser.add_argument('--key', type=workspace_path, required=True)
    parser.add_argument('--known-hosts', type=workspace_path, required=True,
                        help='Existing verified host-key file inside the workspace')
    parser.add_argument('--connections', type=int, default=8)
    parser.add_argument('--chunk-mib', type=int, default=256)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--pull', action='store_true',
                        help='Download source from SSH host to local destination')
    parser.add_argument('--expected-sha256',
                        help='Reject a source that does not match this known SHA256')
    parser.add_argument('--manifest-dir', type=workspace_path,
                        default=ROOT / 'tmp' / 'artifact-transfer')
    args = parser.parse_args()
    if args.connections < 1 or args.chunk_mib < 1 or args.retries < 1:
        parser.error('Connections, chunk size, and retries must be positive')
    if args.host.startswith('-') or not 1 <= args.port <= 65535:
        parser.error('Invalid SSH host or port')
    if args.expected_sha256 is not None:
        if re.fullmatch(r'[0-9a-fA-F]{64}', args.expected_sha256) is None:
            parser.error('--expected-sha256 must contain exactly 64 hexadecimal digits')
        args.expected_sha256 = args.expected_sha256.lower()
    for path in (args.key, args.known_hosts) + (() if args.pull else (args.source,)):
        identity(path)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    workspace_path(args.manifest_dir)
    print(f'Computing source SHA256 and {args.chunk_mib} MiB chunk hashes: {args.source}',
          flush=True)
    remote_source = SSHSource(args, args.source) if args.pull else None
    if args.pull:
        manifest, initial_identity = remote_source.manifest(args.chunk_mib * BLOCK)
    else:
        manifest, initial_identity = source_manifest(args.source, args.chunk_mib * BLOCK)
    if args.expected_sha256 is not None and manifest['sha256'] != args.expected_sha256:
        raise ValueError('Source SHA256 does not match --expected-sha256')
    manifest_key = (f'{args.host}:{args.port}:{args.source}' if args.pull else str(args.source))
    manifest_name = hashlib.sha256(manifest_key.encode()).hexdigest()[:20] + '.json'
    manifest_path = workspace_path(args.manifest_dir / manifest_name)
    # This local manifest is informational; each invocation hashes its source
    # afresh and validates remote receipts rather than trusting cached claims.
    temporary = workspace_path(str(manifest_path) + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump({'source': str(args.source), **manifest}, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_path)
    print(f'Source: {manifest["size"]:,} bytes; SHA256 {manifest["sha256"]}', flush=True)
    receiver = (PullReceiver(args, args.destination, manifest, remote_source, initial_identity)
                if args.pull else SSHReceiver(args, args.destination, manifest))
    prepared = receiver.call('prepare', json.dumps(manifest).encode())
    if prepared.get('complete'):
        print('Destination already exists and its full SHA256 matches.', flush=True)
        return
    completed = set(prepared['completed'])
    pending = [i for i in range(len(manifest['chunk_sha256'])) if i not in completed]
    done_bytes = sum(min(manifest['chunk_size'], manifest['size'] - i * manifest['chunk_size'])
                     for i in completed)
    initial_bytes = done_bytes
    start = time.monotonic()
    print(f'Resuming: {len(completed)} verified chunks; {len(pending)} remaining; '
          f'{args.connections} SSH connections.', flush=True)

    def send_chunk(index):
        for attempt in range(args.retries):
            if not args.pull and identity(args.source) != initial_identity:
                raise ValueError('Source changed during transfer')
            try:
                result = receiver.call('chunk', source=args.source, index=index)
                if (result['index'] != index or
                        result['sha256'] != manifest['chunk_sha256'][index]):
                    raise ValueError('Receiver returned an invalid chunk acknowledgment')
                return result
            except Exception:
                if receiver.cancelled.is_set() or attempt + 1 == args.retries:
                    raise
                receiver.cancelled.wait(min(2 ** attempt, 10))

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.connections)
    try:
        futures = [executor.submit(send_chunk, index) for index in pending]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed.add(result['index'])
            done_bytes += result['bytes']
            elapsed = max(time.monotonic() - start, 0.001)
            speed = (done_bytes - initial_bytes) / elapsed / BLOCK
            percent = 100 * done_bytes / max(manifest['size'], 1)
            print(f'{percent:6.2f}%  {done_bytes:,}/{manifest["size"]:,} bytes  '
                  f'{len(completed)}/{len(manifest["chunk_sha256"])} chunks  '
                  f'{speed:.1f} MiB/s', flush=True)
    except BaseException:
        receiver.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    executor.shutdown(wait=True)
    if args.pull:
        remote_source.check_identity(initial_identity)
    elif identity(args.source) != initial_identity:
        raise ValueError('Source changed during transfer; refusing final publication')
    print('Verifying full destination SHA256 before atomic publication...', flush=True)
    result = receiver.call('finalize')
    if result['sha256'] != manifest['sha256'] or not result['complete']:
        raise ValueError('Invalid final verification response')
    print(f'Complete: {args.destination}; SHA256 {result["sha256"]}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; verified chunks are retained for resume.', file=sys.stderr)
        raise SystemExit(130)
