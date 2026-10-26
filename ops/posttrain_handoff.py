#!/usr/bin/env python3
"""Wait for a clean exact-step base exit, seal its weights, then run SFT once.

This is an explicitly armed Supervisor job. It never starts/stops base training,
changes its checkpoint, or runs DPO. All checkpoints loaded here are trusted
local artifacts. No CUDA calls are made by this orchestration process.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import http.client
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import xmlrpc.client

WORKSPACE = Path('/workspace/WarpState')
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / 'ops'))
import torch
from chat import capture_latest, sha256
from launch_training import supervise, training_environment, workspace_path

ARCHIVE_FILES = ('model.pt', 'snapshot.json', 'tokenizer.model', 'ARCHIVE.md')
STOPPED = False


def utc_now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def flush_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    flush_directory(path.parent)


def publish_directory(staging, destination):
    """Atomic no-replace publication, including when an empty target exists."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                       ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(destination), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))
    flush_directory(Path(destination).parent)


@contextlib.contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream.fileno()


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__('localhost', timeout=10)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class UnixTransport(xmlrpc.client.Transport):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def make_connection(self, host):
        return UnixHTTPConnection(self.path)


def process_status(socket_path, program):
    with xmlrpc.client.ServerProxy('http://localhost/RPC2',
                                  transport=UnixTransport(socket_path)) as rpc:
        return rpc.supervisor.getProcessInfo(program)


def clean_exit(info):
    return info.get('statename') == 'EXITED' and info.get('exitstatus') == 0


def validate_boundary(checkpoint, base, tokenizer_hash):
    if checkpoint.get('format') != 'warpstate-server-v1':
        raise ValueError('Expected a full base training checkpoint')
    for key in ('step', 'tokens_seen'):
        if checkpoint.get(key) != base[key]:
            raise ValueError(f'Base {key} must equal {base[key]}, got {checkpoint.get(key)}')
    if (checkpoint.get('val_step') != base['step'] or
            checkpoint.get('val_tokens') != base['tokens_seen']):
        raise ValueError('Base checkpoint must contain final validation at the exact stop boundary')
    if not math.isfinite(checkpoint.get('val_loss', float('nan'))):
        raise ValueError('Base validation loss is not finite')
    actual = checkpoint['signature']['data']['signature']['tokenizer_sha256']
    if actual != tokenizer_hash or actual != base['tokenizer_sha256']:
        raise ValueError('Base tokenizer identity differs from the pinned posttraining plan')
    if not checkpoint.get('model'):
        raise ValueError('Base checkpoint contains no model tensors')
    for name, tensor in checkpoint['model'].items():
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f'Expected original FP32 model weights: {name}: {tensor.dtype}')


def copy_regular(source, destination):
    """Physical byte copy; never a hardlink, reflink, or mutable shared snapshot."""
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'Expected regular source file: {source}')
    with source.open('rb') as reader, destination.open('xb') as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024**2)
        writer.flush()
        os.fsync(writer.fileno())
    os.chmod(destination, 0o444)


def verify_archive(folder, base, expected_model_hash=None, *, sealed=True):
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f'Archive is not a regular directory: {folder}')
    checksums = folder / 'SHA256SUMS'
    if checksums.is_symlink() or not checksums.is_file():
        raise ValueError('Archive checksum manifest is missing')
    entries = {}
    for line in checksums.read_text().splitlines():
        digest, name = line.split('  ', 1)
        if name not in ARCHIVE_FILES or name in entries:
            raise ValueError('Unexpected or duplicate archive checksum entry')
        entries[name] = digest
    if set(entries) != set(ARCHIVE_FILES):
        raise ValueError('Archive checksum manifest is incomplete')
    for name in (*ARCHIVE_FILES, 'SHA256SUMS'):
        path = folder / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Archive file missing or not regular: {name}')
        if sealed and stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ValueError(f'Archive is not sealed read-only: {name}')
        if name in entries and sha256(path) != entries[name]:
            raise ValueError(f'Archive checksum mismatch: {name}')
    metadata = json.loads((folder / 'snapshot.json').read_text())
    if (metadata.get('format') != 'warpstate-chat-v1' or
            metadata.get('step') != base['step'] or
            metadata.get('tokens_seen') != base['tokens_seen'] or
            metadata.get('tokenizer_sha256') != base['tokenizer_sha256']):
        raise ValueError('Archive metadata does not match the requested base boundary')
    if (metadata.get('snapshot_sha256') != entries['model.pt'] or
            sha256(folder / 'tokenizer.model') != base['tokenizer_sha256'] or
            metadata.get('snapshot_bytes') != (folder / 'model.pt').stat().st_size):
        raise ValueError('Archive model/tokenizer metadata mismatch')
    if expected_model_hash and entries['model.pt'] != expected_model_hash:
        raise ValueError('Existing archive differs from its recorded immutable identity')
    return metadata


def tensors_equal(original, exported):
    if original.keys() != exported.keys():
        raise ValueError('Archive model parameter names differ from the raw checkpoint')
    for name, value in original.items():
        other = exported[name]
        if value.dtype != other.dtype or value.shape != other.shape or not torch.equal(value, other):
            raise ValueError(f'Archive changed a raw model tensor: {name}')


def archive_checkpoint(source, tokenizer, destination, base, revision):
    """Called only while both base exclusion locks are held after clean exit."""
    source = Path(source)
    if source.is_symlink():
        raise ValueError('The base checkpoint must not be a symlink')
    before = source.stat()
    checkpoint = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    validate_boundary(checkpoint, base, sha256(tokenizer))
    if destination.exists() or destination.is_symlink():
        metadata = verify_archive(destination, base)
        exported = torch.load(destination / 'model.pt', map_location='cpu',
                              weights_only=False, mmap=True)
        tensors_equal(checkpoint['model'], exported['model'])
        return metadata
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix='.archive-76294-', dir=destination.parent))
    # On failure retain the hidden staging directory for diagnosis; never expose
    # a partially sealed archive under the requested historical archive name.
    exported_path = capture_latest(source, tokenizer, staging_root)
    staging = exported_path.parent
    exported = torch.load(exported_path, map_location='cpu', weights_only=False, mmap=True)
    tensors_equal(checkpoint['model'], exported['model'])
    full_hash = sha256(source)
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError('Base checkpoint changed while it was being archived')
    copy_regular(tokenizer, staging / 'tokenizer.model')
    note = f'''# KEEP: historical WarpState step {base['step']} weights

User-requested raw model archive created {utc_now()} before SFT.
Preserve this directory during cleanup and chat refreshes. Remove only when
the user explicitly requests removal of this historical archive.

- Completed update: {base['step']:,}; tokens trained: {base['tokens_seen']:,}.
- Base validation loss: {checkpoint['val_loss']:.12f}.
- Original FP32 model tensors exported by `chat.capture_latest`; every tensor
  was compared exactly with the stopped base checkpoint before publication.
- Inference/model weights only: no Adam or RNG state. The full base resume
  checkpoint is retained separately at `{source}`.
- Full source checkpoint SHA256: `{full_hash}`.
- Repository revision at archival time: `{revision}`.
- Independent physical files: no hardlinks or reflinks. All archive files
  are read-only. SFT reads an independent copy and writes its own run directory.
- This is an on-device archive, not an off-device backup.

Verify from this directory: `sha256sum -c SHA256SUMS`.
Files: model.pt, original snapshot.json, exact tokenizer.model, ARCHIVE.md,
and SHA256SUMS. The step 8K, 16K and 40K archives are preserved separately.
'''
    with (staging / 'ARCHIVE.md').open('x') as stream:
        stream.write(note)
        stream.flush()
        os.fsync(stream.fileno())
    with (staging / 'SHA256SUMS').open('x') as stream:
        for name in ARCHIVE_FILES:
            stream.write(f'{sha256(staging / name)}  {name}\n')
        stream.flush()
        os.fsync(stream.fileno())
    for name in (*ARCHIVE_FILES, 'SHA256SUMS'):
        os.chmod(staging / name, 0o444)
    os.chmod(staging, 0o555)
    flush_directory(staging)
    metadata = verify_archive(staging, base)
    publish_directory(staging, destination)
    staging_root.rmdir()
    return metadata


def independent_initialization(archive, destination, base, model_hash):
    verify_archive(archive, base, model_hash)
    if not destination.exists() and not destination.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.initialization-', dir=destination.parent))
        for name in (*ARCHIVE_FILES, 'SHA256SUMS'):
            copy_regular(archive / name, staging / name)
        verify_archive(staging, base, model_hash)
        flush_directory(staging)
        publish_directory(staging, destination)
    verify_archive(destination, base, model_hash)
    for name in (*ARCHIVE_FILES, 'SHA256SUMS'):
        original, copied = (archive / name).stat(), (destination / name).stat()
        if (original.st_dev, original.st_ino) == (copied.st_dev, copied.st_ino):
            raise ValueError(f'SFT initialization must not hardlink its raw archive: {name}')
    return destination / 'model.pt'


def data_readiness(folder, plan, *, verify=False):
    path = folder / 'manifest.json'
    if not path.is_file():
        return False, 'waiting for data manifest'
    manifest = json.loads(path.read_text())
    if manifest.get('format') != 'warpstate-posttrain-data-v1' or not manifest.get('complete'):
        return False, 'data preparation is incomplete'
    if manifest.get('tokenizer_sha256') != plan['base']['tokenizer_sha256']:
        raise ValueError('Prepared data uses a different tokenizer')
    if manifest.get('max_length') != plan['training']['sequence_length']:
        raise ValueError('Prepared data context length differs from the SFT plan')
    for name in ('sft_train.jsonl', 'sft_val.jsonl'):
        entry = manifest['outputs'].get(name, {})
        if entry.get('rows', 0) < 1 or not (folder / name).is_file():
            raise ValueError(f'Prepared SFT split is empty or missing: {name}')
        if verify and sha256(folder / name) != entry.get('sha256'):
            raise ValueError(f'Prepared SFT file checksum mismatch: {name}')
    return True, 'data manifest complete' if not verify else 'SFT data hashes verified'


def source_identity(config):
    paths = [config, PACKAGE / 'posttrain.py', PACKAGE / 'run_40b.py',
             PACKAGE / 'ops/posttrain_data.py', PACKAGE / 'ops/posttrain_handoff.py',
             PACKAGE / 'ops/launch_training.py', PACKAGE / 'ops/loss_experiment.py',
             PACKAGE / 'ops/residual_experiment.py', PACKAGE / 'ops/checkpoint_writer.py',
             PACKAGE / 'ops/posttrain_preflight.py', PACKAGE / 'ops/training-env.sh',
             PACKAGE / 'chat.py',
             *sorted((PACKAGE / 'warpstate').glob('*.py'))]
    return {str(path.relative_to(WORKSPACE)): sha256(path) for path in paths}


def verify_completion(work, plan, model_hash):
    """A zero exit can mean pause; only a matching durable final state completes."""
    completion = json.loads((work / 'completion.json').read_text())
    checkpoint = torch.load(work / 'latest.pt', map_location='cpu', weights_only=False, mmap=True)
    report = json.loads((work / 'evaluation.json').read_text())
    step, total = checkpoint['step'], checkpoint['total_steps']
    if (checkpoint.get('format') != 'warpstate-posttrain-v1' or
            checkpoint.get('completed') is not True or step != total or total < 1 or
            checkpoint.get('val_step') != step or not math.isfinite(checkpoint['val_loss'])):
        raise ValueError('SFT completion does not have a durable, fully evaluated final checkpoint')
    if (checkpoint['signature']['plan'] != plan or
            checkpoint['signature']['base_sha256'] != model_hash or
            checkpoint['origin']['step'] != plan['base']['step']):
        raise ValueError('SFT completion belongs to a different plan or initialization')
    if (completion.get('format') != 'warpstate-posttrain-completion-v1' or
            completion.get('stage') != 'sft' or completion.get('completed') is not True or
            completion.get('paused_for_evaluation') is not True or
            completion.get('step') != step or completion.get('total_steps') != total or
            completion.get('base_step') != plan['base']['step']):
        raise ValueError('SFT completion report differs from its checkpoint')
    if (report.get('stage') != 'sft' or report.get('step') != step or
            report.get('base_sha256') != model_hash or report.get('val_loss') != checkpoint['val_loss']):
        raise ValueError('SFT evaluation report differs from its completed checkpoint')
    return completion


def verify_preflight(path, processes):
    from posttrain import source_identity as training_sources
    report = json.loads(path.read_text())
    if (report.get('format') != 'warpstate-posttrain-preflight-v1' or
            report.get('accepted') is not True or report.get('processes') != processes or
            report.get('bf16_compute') is not True or
            report.get('fp32_parameters_and_adam') is not True or
            report.get('source_sha256') != training_sources() or
            report.get('generator_sha256') != sha256(PACKAGE / 'ops/posttrain_preflight.py')):
        raise ValueError('GPU preflight report is missing, failed, or belongs to different sources')
    return report


def load_plan(path):
    plan = json.loads(path.read_text())
    if (plan.get('format') != 'warpstate-posttrain-plan-v1' or
            plan.get('stage') != 'sft' or plan.get('after_completion') != 'pause_for_evaluation'):
        raise ValueError('This handoff supports SFT followed by an evaluation pause only')
    options = plan['handoff']
    paths = {key: workspace_path(WORKSPACE / options[key]) for key in (
        'base_work_dir', 'supervisor_socket', 'archive_dir', 'work_dir', 'data_dir', 'tokenizer')}
    base, work, archive = paths['base_work_dir'], paths['work_dir'], paths['archive_dir']
    if (work == base or work.is_relative_to(base / 'run') or
            work.is_relative_to(base / 'archives') or work.is_relative_to(archive)):
        raise ValueError('Posttraining must have a separate run directory outside the base run/archive')
    if (archive.is_relative_to(work) or paths['data_dir'].is_relative_to(work) or
            work.is_relative_to(paths['data_dir'])):
        raise ValueError('Archive and prepared data must stay outside the mutable posttraining run')
    if options['processes'] != 2 or not 1 <= options['poll_seconds'] <= 60:
        raise ValueError('Invalid process count or polling interval')
    return plan, paths


def request_stop(signum, frame):
    global STOPPED
    STOPPED = True


def run_handoff(config, *, check=False):
    plan, paths = load_plan(config)
    options, base = plan['handoff'], plan['base']
    work, base_work = paths['work_dir'], paths['base_work_dir']
    state_path = work / 'control/handoff.json'
    def status():
        return process_status(paths['supervisor_socket'], options['base_program'])
    if check:
        info = status()
        ready, reason = data_readiness(paths['data_dir'], plan)
        print(json.dumps(dict(base_process=info, target=base, data_ready=ready,
                              data_status=reason, state_path=str(state_path),
                              handoff=json.loads(state_path.read_text()) if state_path.exists() else None),
                         indent=2))
        return 0
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(work / 'control/handoff.lock') as handoff_fd:
        identity = source_identity(config)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['source_sha256'] != identity:
                raise ValueError('Armed handoff source/config changed; inspect and explicitly re-arm')
            if state.get('phase') == 'complete':
                print('SFT already completed; remaining paused for evaluation.', flush=True)
                return 0
        else:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORKSPACE,
                                               text=True).strip()
            state = dict(format='warpstate-posttrain-handoff-v1', armed_at=utc_now(),
                         source_sha256=identity, git_revision=revision, target=base,
                         archive_dir=str(paths['archive_dir']), work_dir=str(work))

        def record(phase, message, **extra):
            changed = state.get('phase') != phase or state.get('message') != message
            state.update(phase=phase, message=message, updated_at=utc_now(), **extra)
            atomic_json(state_path, state)
            if changed:
                print(f'[{utc_now()}] {phase}: {message}', flush=True)

        stop_path = work / 'control/training-stop-request'
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, request_stop)
        try:
            while not STOPPED:
                info = status()
                ready, reason = data_readiness(paths['data_dir'], plan)
                if not clean_exit(info):
                    record('waiting_for_base', f'Base {info["statename"]}; {reason}',
                           data_ready=ready)
                    time.sleep(options['poll_seconds'])
                    continue
                # Match the base launcher's outer -> inner lock order. Surviving
                # SFT ranks hold a SH pipeline lock, so a retry fails closed.
                try:
                    launcher_context = exclusive_lock(base_work / 'control/training-launcher.lock')
                    launcher_fd = launcher_context.__enter__()
                except BlockingIOError:
                    record('waiting_for_locks', 'Base launcher is still occupied')
                    time.sleep(options['poll_seconds'])
                    continue
                try:
                    with exclusive_lock(base_work / 'pipeline.lock'):
                        if not clean_exit(status()):
                            continue
                        checkpoint_path = base_work / 'run/latest.pt'
                        raw = torch.load(checkpoint_path, map_location='cpu',
                                         weights_only=False, mmap=True)
                        if raw.get('step', -1) < base['step']:
                            record('waiting_for_base', f'Base exited early at step {raw.get("step")}')
                            del raw
                            time.sleep(options['poll_seconds'])
                            continue
                        validate_boundary(raw, base, sha256(paths['tokenizer']))
                        del raw
                        if source_identity(config) != identity:
                            raise ValueError('Armed source/config changed before handoff')
                        record('archiving', 'Sealing original raw FP32 weights at the exact final step')
                        metadata = archive_checkpoint(checkpoint_path, paths['tokenizer'],
                                                      paths['archive_dir'], base, state['git_revision'])
                        model_hash = metadata['snapshot_sha256']
                        record('copying_initialization', 'Creating/verifying independent SFT initialization',
                               archive_sha256=model_hash)
                        weights = independent_initialization(paths['archive_dir'], work / 'initialization',
                                                             base, model_hash)
                        while not STOPPED:
                            ready, reason = data_readiness(paths['data_dir'], plan)
                            if ready:
                                break
                            record('waiting_for_data', reason)
                            time.sleep(options['poll_seconds'])
                        if STOPPED:
                            break
                        data_readiness(paths['data_dir'], plan, verify=True)
                        if source_identity(config) != identity:
                            raise ValueError('Armed source/config changed before SFT launch')
                        # Only clear a previous stop after proving there are no
                        # surviving workers that might still need to observe it.
                        stop_path.unlink(missing_ok=True)
                        atomic_json(work / 'control/execution.json', dict(
                            plan=plan, source_sha256=identity, git_revision=state['git_revision'],
                            archive_sha256=model_hash, data_manifest_sha256=sha256(
                                paths['data_dir'] / 'manifest.json')))
                    # Release pipeline EX before workers acquire SH. Launcher EX
                    # prevents a base launch in this transition. Pass both outer
                    # locks to torchrun so watcher death cannot duplicate launch.
                    if STOPPED:
                        break
                    environment = training_environment(work)
                    preflight_command = [sys.executable, '-B', str(PACKAGE / 'ops/posttrain_preflight.py'),
                                         '--work-dir', str(work / 'preflight'), '--base-work-dir',
                                         str(base_work), '--processes', str(options['processes'])]
                    record('gpu_preflight', 'Checking two-GPU BF16 SFT, masked tails, and checkpoint resume')
                    result = supervise(preflight_command, stop_path, environment,
                                       pass_fds=(handoff_fd, launcher_fd))
                    if result:
                        raise RuntimeError(f'GPU preflight exited with status {result}; full SFT was not started')
                    if stop_path.exists() or STOPPED:
                        record('paused', 'Handoff stopped during GPU preflight; full SFT was not started')
                        return 0
                    preflight = verify_preflight(work / 'preflight/report.json', options['processes'])
                    state['preflight'] = preflight
                    command = [sys.executable, '-B', '-m', 'torch.distributed.run',
                               '--standalone', '--nnodes', '1', '--nproc-per-node',
                               str(options['processes']), '--log-dir', str(work / 'logs/torchrun'),
                               str(PACKAGE / 'posttrain.py'), '--config', str(config),
                               '--work-dir', str(work / 'run'), '--data-dir', str(paths['data_dir']),
                               '--weights', str(weights), '--tokenizer', str(paths['tokenizer']),
                               '--stop-file', str(stop_path)]
                    record('sft', 'Starting/resuming supervised fine-tuning; fresh optimizer on first start',
                           command=command)
                    result = supervise(command, stop_path, environment,
                                       pass_fds=(handoff_fd, launcher_fd))
                    if result:
                        raise RuntimeError(f'SFT child exited with status {result}; preserved checkpoints for diagnosis')
                    # Completion is a trainer-published artifact, never inferred
                    # from return code alone (graceful early stops also return 0).
                    completion = work / 'run/completion.json'
                    if completion.is_file():
                        completed = verify_completion(work / 'run', plan, model_hash)
                        verify_archive(paths['archive_dir'], base, model_hash)
                        if sha256(weights) != model_hash:
                            raise ValueError('SFT initialization changed during training')
                        record('complete', 'SFT finished; paused for evaluation. DPO has not been started.',
                               completion=completed)
                    else:
                        record('paused', 'SFT stopped before completion; its independent resume checkpoint is retained')
                    return 0
                finally:
                    launcher_context.__exit__(None, None, None)
            record('paused', 'Handoff paused by request; no further training will start automatically')
            return 0
        except BaseException as error:
            record('failed', f'{type(error).__name__}: {error}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Read-only status; do not arm or launch')
    args = parser.parse_args()
    return run_handoff(workspace_path(args.config), check=args.check)


if __name__ == '__main__':
    sys.exit(main())
