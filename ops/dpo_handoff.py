#!/usr/bin/env python3
"""Archive a successfully completed SFT run, then initialize separate BF16 DPO.

This explicitly authorized follow-on job does not modify or interrupt SFT. Its
archive includes the original full optimizer/RNG checkpoint and a tensor-exact
model-only export. DPO's policy and fixed reference read an independent copy.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))
import torch
from ops.launch_training import supervise, training_environment, workspace_path
from ops.posttrain_handoff import (atomic_json, clean_exit, copy_regular, exclusive_lock,
    flush_directory, process_status, publish_directory, sha256, tensors_equal,
    utc_now, verify_completion as verify_sft_completion,
    source_identity as sft_source_identity)

STOPPED = False
ARCHIVE_FILES = ('checkpoint.pt', 'model.pt', 'snapshot.json', 'tokenizer.model',
                 'settings.json', 'evaluation.json', 'completion.json', 'metrics.jsonl',
                 'sft-execution.json', 'sft-handoff.json', 'PROVENANCE.json', 'ARCHIVE.md')
INIT_FILES = ('model.pt', 'snapshot.json', 'tokenizer.model')


def read_json(path):
    return json.loads(Path(path).read_text())


def load_plan(path):
    plan = read_json(path)
    if (plan.get('format') != 'warpstate-dpo-plan-v1' or plan.get('stage') != 'dpo' or
            plan.get('after_completion') != 'pause_for_evaluation'):
        raise ValueError('Expected a DPO plan that pauses after completion')
    options = plan['handoff']
    paths = {key: workspace_path(WORKSPACE / options[key]) for key in (
        'base_work_dir', 'sft_work_dir', 'sft_plan', 'supervisor_socket', 'archive_dir',
        'work_dir', 'data_dir', 'tokenizer')}
    work, base, sft, archive, data = (paths[k] for k in (
        'work_dir', 'base_work_dir', 'sft_work_dir', 'archive_dir', 'data_dir'))
    if work == base or work == WORKSPACE or not work.is_relative_to(base / 'posttrain'):
        raise ValueError('DPO output must use its own directory under base work/posttrain')
    for protected in (base / 'run', base / 'archives', sft, archive, data):
        if work.is_relative_to(protected) or protected.is_relative_to(work):
            raise ValueError(f'DPO outputs overlap protected inputs: {protected}')
    if not archive.is_relative_to(base / 'archives') or archive == base / 'archives':
        raise ValueError('Select a new historical archive directory under base work/archives')
    if options['processes'] != 2 or not 1 <= options['poll_seconds'] <= 60:
        raise ValueError('This handoff requires two GPUs and a 1..60 second poll interval')
    original = read_json(paths['sft_plan'])
    if (original['base']['step'] != plan['sft']['base_step'] or
            original['base']['tokens_seen'] != plan['sft']['base_tokens_seen'] or
            original['base']['tokenizer_sha256'] != plan['sft']['tokenizer_sha256']):
        raise ValueError('SFT parent identity differs from the DPO plan')
    return plan, paths


def source_identity(config, sft_plan):
    from dpo import source_identity as dpo_sources
    result = {str((PACKAGE / name).relative_to(WORKSPACE)): checksum
              for name, checksum in dpo_sources().items()}
    result.update(sft_source_identity(sft_plan))
    for path in (config, Path(__file__), PACKAGE / 'ops/dpo_preflight.py',
                 PACKAGE / 'ops/dpo-handoff.sh'):
        result[str(path.relative_to(WORKSPACE))] = sha256(path)
    return result


def data_readiness(folder, plan, *, verify=False):
    manifest = folder / 'manifest.json'
    if not manifest.is_file():
        return False
    data = read_json(manifest)
    if data.get('format') != 'warpstate-posttrain-data-v1' or data.get('complete') is not True:
        return False
    if (sha256(manifest) != plan['handoff']['data_manifest_sha256'] or
            data.get('tokenizer_sha256') != plan['sft']['tokenizer_sha256'] or
            data.get('max_length') != plan['training']['sequence_length']):
        raise ValueError('Prepared preference data differs from the authorized plan')
    for name in ('dpo_train.jsonl', 'dpo_val.jsonl'):
        entry = data.get('outputs', {}).get(name, {})
        file = folder / name
        if not file.is_file() or file.is_symlink() or entry.get('rows', 0) < 1:
            raise ValueError(f'Prepared preference split is empty or missing: {name}')
        if verify and sha256(file) != entry.get('sha256'):
            raise ValueError(f'Preference split checksum differs: {name}')
    return True


def validate_sft(checkpoint, evaluation, completion, plan, sft_plan):
    expected = plan['sft']
    if (checkpoint.get('format') != 'warpstate-posttrain-v1' or
            checkpoint.get('completed') is not True or
            checkpoint.get('step') != expected['step'] or
            checkpoint.get('total_steps') != expected['step'] or
            checkpoint.get('sample_cursor') != expected['train_rows'] or
            checkpoint.get('tokens_seen') != expected['tokens_seen'] or
            checkpoint.get('assistant_tokens_seen') != expected['assistant_tokens_seen'] or
            checkpoint.get('val_step') != expected['step']):
        raise ValueError('SFT must complete the exact full epoch and final validation before DPO')
    if not math.isfinite(checkpoint.get('val_loss', float('nan'))):
        raise ValueError('SFT final validation loss is not finite')
    signature = checkpoint['signature']
    if (signature['plan'] != sft_plan or signature['base_sha256'] != expected['raw_archive_sha256'] or
            signature['tokenizer_sha256'] != expected['tokenizer_sha256'] or
            signature['data_sha256'] != plan['handoff']['data_manifest_sha256'] or
            checkpoint['origin']['step'] != expected['base_step'] or
            checkpoint['origin']['tokens_seen'] != expected['base_tokens_seen']):
        raise ValueError('Completed SFT checkpoint belongs to a different base, tokenizer, data, or plan')
    if not checkpoint.get('optimizer') or len(checkpoint.get('rng_by_rank', [])) != 2:
        raise ValueError('Full SFT optimizer or per-rank resume state is missing')
    if (completion.get('format') != 'warpstate-posttrain-completion-v1' or
            completion.get('stage') != 'sft' or completion.get('completed') is not True or
            completion.get('step') != expected['step'] or completion.get('total_steps') != expected['step']):
        raise ValueError('SFT completion report is missing or inconsistent')
    if (evaluation.get('stage') != 'sft' or evaluation.get('step') != expected['step'] or
            evaluation.get('full_holdout') is not True or
            evaluation.get('evaluated_samples') != expected['validation_rows'] or
            evaluation.get('available_validation_samples') != expected['validation_rows'] or
            evaluation.get('val_loss') != checkpoint['val_loss'] or
            evaluation.get('base_sha256') != expected['raw_archive_sha256']):
        raise ValueError('SFT archive requires the matching full-holdout evaluation')
    if not checkpoint.get('model'):
        raise ValueError('SFT model tensors are missing')
    for name, tensor in checkpoint['model'].items():
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f'Expected untouched FP32 SFT weights: {name}')


def verify_files(folder, names):
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError(f'Artifact directory is missing or is a symlink: {folder}')
    checksums = folder / 'SHA256SUMS'
    if not checksums.is_file() or checksums.is_symlink():
        raise ValueError('Artifact checksum manifest is missing')
    entries = {}
    for line in checksums.read_text().splitlines():
        checksum, name = line.split('  ', 1)
        if name not in names or name in entries:
            raise ValueError('Unexpected/duplicate artifact checksum entry')
        entries[name] = checksum
    if set(entries) != set(names):
        raise ValueError('Artifact checksum manifest is incomplete')
    for name in (*names, 'SHA256SUMS'):
        path = folder / name
        if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ValueError(f'Artifact file is missing or not sealed read-only: {name}')
        if name in entries and sha256(path) != entries[name]:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    return entries


def seal(folder, names):
    checksums = folder / 'SHA256SUMS'
    with checksums.open('x') as stream:
        for name in names:
            stream.write(f'{sha256(folder / name)}  {name}\n')
        stream.flush()
        os.fsync(stream.fileno())
    for name in (*names, 'SHA256SUMS'):
        os.chmod(folder / name, 0o444)
    os.chmod(folder, 0o555)
    flush_directory(folder)


def verify_sft_archive(folder, plan, sft_plan, *, expected_checkpoint_hash=None):
    entries = verify_files(folder, ARCHIVE_FILES)
    if expected_checkpoint_hash and entries['checkpoint.pt'] != expected_checkpoint_hash:
        raise ValueError('Archived SFT state differs from the completed source checkpoint')
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
    validate_sft(checkpoint, read_json(folder / 'evaluation.json'), read_json(folder / 'completion.json'), plan, sft_plan)
    metadata = read_json(folder / 'snapshot.json')
    if (metadata.get('format') != 'warpstate-chat-v1' or metadata.get('stage') != 'sft' or
            metadata.get('step') != plan['sft']['step'] or
            metadata.get('tokens_seen') != checkpoint['tokens_seen'] or
            metadata.get('tokenizer_sha256') != plan['sft']['tokenizer_sha256'] or
            metadata.get('snapshot_sha256') != entries['model.pt'] or
            metadata.get('snapshot_bytes') != (folder / 'model.pt').stat().st_size or
            metadata.get('source', {}).get('sha256') != entries['checkpoint.pt'] or
            entries['tokenizer.model'] != plan['sft']['tokenizer_sha256']):
        raise ValueError('SFT model export metadata differs from the full archive')
    exported = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False, mmap=True)
    tensors_equal(checkpoint['model'], exported['model'])
    return metadata, entries


def archive_sft(sft_work, tokenizer, destination, plan, sft_plan, revision):
    run = sft_work / 'run'
    source = run / 'latest.pt'
    if source.is_symlink():
        raise ValueError('SFT checkpoint must be a regular completed file')
    before = source.stat()
    checkpoint = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    evaluation, completion = read_json(run / 'evaluation.json'), read_json(run / 'completion.json')
    validate_sft(checkpoint, evaluation, completion, plan, sft_plan)
    full_hash = sha256(source)
    if destination.exists() or destination.is_symlink():
        return verify_sft_archive(destination, plan, sft_plan, expected_checkpoint_hash=full_hash)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.sft-final-', dir=destination.parent))
    # Full-state archive is a byte-for-byte copy, preserving Adam/RNG/signature.
    copy_regular(source, staging / 'checkpoint.pt')
    if sha256(staging / 'checkpoint.pt') != full_hash:
        raise ValueError('SFT full-state archive copy differs from its source')
    metadata = dict(format='warpstate-chat-v1', inference_only=True, stage='sft',
        config=checkpoint['config'], step=checkpoint['step'], tokens_seen=checkpoint['tokens_seen'],
        assistant_tokens_seen=checkpoint['assistant_tokens_seen'], val_loss=checkpoint['val_loss'],
        tokenizer_sha256=plan['sft']['tokenizer_sha256'], origin=checkpoint['origin'],
        captured_at_utc=utc_now(), source=dict(path=str(source), device=before.st_dev,
            inode=before.st_ino, size=before.st_size, mtime_ns=before.st_mtime_ns, sha256=full_hash))
    with (staging / 'model.pt').open('xb') as stream:
        torch.save(dict(metadata, model=checkpoint['model']), stream)
        stream.flush()
        os.fsync(stream.fileno())
    metadata.update(snapshot_sha256=sha256(staging / 'model.pt'),
                    snapshot_bytes=(staging / 'model.pt').stat().st_size)
    atomic_json(staging / 'snapshot.json', metadata)
    copy_regular(tokenizer, staging / 'tokenizer.model')
    for name in ('settings.json', 'evaluation.json', 'completion.json', 'metrics.jsonl'):
        copy_regular(run / name, staging / name)
    copy_regular(sft_work / 'control/execution.json', staging / 'sft-execution.json')
    copy_regular(sft_work / 'control/handoff.json', staging / 'sft-handoff.json')
    atomic_json(staging / 'PROVENANCE.json', dict(format='warpstate-sft-archive-v1',
        repository_revision=revision, source_checkpoint_sha256=full_hash,
        model_sha256=metadata['snapshot_sha256'], sft_step=checkpoint['step'],
        sft_input_tokens=checkpoint['tokens_seen'], sft_assistant_tokens=checkpoint['assistant_tokens_seen'],
        raw_base_step=plan['sft']['base_step'], raw_base_tokens=plan['sft']['base_tokens_seen'],
        raw_base_sha256=plan['sft']['raw_archive_sha256'], data_sha256=checkpoint['signature']['data_sha256'],
        tokenizer_sha256=plan['sft']['tokenizer_sha256'], sft_signature=checkpoint['signature']))
    with (staging / 'ARCHIVE.md').open('x') as stream:
        stream.write(f'''# KEEP: completed SFT after raw step {plan['sft']['base_step']}

User-requested historical archive created {utc_now()}. Preserve during routine
cleanup, DPO, and chat refreshes. Remove only on explicit user instruction.

- Completed SFT update: {checkpoint['step']:,}; SFT input tokens: {checkpoint['tokens_seen']:,}.
- Supervised assistant tokens: {checkpoint['assistant_tokens_seen']:,}.
- Original raw base: step {plan['sft']['base_step']:,}, {plan['sft']['base_tokens_seen']:,} tokens.
- Final full-holdout SFT loss: {checkpoint['val_loss']:.12f}.
- `checkpoint.pt` is the untouched complete SFT checkpoint: FP32 model, Adam,
  both ranks' RNG, sample cursor, configuration and source/data signatures.
- `model.pt` is a tensor-exact FP32 model-only export with original tokenizer.
- Original evaluation, completion, settings, metrics and handoff provenance are
  included. SFT and pretraining counters refer to distinct training stages.
- Independent physical files, no hardlinks/reflinks; every file is read-only.
- DPO uses an independent model copy and fresh Adam; its reference stays fixed
  at this SFT model. The original SFT run checkpoint and raw archive are kept.
- Repository revision at archival time: `{revision}`.

Verify here with `sha256sum -c SHA256SUMS`. This is an on-device archive.
''')
        stream.flush()
        os.fsync(stream.fileno())
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('SFT checkpoint changed while archival was in progress')
    seal(staging, ARCHIVE_FILES)
    verified = verify_sft_archive(staging, plan, sft_plan, expected_checkpoint_hash=full_hash)
    publish_directory(staging, destination)
    return verified


def initialize_copy(archive, destination, metadata):
    if not destination.exists() and not destination.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.dpo-init-', dir=destination.parent))
        for name in INIT_FILES:
            copy_regular(archive / name, staging / name)
        seal(staging, INIT_FILES)
        publish_directory(staging, destination)
    entries = verify_files(destination, INIT_FILES)
    if entries['model.pt'] != metadata['snapshot_sha256']:
        raise ValueError('DPO initialization differs from the sealed SFT model')
    for name in INIT_FILES:
        original, copied = (archive / name).stat(), (destination / name).stat()
        if ((original.st_dev, original.st_ino) == (copied.st_dev, copied.st_ino) or
                sha256(archive / name) != entries[name]):
            raise ValueError(f'DPO initialization must be an independent identical copy: {name}')
    return destination / 'model.pt'


def verify_preflight(path, processes):
    from dpo import source_identity as dpo_sources
    report = read_json(path)
    if (report.get('format') != 'warpstate-dpo-preflight-v1' or report.get('accepted') is not True or
            report.get('processes') != processes or report.get('source_sha256') != dpo_sources() or
            report.get('bf16_compute') is not True or
            report.get('fp32_parameters_and_adam') is not True or
            report.get('frozen_reference_unchanged') is not True or
            report.get('generator_sha256') != sha256(PACKAGE / 'ops/dpo_preflight.py')):
        raise ValueError('DPO GPU preflight failed or belongs to different sources')
    return report


def verify_dpo_completion(run, plan, model_hash):
    from dpo import DEFAULTS, reference_identity, validate_cache
    completion = read_json(run / 'completion.json')
    evaluation = read_json(run / 'evaluation.json')
    checkpoint = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False, mmap=True)
    manifest_path = WORKSPACE / plan['handoff']['data_dir'] / 'manifest.json'
    manifest = read_json(manifest_path)
    if sha256(manifest_path) != plan['handoff']['data_manifest_sha256']:
        raise ValueError('DPO data manifest changed before completion')
    training, validation = (manifest['outputs'][name] for name in ('dpo_train.jsonl', 'dpo_val.jsonl'))
    settings = DEFAULTS | plan['training']
    batch = settings['micro_batch'] * settings['grad_accum'] * plan['handoff']['processes']
    total_steps = math.ceil(training['rows'] / batch) * settings['epochs']
    sample_count = training['rows'] * settings['epochs']
    expected_tokens = (training['tokens'] - 2 * training['rows']) * settings['epochs']
    expected_targets = training['supervised_tokens'] * settings['epochs']
    if (checkpoint.get('format') != 'warpstate-dpo-v1' or checkpoint.get('completed') is not True or
            checkpoint.get('step') != total_steps or checkpoint.get('total_steps') != total_steps or
            checkpoint.get('sample_cursor') != sample_count or checkpoint.get('tokens_seen') != expected_tokens or
            checkpoint.get('response_tokens_seen') != expected_targets or
            checkpoint.get('val_step') != checkpoint['step'] or
            not math.isfinite(checkpoint.get('val_loss', float('nan'))) or
            checkpoint['signature']['plan'] != plan or checkpoint['signature']['base_sha256'] != model_hash or
            checkpoint['signature']['settings'] != settings or
            checkpoint['signature']['data_sha256'] != plan['handoff']['data_manifest_sha256']):
        raise ValueError('DPO did not publish a matching fully completed checkpoint')
    if (completion.get('format') != 'warpstate-dpo-completion-v1' or completion.get('stage') != 'dpo' or
            completion.get('completed') is not True or completion.get('paused_for_evaluation') is not True or
            any(completion.get(key) != checkpoint[key] for key in ('step', 'total_steps', 'sample_cursor',
                'tokens_seen', 'response_tokens_seen', 'val_loss'))):
        raise ValueError('DPO completion report differs from its checkpoint')
    if (evaluation.get('stage') != 'dpo' or evaluation.get('step') != checkpoint['step'] or
            evaluation.get('full_holdout') is not True or
            evaluation.get('evaluated_samples') != validation['rows'] or
            evaluation.get('available_validation_samples') != validation['rows'] or
            evaluation.get('val_loss') != checkpoint['val_loss'] or
            evaluation.get('base_sha256') != model_hash or
            evaluation.get('data_sha256') != checkpoint['signature']['data_sha256'] or
            not all(math.isfinite(evaluation.get(key, float('nan'))) for key in
                ('reward_margin', 'reward_accuracy', 'chosen_nll'))):
        raise ValueError('DPO final full-holdout evaluation is missing')
    identity = reference_identity(model_hash, plan['handoff']['data_manifest_sha256'],
        plan['sft']['tokenizer_sha256'], settings, torch.device('cuda'), plan['handoff']['processes'])
    _, _, reference_hash = validate_cache(run.parent / 'reference-cache', identity,
                                          (training['rows'], validation['rows']))
    if any(value != reference_hash for value in (checkpoint['signature']['reference_sha256'],
        completion.get('reference_sha256'), evaluation.get('reference_sha256'))):
        raise ValueError('DPO completion no longer matches its fixed SFT reference cache')
    return completion


def request_stop(signum, frame):
    global STOPPED
    STOPPED = True


def run_handoff(config, *, check=False):
    plan, paths = load_plan(config)
    work, sft, base = (paths[k] for k in ('work_dir', 'sft_work_dir', 'base_work_dir'))
    state_path = work / 'control/handoff.json'
    original_plan = read_json(paths['sft_plan'])
    def status():
        return process_status(paths['supervisor_socket'], plan['handoff']['sft_program'])
    if check:
        print(json.dumps(dict(sft_process=status(), data_ready=data_readiness(paths['data_dir'], plan),
            handoff=read_json(state_path) if state_path.exists() else None), indent=2))
        return 0
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(work / 'control/handoff.lock') as own_fd:
        identity = source_identity(config, paths['sft_plan'])
        if state_path.exists():
            state = read_json(state_path)
            if state['source_sha256'] != identity:
                raise ValueError('Armed DPO sources/config changed; inspect before explicitly re-arming')
            if state.get('phase') == 'complete':
                print('DPO already completed; remaining paused for evaluation.', flush=True)
                return 0
        else:
            state = dict(format='warpstate-dpo-handoff-v1', armed_at=utc_now(),
                         source_sha256=identity, sft=plan['sft'],
                         git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                             cwd=WORKSPACE, text=True).strip())
        def record(phase, message, **extra):
            changed = state.get('phase') != phase or state.get('message') != message
            state.update(phase=phase, message=message, updated_at=utc_now(), **extra)
            atomic_json(state_path, state)
            if changed:
                print(f'[{utc_now()}] {phase}: {message}', flush=True)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            signal.signal(sig, request_stop)
        stop_file = work / 'control/training-stop-request'
        try:
            while not STOPPED:
                info = status()
                ready = data_readiness(paths['data_dir'], plan)
                prior = read_json(sft / 'control/handoff.json')
                if not clean_exit(info) or prior.get('phase') != 'complete':
                    record('waiting_for_sft', f'SFT {info["statename"]}; final state is not yet complete', data_ready=ready)
                    time.sleep(plan['handoff']['poll_seconds'])
                    continue
                with contextlib.ExitStack() as locks:
                    # Identical outer-to-inner order to the SFT guardian. Any
                    # surviving child owns a SH pipeline lock and blocks retries.
                    try:
                        sft_fd = locks.enter_context(exclusive_lock(sft / 'control/handoff.lock'))
                        launcher_fd = locks.enter_context(exclusive_lock(base / 'control/training-launcher.lock'))
                        pipeline = exclusive_lock(base / 'pipeline.lock')
                        pipeline.__enter__()
                    except BlockingIOError:
                        record('waiting_for_locks', 'A training worker or launcher still owns an exclusion lock')
                        time.sleep(plan['handoff']['poll_seconds'])
                        continue
                    try:
                        if not clean_exit(status()) or read_json(sft / 'control/handoff.json').get('phase') != 'complete':
                            continue
                        if source_identity(config, paths['sft_plan']) != identity:
                            raise ValueError('Armed source identity changed before SFT archival')
                        verify_sft_completion(sft / 'run', original_plan, plan['sft']['raw_archive_sha256'])
                        record('archiving_sft', 'Sealing the full final SFT checkpoint and original model tensors')
                        metadata, checksums = archive_sft(sft, paths['tokenizer'], paths['archive_dir'],
                            plan, original_plan, state['git_revision'])
                        record('copying_initialization', 'Creating/verifying independent policy and fixed-reference initialization',
                            archive_dir=str(paths['archive_dir']), archive_model_sha256=metadata['snapshot_sha256'],
                            archive_checkpoint_sha256=checksums['checkpoint.pt'])
                        weights = initialize_copy(paths['archive_dir'], work / 'initialization', metadata)
                        while not STOPPED and not data_readiness(paths['data_dir'], plan):
                            record('waiting_for_data', 'Waiting for completed pinned preference data')
                            time.sleep(plan['handoff']['poll_seconds'])
                        if STOPPED:
                            break
                        data_readiness(paths['data_dir'], plan, verify=True)
                        if source_identity(config, paths['sft_plan']) != identity:
                            raise ValueError('Armed source identity changed before DPO launch')
                        stop_file.unlink(missing_ok=True)
                        atomic_json(work / 'control/execution.json', dict(plan=plan, source_sha256=identity,
                            sft_model_sha256=metadata['snapshot_sha256'], sft_checkpoint_sha256=checksums['checkpoint.pt'],
                            git_revision=state['git_revision']))
                    finally:
                        pipeline.__exit__(None, None, None)
                    if STOPPED:
                        break
                    environment = training_environment(work)
                    inherited = (own_fd, sft_fd, launcher_fd)
                    preflight_command = [sys.executable, '-B', str(PACKAGE / 'ops/dpo_preflight.py'),
                        '--work-dir', str(work / 'preflight'), '--base-work-dir', str(base), '--processes', '2']
                    record('gpu_preflight', 'Checking BF16 DPO, frozen reference scoring, pair masks, and resume')
                    result = supervise(preflight_command, stop_file, environment, pass_fds=inherited)
                    if result:
                        raise RuntimeError(f'DPO GPU preflight failed with exit status {result}')
                    if STOPPED or stop_file.exists():
                        record('paused', 'Paused during preflight; full DPO was not started')
                        return 0
                    state['preflight'] = verify_preflight(work / 'preflight/report.json', 2)
                    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
                        '--nnodes', '1', '--nproc-per-node', '2', '--log-dir', str(work / 'logs/torchrun'),
                        str(PACKAGE / 'dpo.py'), '--config', str(config), '--work-dir', str(work / 'run'),
                        '--data-dir', str(paths['data_dir']), '--weights', str(weights),
                        '--tokenizer', str(paths['tokenizer']), '--reference-cache-dir', str(work / 'reference-cache'),
                        '--stop-file', str(stop_file)]
                    record('dpo', 'Preparing frozen SFT reference scores, then starting/resuming DPO', command=command)
                    result = supervise(command, stop_file, environment, pass_fds=inherited)
                    if result:
                        raise RuntimeError(f'DPO exited with status {result}; independent checkpoints are retained')
                    if (work / 'run/completion.json').exists():
                        completed = verify_dpo_completion(work / 'run', plan, metadata['snapshot_sha256'])
                        verify_sft_archive(paths['archive_dir'], plan, original_plan,
                            expected_checkpoint_hash=checksums['checkpoint.pt'])
                        if sha256(weights) != metadata['snapshot_sha256']:
                            raise ValueError('DPO changed its original SFT initialization')
                        record('complete', 'DPO finished and paused for evaluation; the archived SFT state is unchanged',
                               completion=completed)
                    else:
                        record('paused', 'DPO/reference preparation paused; resume state is retained')
                    return 0
            record('paused', 'DPO handoff paused by request; no further job will start')
            return 0
        except BaseException as error:
            record('failed', f'{type(error).__name__}: {error}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    return run_handoff(workspace_path(args.config), check=args.check)


if __name__ == '__main__':
    sys.exit(main())
