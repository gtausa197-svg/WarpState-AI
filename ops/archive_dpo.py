#!/usr/bin/env python3
"""Seal a completed DPO run as independent, read-only full-state/model archives.

Run from /workspace/WarpState after the DPO guardian exits successfully. The
default destination is work/archives/step-00076294-dpo-final. Existing archives
are verified and reused; they are never replaced or modified.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))

import numpy as np
import torch

from dpo import DEFAULTS, identity_digest
from ops.dpo_handoff import load_plan, read_json, seal, verify_files, verify_dpo_completion
from ops.launch_training import workspace_path
from ops.posttrain_handoff import (atomic_json, clean_exit, copy_regular, exclusive_lock,
    process_status, publish_directory, sha256, tensors_equal, utc_now)

FILES = ('checkpoint.pt', 'model.pt', 'snapshot.json', 'tokenizer.model', 'plan.json',
         'settings.json', 'evaluation.json', 'completion.json', 'metrics.jsonl',
         'dpo-execution.json', 'dpo-handoff.json', 'gpu-preflight.json',
         'data-manifest.json', 'reference-manifest.json', 'reference-train.npy',
         'reference-val.npy', 'PROVENANCE.json', 'ARCHIVE.md')


def validate_state(checkpoint, evaluation, completion, settings_report, data, reference, plan):
    """Validate an archive using its own preserved records, without a live run."""
    settings = DEFAULTS | plan['training']
    processes = plan['handoff']['processes']
    train, val = (data['outputs'][name] for name in ('dpo_train.jsonl', 'dpo_val.jsonl'))
    total_steps = math.ceil(train['rows'] / (settings['micro_batch'] * settings['grad_accum'] * processes)) * settings['epochs']
    if settings['max_steps'] is not None:
        raise ValueError('Archive requires a completed whole DPO epoch, without a plan step cap')
    expected = dict(step=total_steps, total_steps=total_steps,
        sample_cursor=train['rows'] * settings['epochs'],
        tokens_seen=(train['tokens'] - 2 * train['rows']) * settings['epochs'],
        response_tokens_seen=train['supervised_tokens'] * settings['epochs'], val_step=total_steps)
    if (checkpoint.get('format') != 'warpstate-dpo-v1' or checkpoint.get('completed') is not True or
            any(checkpoint.get(key) != value for key, value in expected.items()) or
            not math.isfinite(checkpoint.get('val_loss', float('nan')))):
        raise ValueError('DPO checkpoint is not the fully completed, finally evaluated state')
    signature = checkpoint['signature']
    if (signature.get('plan') != plan or signature.get('settings') != settings or
            signature.get('world_size') != processes or
            signature.get('tokenizer_sha256') != plan['sft']['tokenizer_sha256'] or
            signature.get('data_sha256') != plan['handoff']['data_manifest_sha256'] or
            settings_report.get('signature') != signature or
            settings_report.get('total_steps') != total_steps or
            settings_report.get('train_rows') != train['rows'] or
            settings_report.get('val_rows') != val['rows']):
        raise ValueError('DPO checkpoint/settings signatures or dataset counts differ')
    if (data.get('format') != 'warpstate-posttrain-data-v1' or data.get('complete') is not True or
            data.get('tokenizer_sha256') != signature['tokenizer_sha256'] or
            data.get('max_length') != settings['sequence_length']):
        raise ValueError('DPO data provenance differs from its signature')
    origin = checkpoint['origin']
    if (origin.get('stage') != 'sft' or origin.get('step') != plan['sft']['step'] or
            origin.get('weights_sha256') != signature['base_sha256'] or
            origin.get('tokens_seen') != plan['sft']['tokens_seen'] or
            origin.get('origin', {}).get('step') != plan['sft']['base_step'] or
            origin.get('origin', {}).get('tokens_seen') != plan['sft']['base_tokens_seen']):
        raise ValueError('DPO SFT/raw ancestry differs from the authorized plan')
    if (not checkpoint.get('optimizer', {}).get('state') or
            len(checkpoint.get('rng_by_rank', [])) != processes or
            any('torch' not in value or 'cuda' not in value for value in checkpoint['rng_by_rank'])):
        raise ValueError('DPO optimizer or per-rank RNG state is missing')
    for value in checkpoint['optimizer']['state'].values():
        if float(value['step']) != total_steps:
            raise ValueError('DPO Adam step differs from the final checkpoint')
        for name in ('exp_avg', 'exp_avg_sq'):
            if value[name].dtype != torch.float32:
                raise ValueError('DPO Adam state must retain its original FP32 tensors')
    if not checkpoint.get('model'):
        raise ValueError('DPO model tensors are missing')
    for name, tensor in checkpoint['model'].items():
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f'DPO model tensor is not original FP32: {name}')
    counters = ('step', 'total_steps', 'sample_cursor', 'tokens_seen', 'response_tokens_seen', 'val_loss')
    if (completion.get('format') != 'warpstate-dpo-completion-v1' or completion.get('stage') != 'dpo' or
            completion.get('completed') is not True or completion.get('paused_for_evaluation') is not True or
            any(completion.get(key) != checkpoint[key] for key in counters)):
        raise ValueError('DPO completion report differs from its checkpoint')
    if (evaluation.get('stage') != 'dpo' or evaluation.get('step') != total_steps or
            evaluation.get('full_holdout') is not True or evaluation.get('evaluated_samples') != val['rows'] or
            evaluation.get('available_validation_samples') != val['rows'] or
            evaluation.get('objective') != settings['objective'] or
            any(evaluation.get(key) != checkpoint['validation'].get(key) for key in
                ('val_loss', 'reward_margin', 'reward_accuracy', 'chosen_nll', 'evaluated_samples')) or
            any(evaluation.get(key) != signature[key] for key in
                ('base_sha256', 'data_sha256', 'tokenizer_sha256', 'reference_sha256')) or
            not all(math.isfinite(evaluation.get(key, float('nan'))) for key in
                ('val_loss', 'reward_margin', 'reward_accuracy', 'chosen_nll'))):
        raise ValueError('DPO final full-holdout evaluation differs from its checkpoint')
    if not 0 <= evaluation['reward_accuracy'] <= 1 or evaluation['chosen_nll'] < 0:
        raise ValueError('DPO evaluation contains impossible accuracy/NLL')
    identity = reference.get('identity', {})
    if (reference.get('format') != 'warpstate-dpo-reference-v1' or reference.get('complete') is not True or
            reference.get('identity_sha256') != identity_digest(identity) or
            identity.get('weights_sha256') != signature['base_sha256'] or
            identity.get('data_sha256') != signature['data_sha256'] or
            identity.get('tokenizer_sha256') != signature['tokenizer_sha256'] or
            identity.get('sources') != signature['sources'] or
            identity.get('renderer_sha256') != data.get('renderer_sha256') or
            identity.get('mask_format') != signature['mask'] or
            identity.get('device_type') != signature['device_type'] or
            identity.get('world_size') != processes or
            identity.get('torch_version') != signature['torch_version'] or
            any(value != settings.get(key) for key, value in identity.get('settings', {}).items()) or
            completion.get('reference_sha256') != signature['reference_sha256']):
        raise ValueError('DPO fixed-reference identity differs from its final checkpoint')
    return expected


def verify_dpo_archive(folder, *, expected_checkpoint_hash=None):
    folder = Path(folder)
    entries = verify_files(folder, FILES)
    if stat.S_IMODE(folder.stat().st_mode) & 0o222 or {path.name for path in folder.iterdir()} != set(FILES) | {'SHA256SUMS'}:
        raise ValueError('DPO archive directory is not sealed or contains unexpected files')
    if expected_checkpoint_hash and entries['checkpoint.pt'] != expected_checkpoint_hash:
        raise ValueError('Archived DPO state differs from the source checkpoint')
    checkpoint = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
    plan, data = read_json(folder / 'plan.json'), read_json(folder / 'data-manifest.json')
    reference = read_json(folder / 'reference-manifest.json')
    validate_state(checkpoint, read_json(folder / 'evaluation.json'), read_json(folder / 'completion.json'),
                   read_json(folder / 'settings.json'), data, reference, plan)
    if (entries['data-manifest.json'] != checkpoint['signature']['data_sha256'] or
            entries['reference-manifest.json'] != checkpoint['signature']['reference_sha256'] or
            entries['tokenizer.model'] != checkpoint['signature']['tokenizer_sha256']):
        raise ValueError('Archived data/reference/tokenizer hashes differ from checkpoint signatures')
    for split in ('train', 'val'):
        entry = reference['files'][split + '.npy']
        name = f'reference-{split}.npy'
        rows = data['outputs'][f'dpo_{split}.jsonl']['rows']
        values = np.load(folder / name, allow_pickle=False, mmap_mode='r')
        if (entries[name] != entry['sha256'] or entry['rows'] != rows or
                values.shape != (rows, 2) or values.dtype != np.float64 or
                not np.isfinite(values).all() or not (values <= 0).all()):
            raise ValueError('Archived fixed-reference score array differs')
    metadata = read_json(folder / 'snapshot.json')
    if (metadata.get('format') != 'warpstate-chat-v1' or metadata.get('inference_only') is not True or
            metadata.get('stage') != 'dpo' or metadata.get('config') != checkpoint['config'] or
            any(metadata.get(key) != checkpoint[key] for key in
                ('step', 'tokens_seen', 'response_tokens_seen', 'val_loss', 'origin')) or
            metadata.get('tokenizer_sha256') != entries['tokenizer.model'] or
            metadata.get('snapshot_sha256') != entries['model.pt'] or
            metadata.get('snapshot_bytes') != (folder / 'model.pt').stat().st_size or
            metadata.get('source', {}).get('sha256') != entries['checkpoint.pt']):
        raise ValueError('DPO model export metadata differs from the full archive')
    exported = torch.load(folder / 'model.pt', map_location='cpu', weights_only=False, mmap=True)
    if {key: value for key, value in exported.items() if key != 'model'} != {
            key: value for key, value in metadata.items() if key not in ('snapshot_sha256', 'snapshot_bytes')}:
        raise ValueError('DPO exported model metadata differs from snapshot.json')
    tensors_equal(checkpoint['model'], exported['model'])
    return metadata, entries


def archive_dpo(work, tokenizer, destination, plan, revision):
    """Caller owns all training exclusion locks; no source files are changed."""
    work, tokenizer, destination = map(Path, (work, tokenizer, destination))
    run, reference_dir = work / 'run', work / 'reference-cache'
    source = run / 'latest.pt'
    if source.is_symlink() or not source.is_file():
        raise ValueError('DPO checkpoint must be a regular completed file')
    before = source.stat()
    checkpoint = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    data_path = workspace_path(WORKSPACE / plan['handoff']['data_dir']) / 'manifest.json'
    validate_state(checkpoint, read_json(run / 'evaluation.json'), read_json(run / 'completion.json'),
        read_json(run / 'settings.json'), read_json(data_path), read_json(reference_dir / 'manifest.json'), plan)
    full_hash = sha256(source)
    if destination.exists() or destination.is_symlink():
        return verify_dpo_archive(destination, expected_checkpoint_hash=full_hash)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.dpo-final-', dir=destination.parent))
    copy_regular(source, staging / 'checkpoint.pt')
    if sha256(staging / 'checkpoint.pt') != full_hash:
        raise ValueError('DPO full-state archive copy differs from its source')
    metadata = dict(format='warpstate-chat-v1', inference_only=True, stage='dpo',
        config=checkpoint['config'], step=checkpoint['step'], tokens_seen=checkpoint['tokens_seen'],
        response_tokens_seen=checkpoint['response_tokens_seen'], val_loss=checkpoint['val_loss'],
        tokenizer_sha256=checkpoint['signature']['tokenizer_sha256'], origin=checkpoint['origin'],
        captured_at_utc=utc_now(), source=dict(path=str(source), device=before.st_dev,
            inode=before.st_ino, size=before.st_size, mtime_ns=before.st_mtime_ns, sha256=full_hash))
    with (staging / 'model.pt').open('xb') as stream:
        torch.save(dict(metadata, model=checkpoint['model']), stream)
        stream.flush()
        os.fsync(stream.fileno())
    metadata.update(snapshot_sha256=sha256(staging / 'model.pt'),
                    snapshot_bytes=(staging / 'model.pt').stat().st_size)
    atomic_json(staging / 'snapshot.json', metadata)
    atomic_json(staging / 'plan.json', plan)
    copies = {name: run / name for name in ('settings.json', 'evaluation.json', 'completion.json', 'metrics.jsonl')}
    copies.update({'tokenizer.model': tokenizer, 'dpo-execution.json': work / 'control/execution.json',
        'dpo-handoff.json': work / 'control/handoff.json', 'gpu-preflight.json': work / 'preflight/report.json',
        'data-manifest.json': data_path, 'reference-manifest.json': reference_dir / 'manifest.json',
        'reference-train.npy': reference_dir / 'train.npy', 'reference-val.npy': reference_dir / 'val.npy'})
    for name, path in copies.items():
        copy_regular(path, staging / name)
    atomic_json(staging / 'PROVENANCE.json', dict(format='warpstate-dpo-archive-v1',
        repository_revision=revision, archive_command_sha256=sha256(Path(__file__)),
        source_checkpoint_sha256=full_hash, model_sha256=metadata['snapshot_sha256'],
        dpo_step=checkpoint['step'], dpo_input_tokens=checkpoint['tokens_seen'],
        dpo_response_tokens=checkpoint['response_tokens_seen'], dpo_pairs=checkpoint['sample_cursor'],
        sft_step=plan['sft']['step'], sft_model_sha256=checkpoint['signature']['base_sha256'],
        raw_base_step=plan['sft']['base_step'], raw_base_tokens=plan['sft']['base_tokens_seen'],
        signature=checkpoint['signature']))
    note = f'''# KEEP: completed DPO after raw step {plan['sft']['base_step']}

User-requested historical archive created {utc_now()}. Preserve during routine
cleanup, further training and chat refreshes. Delete only on explicit request.

- DPO update {checkpoint['step']:,}; {checkpoint['sample_cursor']:,} preference pairs.
- DPO input tokens {checkpoint['tokens_seen']:,}; response targets {checkpoint['response_tokens_seen']:,}.
- Parent SFT update {plan['sft']['step']:,}; raw base update {plan['sft']['base_step']:,}.
- Final full-holdout DPO objective loss: {checkpoint['val_loss']:.12f}.
- `checkpoint.pt`: byte-identical full checkpoint, including FP32 model and Adam,
  per-rank RNG, sample cursor, model configuration and immutable signatures.
- `model.pt`: tensor-exact FP32 inference export with the original tokenizer.
- Plan, data/reference manifests and reference scores, original settings,
  evaluation, completion, metrics, preflight and execution provenance are included.
- Stage counters are distinct. DPO objective loss is not language-model perplexity.
- All files are independent physical copies, read-only, without hardlinks/reflinks.
- Existing raw/SFT archives and the original DPO run remain unchanged.
- Repository revision at archival time: `{revision}`.

Verify with `sha256sum -c SHA256SUMS`, or run `archive_dpo.py --verify-only`
with this directory as `--destination`. This is an on-device archive.
'''
    with (staging / 'ARCHIVE.md').open('x') as stream:
        stream.write(note)
        stream.flush()
        os.fsync(stream.fileno())
    after = source.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('DPO checkpoint changed during archival')
    seal(staging, FILES)
    result = verify_dpo_archive(staging, expected_checkpoint_hash=full_hash)
    publish_directory(staging, destination)
    return result


def run(config, destination=None, *, program='warpstate-dpo', verify_only=False):
    config = workspace_path(config)
    plan, paths = load_plan(config)
    work, base = paths['work_dir'], paths['base_work_dir']
    destination = workspace_path(destination or (base / 'archives' / f"step-{plan['sft']['base_step']:08d}-dpo-final"))
    if (destination.parent != base / 'archives' or
            destination == paths['archive_dir']):
        raise ValueError('DPO archive must use a distinct direct child of work/archives')
    if verify_only:
        metadata, checksums = verify_dpo_archive(destination)
    else:
        with contextlib.ExitStack() as locks:
            for path in (work / 'control/archive.lock', work / 'control/handoff.lock',
                         paths['sft_work_dir'] / 'control/handoff.lock',
                         base / 'control/training-launcher.lock', base / 'pipeline.lock'):
                locks.enter_context(exclusive_lock(path))
            if not clean_exit(process_status(paths['supervisor_socket'], program)):
                raise ValueError('DPO guardian must have exited successfully before archival')
            if read_json(work / 'control/handoff.json').get('phase') != 'complete':
                raise ValueError('DPO guardian has not completed its final integrity checks')
            model_hash = sha256(work / 'initialization/model.pt')
            if model_hash != sha256(paths['archive_dir'] / 'model.pt'):
                raise ValueError('Original SFT archive and DPO initialization differ')
            verify_dpo_completion(work / 'run', plan, model_hash)
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORKSPACE, text=True).strip()
            metadata, checksums = archive_dpo(work, paths['tokenizer'], destination, plan, revision)
    print(json.dumps(dict(archive=str(destination), step=metadata['step'], stage='dpo',
        checkpoint_sha256=checksums['checkpoint.pt'], model_sha256=checksums['model.pt'],
        tokens_seen=metadata['tokens_seen'], response_tokens_seen=metadata['response_tokens_seen']), indent=2))
    return metadata, checksums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PACKAGE / 'configs/dpo_10b.json')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--program', default='warpstate-dpo')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    run(args.config, args.destination, program=args.program, verify_only=args.verify_only)


if __name__ == '__main__':
    main()
