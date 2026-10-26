#!/usr/bin/env python3
"""Independent full-parameter assistant-only SFT from a frozen model export.

The output directory owns a fresh AdamW optimizer, token-weighted SFT metrics,
and resumable checkpoints. No pretraining checkpoint or source export is edited.
Run with torchrun for DDP; --device cpu exists only for small correctness fixtures.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist

from ops.posttrain_data import encode_messages, renderer_sha256, RENDER_FORMAT


FORMAT = 'warpstate-posttrain-v1'
WORKSPACE = Path('/workspace/WarpState')
STOP = False
DEFAULTS = dict(sequence_length=2048, micro_batch=4, grad_accum=4, lr=5e-6,
                min_lr=0.0, warmup_ratio=0.03, epochs=1, max_steps=None,
                weight_decay=0.0, seed=42, eval_every=100, save_every=100,
                eval_batches=32, compile_core=True, loss_mode='compiled-full',
                bf16_residuals=True, gradient_checkpointing=False, workers=4,
                deterministic=True, clip_grad_norm=1.0)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, report):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('format') != 'warpstate-posttrain-plan-v1' or plan.get('stage') != 'sft':
        raise ValueError('Expected warpstate-posttrain-plan-v1 stage sft')
    supplied = plan.get('training', {})
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown training settings: {sorted(unknown)}')
    settings = DEFAULTS | supplied
    for name in ('sequence_length', 'micro_batch', 'grad_accum', 'epochs',
                 'eval_every', 'save_every', 'eval_batches', 'workers'):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    ratio = settings['warmup_ratio']
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio < 1:
        raise ValueError('warmup_ratio must be in [0, 1)')
    if settings['max_steps'] is not None and (
            type(settings['max_steps']) is not int or settings['max_steps'] < 1):
        raise ValueError('max_steps must be null or a positive integer')
    for name in ('lr', 'min_lr', 'weight_decay', 'clip_grad_norm'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not 0 < settings['lr'] or settings['min_lr'] > settings['lr'] or not settings['clip_grad_norm']:
        raise ValueError('Require 0 <= min_lr <= lr and positive lr/clip_grad_norm')
    if type(settings['seed']) is not int or settings['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    for name in ('compile_core', 'bf16_residuals', 'gradient_checkpointing', 'deterministic'):
        if type(settings[name]) is not bool:
            raise ValueError(f'{name} must be a boolean')
    if settings['loss_mode'] not in ('full', 'compiled-full', 'checkpointed'):
        raise ValueError('Unsupported loss_mode')
    return plan, settings


def validate_paths(work_dir, data_dir, weights, tokenizer, config):
    paths = [Path(p).resolve() for p in (work_dir, data_dir, weights, tokenizer, config)]
    if any(not path.is_relative_to(WORKSPACE) for path in paths):
        raise ValueError('All inputs and outputs must stay inside /workspace/WarpState')
    work, data, source, tok, plan = paths
    if work == WORKSPACE or any(path == work or path.is_relative_to(work) for path in (data, source, tok, plan)):
        raise ValueError('Training outputs must be separate from every input')
    if work.is_relative_to(data) or work.is_relative_to(source.parent):
        raise ValueError('Training outputs must not be inside data or source-weight directories')
    for path in (source, tok, plan):
        if not path.is_file():
            raise ValueError(f'Missing required input: {path}')
    if not data.is_dir():
        raise ValueError(f'Missing prepared data directory: {data}')
    return paths


def load_data(data_dir, tokenizer, tokenizer_hash, sequence_length):
    """Refuse partial/mismatched data; re-encode exactly before model allocation."""
    directory = Path(data_dir)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 'warpstate-posttrain-data-v1' or manifest.get('complete') is not True:
        raise ValueError('Posttraining data is not complete')
    if manifest.get('tokenizer_sha256') != tokenizer_hash:
        raise ValueError('Prepared data tokenizer hash differs')
    if manifest.get('renderer_sha256') != renderer_sha256():
        raise ValueError('Prepared data renderer hash differs')
    if manifest.get('max_length') != sequence_length:
        raise ValueError('Prepared data max_length differs from sequence_length')
    result = []
    for filename in ('sft_train.jsonl', 'sft_val.jsonl'):
        info = manifest.get('outputs', {}).get(filename)
        path = directory / filename
        if not info or not path.is_file() or digest(path) != info.get('sha256'):
            raise ValueError(f'Missing or changed prepared file: {filename}')
        records = []
        token_count = target_count = 0
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                row = json.loads(line)
                if row.get('tools'):
                    raise ValueError(f'{filename}:{number}: tools must be injected into messages')
                encoded = encode_messages(row['messages'], tokenizer, max_length=sequence_length)
                tokens = len(encoded['input_ids'])
                targets = sum(label != -100 for label in encoded['labels'][1:])
                if row.get('tokens') != tokens or row.get('supervised_tokens') != targets:
                    raise ValueError(f'{filename}:{number}: token metadata differs from exact encoding')
                records.append(encoded)
                token_count += tokens
                target_count += targets
        if not records or len(records) != info.get('rows'):
            raise ValueError(f'{filename}: empty data or row count mismatch')
        if token_count != info.get('tokens') or target_count != info.get('supervised_tokens'):
            raise ValueError(f'{filename}: manifest token totals differ')
        result.append(records)
    return *result, manifest, digest(manifest_path)


def collate(records, sequence_length, pad_id):
    """Right-pad independent causal examples; mask padding and all prompt labels."""
    x = torch.full((len(records), sequence_length), pad_id, dtype=torch.long)
    y = torch.full_like(x, -100)
    for index, record in enumerate(records):
        if record is None:
            continue
        ids, labels = record['input_ids'], record['labels']
        if len(ids) != len(labels) or not 2 <= len(ids) <= sequence_length:
            raise ValueError('Invalid encoded record length')
        length = len(ids) - 1
        x[index, :length] = torch.tensor(ids[:-1])
        y[index, :length] = torch.tensor(labels[1:])
    return x, y


@lru_cache(maxsize=2)
def epoch_order(size, seed, epoch):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return tuple(order)


def update_indices(size, seed, cursor, epochs, global_batch):
    """One deterministic batch, never crossing an epoch or repeating tail data."""
    epoch, offset = divmod(cursor, size)
    if epoch >= epochs:
        return [], cursor
    count = min(global_batch, size - offset)
    order = epoch_order(size, seed, epoch)
    return list(order[offset:offset + count]), cursor + count


def learning_rate(step, total_steps, settings):
    warmup = min(math.ceil(settings['warmup_ratio'] * total_steps), max(0, total_steps - 1))
    if step < warmup:
        return settings['lr'] * (step + 1) / warmup
    progress = min(1., (step - warmup) / max(1, total_steps - warmup - 1))
    return settings['min_lr'] + .5 * (settings['lr'] - settings['min_lr']) * (1 + math.cos(math.pi * progress))


def token_weighted_loss(mean_loss, local_targets, global_targets, world_size):
    """DDP averages ranks, so multiply each local token sum by world/global sum."""
    if global_targets < 1:
        raise ValueError('Update has no supervised tokens')
    return mean_loss * (local_targets * world_size / global_targets)


def rng_state(device):
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device)
            if device.type == 'cuda' else None, 'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def validate_resume(checkpoint, signature, world_size):
    if checkpoint.get('format') != FORMAT or checkpoint.get('signature') != signature:
        raise ValueError('Posttraining resume signature differs; use a new output directory')
    if len(checkpoint.get('rng_by_rank', [])) != world_size:
        raise ValueError('Checkpoint lacks exact per-rank RNG state')


@contextlib.contextmanager
def shared_pipeline_lock(path):
    """Every rank excludes pretraining, even if its launcher dies unexpectedly."""
    path = Path(path).resolve()
    if not path.is_relative_to(WORKSPACE):
        raise ValueError('Pipeline lock must stay inside the workspace')
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Pretraining or its handoff currently owns pipeline.lock') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def base_pipeline_lock(plan):
    base = plan.get('handoff', {}).get('base_work_dir')
    if base is None:
        yield
        return
    path = (WORKSPACE / base).resolve()
    if not path.is_relative_to(WORKSPACE) or path == WORKSPACE or not path.is_dir():
        raise ValueError('handoff.base_work_dir must be an existing workspace directory')
    with shared_pipeline_lock(path / 'pipeline.lock'):
        yield


def training_update(model, optimizer, microbatches, global_targets, world_size, device, clip_norm):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    target_count = 0
    for index, (x, y) in enumerate(microbatches):
        targets = int(y.ne(-100).sum())
        target_count += targets
        sync = model.no_sync() if world_size > 1 and index < len(microbatches) - 1 else contextlib.nullcontext()
        with sync:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=True, cache_enabled=False):
                _, loss = model(x.to(device), y.to(device), return_logits=False)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite SFT training loss')
            token_weighted_loss(loss, targets, global_targets, world_size).backward()
        loss_sum += loss.detach().double() * targets
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    totals = torch.stack((loss_sum, torch.tensor(target_count, device=device, dtype=torch.float64)))
    if world_size > 1:
        dist.all_reduce(totals)
    return float(totals[0] / totals[1]), float(grad_norm)


def evaluate(model, records, settings, device, pad_id, *, full=False):
    was_training = model.training
    state = rng_state(device)
    model.eval()
    total_loss, total_targets = 0., 0
    try:
        with torch.no_grad():
            limit = len(records) if full else settings['eval_batches'] * settings['micro_batch']
            for start in range(0, min(len(records), limit), settings['micro_batch']):
                batch = records[start:min(start + settings['micro_batch'], limit)]
                x, y = collate(batch, settings['sequence_length'], pad_id)
                targets = int(y.ne(-100).sum())
                with torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False):
                    _, loss = model(x.to(device), y.to(device), return_logits=False)
                total_loss += float(loss) * targets
                total_targets += targets
    finally:
        model.train(was_training)
        restore_rng(state, device)
    value = total_loss / total_targets
    if not math.isfinite(value):
        raise RuntimeError('Non-finite SFT validation loss')
    return value, total_targets


def source_identity():
    root = Path(__file__).parent
    files = [Path(__file__), root / 'ops/posttrain_data.py', root / 'ops/loss_experiment.py',
             root / 'ops/residual_experiment.py', root / 'ops/checkpoint_writer.py', root / 'run_40b.py']
    files += sorted((root / 'warpstate').glob('*.py'))
    return {str(path.relative_to(root)): digest(path) for path in files}


def train(args):
    import sentencepiece as spm
    from warpstate import WarpStateConfig, WarpStateLM
    from run_40b import compile_training_cores
    from ops.checkpoint_writer import DistributedCheckpointWriter
    from ops.loss_experiment import apply_loss_execution
    from ops.residual_experiment import apply_bf16_residuals

    work, data_dir, weights, tokenizer_path, config = validate_paths(
        args.work_dir, args.data_dir, args.weights, args.tokenizer, args.config)
    plan, settings = load_plan(config)
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16-capable CUDA GPU required')
    torch.set_num_threads(settings['workers'])
    torch.use_deterministic_algorithms(settings['deterministic'])
    random.seed(settings['seed'] + rank)
    np.random.seed(settings['seed'] + rank)
    torch.manual_seed(settings['seed'] + rank)
    tokenizer_hash = digest(tokenizer_path)
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    train_data, val_data, manifest, manifest_hash = load_data(data_dir, sp, tokenizer_hash, settings['sequence_length'])
    base_hash = digest(weights)
    signature = dict(plan=plan, settings=settings, data_sha256=manifest_hash, tokenizer_sha256=tokenizer_hash,
                     renderer=RENDER_FORMAT, base_sha256=base_hash, world_size=world,
                     device_type=device.type, torch_version=torch.__version__, sources=source_identity())
    latest = work / 'latest.pt'
    resuming = latest.exists()
    checkpoint = torch.load(latest if resuming else weights, map_location='cpu', weights_only=False, mmap=True)
    if resuming:
        validate_resume(checkpoint, signature, world)
        origin = checkpoint['origin']
    else:
        if checkpoint.get('format') != 'warpstate-chat-v1' or checkpoint.get('inference_only') is not True:
            raise ValueError('Initialize SFT from an independent warpstate-chat-v1 model-only export')
        if checkpoint.get('tokenizer_sha256') != tokenizer_hash:
            raise ValueError('Base model export tokenizer differs')
        for key in ('step', 'tokens_seen', 'tokenizer_sha256'):
            if key in plan.get('base', {}) and plan['base'][key] != checkpoint.get(key):
                raise ValueError(f'Base model {key} differs from plan')
        origin = {key: value for key, value in checkpoint.items() if key != 'model'}
        origin.update(weights_sha256=base_hash, weights_path=str(weights))
    cfg = WarpStateConfig(**checkpoint['config'])
    cfg.gradient_checkpointing = settings['gradient_checkpointing']
    cfg.validate()
    if cfg.vocab_size != sp.vocab_size() or settings['sequence_length'] % cfg.chunk_size:
        raise ValueError('Model/tokenizer vocabulary or sequence/chunk mismatch')
    with torch.device('meta'):
        raw = WarpStateLM(cfg)
    raw.load_state_dict(checkpoint['model'], strict=True, assign=True)
    raw.to(device=device, dtype=torch.float32).requires_grad_(True)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=settings['lr'], betas=(.9, .95),
                                 eps=1e-8, weight_decay=settings['weight_decay'], fused=device.type == 'cuda')
    step = cursor = tokens_seen = assistant_tokens_seen = 0
    if resuming:
        optimizer.load_state_dict(checkpoint['optimizer'])
        step, cursor = checkpoint['step'], checkpoint['sample_cursor']
        tokens_seen = checkpoint['tokens_seen']
        assistant_tokens_seen = checkpoint['assistant_tokens_seen']
    resume_rng = checkpoint['rng_by_rank'][rank] if resuming else None
    del checkpoint
    if settings['compile_core']:
        compile_training_cores(raw)
    apply_loss_execution(raw, settings['loss_mode'])
    if settings['bf16_residuals']:
        apply_bf16_residuals(raw)
    if world > 1:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
        model = torch.nn.parallel.DistributedDataParallel(raw,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            broadcast_buffers=False, gradient_as_bucket_view=True, bucket_cap_mb=128)
    else:
        model = raw
    if resume_rng is not None:
        restore_rng(resume_rng, device)
    global_batch = settings['micro_batch'] * settings['grad_accum'] * world
    total_steps = math.ceil(len(train_data) / global_batch) * settings['epochs']
    if settings['max_steps'] is not None:
        total_steps = min(total_steps, settings['max_steps'])
    end_step = min(total_steps, step + args.max_steps) if args.max_steps is not None else total_steps
    work.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (work / 'settings.json').write_text(json.dumps(dict(signature=signature, origin=origin,
            total_steps=total_steps, train_rows=len(train_data), val_rows=len(val_data)), indent=2) + '\n')
        if step < total_steps:
            (work / 'completion.json').unlink(missing_ok=True)
    pad_id = sp.pad_id() if sp.pad_id() >= 0 else sp.eos_id()
    last_validation = None
    last_validation_step = None

    def validation(*, full=False):
        values = [evaluate(raw, val_data, settings, device, pad_id, full=full) if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(values, src=0, device=device)
        return values[0]

    def metric(values):
        if rank == 0:
            values.update(timestamp=time.time(), stage='sft', step=step, tokens_seen=tokens_seen,
                          assistant_tokens_seen=assistant_tokens_seen, sample_cursor=cursor)
            with (work / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(values, allow_nan=False) + '\n')
            if values.get('event') == 'validation':
                count = len(val_data) if values.get('full_holdout') else min(len(val_data), settings['eval_batches'] * settings['micro_batch'])
                report = dict(values, evaluated_samples=count,
                              available_validation_samples=len(val_data), data_sha256=manifest_hash,
                              base_sha256=base_hash, base_step=origin['step'], objective='assistant-token-mean',
                              tokenizer_sha256=tokenizer_hash, renderer_format=RENDER_FORMAT)
                atomic_json(work / 'evaluation.json', report)
            print(json.dumps(values, allow_nan=False), flush=True)

    def save(writer):
        state = rng_state(device)
        rank_states = [None] * world
        if world > 1:
            dist.all_gather_object(rank_states, state)
        else:
            rank_states = [state]
        writer.submit(state_factory=lambda: dict(format=FORMAT, signature=signature, origin=origin,
            config=vars(cfg), model=raw.state_dict(), optimizer=optimizer.state_dict(), step=step,
            sample_cursor=cursor, tokens_seen=tokens_seen, assistant_tokens_seen=assistant_tokens_seen,
            val_loss=last_validation, val_step=last_validation_step, rng_by_rank=rank_states,
            completed=step >= total_steps, total_steps=total_steps))

    def stop_requested():
        flag = STOP or bool(args.stop_file and args.stop_file.exists())
        if world > 1:
            value = torch.tensor(int(flag), device=device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            return bool(value.item())
        return flag

    with DistributedCheckpointWriter(latest, rank=rank, device=device, distributed=world > 1) as writer:
        last_validation, eval_tokens = validation(full=True)
        last_validation_step = step
        metric(dict(event='validation', val_loss=last_validation, val_perplexity=math.exp(last_validation),
                    validation_assistant_tokens=eval_tokens, full_holdout=True))
        last_saved_step = step if resuming else -1
        model.train()
        while step < end_step and not stop_requested():
            writer.check()
            indices, next_cursor = update_indices(len(train_data), settings['seed'], cursor,
                                                  settings['epochs'], global_batch)
            if not indices:
                break
            global_targets = sum(sum(label != -100 for label in train_data[i]['labels'][1:]) for i in indices)
            microbatches = []
            for micro in range(settings['grad_accum']):
                offset = (micro * world + rank) * settings['micro_batch']
                selected = [train_data[i] for i in indices[offset:offset + settings['micro_batch']]]
                selected += [None] * (settings['micro_batch'] - len(selected))
                microbatches.append(collate(selected, settings['sequence_length'], pad_id))
            lr = learning_rate(step, total_steps, settings)
            for group in optimizer.param_groups:
                group['lr'] = lr
            started = time.perf_counter()
            loss, grad_norm = training_update(model, optimizer, microbatches, global_targets, world,
                                              device, settings['clip_grad_norm'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            step += 1
            cursor = next_cursor
            actual_tokens = sum(len(train_data[i]['input_ids']) - 1 for i in indices)
            tokens_seen += actual_tokens
            assistant_tokens_seen += global_targets
            metric(dict(event='train', loss=loss, lr=lr, grad_norm=grad_norm, update_seconds=seconds,
                        tokens_per_second=actual_tokens / seconds,
                        assistant_tokens_per_second=global_targets / seconds, update_samples=len(indices)))
            if step % settings['eval_every'] == 0 or step == end_step:
                full = step >= total_steps
                last_validation, eval_tokens = validation(full=full)
                last_validation_step = step
                metric(dict(event='validation', val_loss=last_validation, val_perplexity=math.exp(last_validation),
                            validation_assistant_tokens=eval_tokens, full_holdout=full))
            if step % settings['save_every'] == 0 or step == end_step:
                save(writer)
                last_saved_step = step
        if last_saved_step != step:
            save(writer)
        writer.flush()
        if rank == 0 and step >= total_steps:
            atomic_json(work / 'completion.json', dict(format='warpstate-posttrain-completion-v1',
                stage='sft', completed=True, paused_for_evaluation=True, step=step,
                total_steps=total_steps, base_step=origin['step'], tokens_seen=tokens_seen,
                assistant_tokens_seen=assistant_tokens_seen, val_loss=last_validation,
                evaluation_file=str(work / 'evaluation.json'), checkpoint=str(latest),
                timestamp=time.time()))
        metric(dict(event='complete' if step >= total_steps else 'paused', total_steps=total_steps,
                    checkpoint=str(latest)))
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'work-dir', 'data-dir', 'weights', 'tokenizer'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-steps', type=int, help='Pause after this many additional updates; schedule is unchanged')
    parser.add_argument('--stop-file', type=Path, help='Defaults to work-dir/control/training-stop-request')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda', help='CPU is only for tiny correctness fixtures')
    args = parser.parse_args()
    if args.stop_file is None:
        args.stop_file = args.work_dir / 'control/training-stop-request'
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.stop_file is not None and not args.stop_file.resolve().is_relative_to(WORKSPACE):
        parser.error('--stop-file must stay inside the workspace')
    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    from run_40b import process_lock
    rank = int(os.environ.get('RANK', '0'))
    # Separate lock files avoid duplicate rank launches without serializing DDP.
    lock_dir = args.work_dir.resolve()
    validate_paths(args.work_dir, args.data_dir, args.weights, args.tokenizer, args.config)
    lock_dir.mkdir(parents=True, exist_ok=True)
    plan, _ = load_plan(args.config)
    with process_lock(lock_dir / f'rank-{rank}.lock'), base_pipeline_lock(plan):
        train(args)


if __name__ == '__main__':
    main()
