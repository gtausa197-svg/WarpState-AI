#!/usr/bin/env python3
"""BF16 DPO with an immutable SFT reference, exact resume and response-only loss.

The reference is scored once, before allocating the policy optimizer. Its cache
is bound to weights, data, masking, execution and source hashes. Policy training
uses fresh FP32 Adam state and saves only into its own output directory.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import types

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from posttrain import (WORKSPACE, atomic_json, base_pipeline_lock, digest,
                       restore_rng, rng_state, update_indices, validate_paths)
from ops.dpo_data import MASK_FORMAT, collate_pairs, load_data
from ops.posttrain_data import RENDER_FORMAT, renderer_sha256

FORMAT = 'warpstate-dpo-v1'
CACHE_FORMAT = 'warpstate-dpo-reference-v1'
STOP = False
DEFAULTS = dict(sequence_length=2048, micro_batch=2, grad_accum=8, lr=5e-7,
                min_lr=0., warmup_ratio=.1, epochs=1, max_steps=None,
                weight_decay=0., seed=42, eval_every=100, save_every=100,
                eval_batches=32, compile_core=True, bf16_residuals=True,
                gradient_checkpointing=False, workers=4, deterministic=True,
                clip_grad_norm=1., beta=5., objective='length-normalized-dpo',
                head_chunk_size=256)


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('format') != 'warpstate-dpo-plan-v1' or plan.get('stage') != 'dpo':
        raise ValueError('Expected warpstate-dpo-plan-v1 stage dpo')
    supplied = plan.get('training', {})
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown DPO training settings: {sorted(unknown)}')
    settings = DEFAULTS | supplied
    for key in ('sequence_length', 'micro_batch', 'grad_accum', 'epochs', 'eval_every',
                'save_every', 'eval_batches', 'workers', 'head_chunk_size'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if settings['max_steps'] is not None and (type(settings['max_steps']) is not int or settings['max_steps'] < 1):
        raise ValueError('max_steps must be null or a positive integer')
    for key in ('lr', 'min_lr', 'warmup_ratio', 'weight_decay', 'clip_grad_norm', 'beta'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if not 0 <= settings['warmup_ratio'] < 1 or not 0 <= settings['min_lr'] <= settings['lr']:
        raise ValueError('Invalid LR schedule')
    if not settings['lr'] or not settings['clip_grad_norm'] or not settings['beta']:
        raise ValueError('lr, clip_grad_norm and beta must be positive')
    for key in ('compile_core', 'bf16_residuals', 'gradient_checkpointing', 'deterministic'):
        if type(settings[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    if type(settings['seed']) is not int or settings['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if settings['objective'] not in ('length-normalized-dpo', 'summed-dpo'):
        raise ValueError('Unsupported DPO objective')
    return plan, settings


def source_identity():
    from posttrain import source_identity as shared_sources
    root = Path(__file__).parent
    return shared_sources() | {str(path.relative_to(root)): digest(path)
                               for path in (Path(__file__), root / 'ops/dpo_data.py')}


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def learning_rate(step, total_steps, settings):
    """Fixed linear warmup/decay schedule, independent of a pause boundary."""
    warmup = min(math.ceil(settings['warmup_ratio'] * total_steps), max(0, total_steps - 1))
    if step < warmup:
        return settings['lr'] * (step + 1) / warmup
    progress = min(1., (step - warmup) / max(1, total_steps - warmup - 1))
    return settings['min_lr'] + (settings['lr'] - settings['min_lr']) * (1. - progress)


def reference_identity(weights_hash, manifest_hash, tokenizer_hash, settings, device, world):
    return dict(format=CACHE_FORMAT, weights_sha256=weights_hash, data_sha256=manifest_hash,
                tokenizer_sha256=tokenizer_hash, renderer_sha256=renderer_sha256(),
                mask_format=MASK_FORMAT, sources=source_identity(), settings={key: settings[key]
                    for key in ('sequence_length', 'micro_batch', 'head_chunk_size', 'compile_core',
                                'bf16_residuals', 'deterministic', 'workers')},
                dropout=0., logps='response-token-sum-float32', dtype='bfloat16',
                device_type=device.type, world_size=world, torch_version=torch.__version__)


def validate_cache(directory, identity, sizes):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest.get('format') != CACHE_FORMAT or manifest.get('complete') is not True or
            manifest.get('identity') != identity or manifest.get('identity_sha256') != identity_digest(identity)):
        raise ValueError('Reference cache identity/completion differs')
    result = []
    for name, size in zip(('train.npy', 'val.npy'), sizes):
        path = directory / name
        info = manifest.get('files', {}).get(name, {})
        if not path.is_file() or path.is_symlink() or digest(path) != info.get('sha256') or info.get('rows') != size:
            raise ValueError(f'Reference cache file changed: {name}')
        values = np.load(path, allow_pickle=False, mmap_mode='r')
        if values.shape != (size, 2) or values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError(f'Invalid reference log probabilities: {name}')
        result.append(values)
    return *result, digest(directory / 'manifest.json')


def apply_sequence_logps(model, chunk_size):
    """Return per-sequence log-probability sums without retaining vocabulary logits.

Vocabulary matmuls use BF16 autocast; cross-entropy reductions use FP32. Each
head chunk is recomputed in backward, so DPO keeps a vector of branch scores.
The backbone modules and canonical parameter/state names are unchanged.
"""
    def forward(self, input_ids, targets=None, *, return_logits=False):
        if targets is None or return_logits:
            raise ValueError('DPO forward requires response targets and no returned logits')
        hidden = self.embed(input_ids)
        for depth in range(self.cfg.logical_depth):
            hidden = hidden * (1. + self.depth_scale[depth]) + self.depth_bias[depth]
            core = self.cores[depth % self.cfg.n_cores]
            if self.training and self.cfg.gradient_checkpointing:
                hidden = activation_checkpoint(core, hidden, use_reentrant=False)
            else:
                hidden = core(hidden)
        hidden = self.final_norm(hidden)

        def score(h, weight, labels):
            logits = F.linear(h, weight).float()
            losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                     reduction='none', ignore_index=-100)
            return -losses.reshape(labels.shape).sum(dim=1)

        logps = hidden.new_zeros((hidden.shape[0],), dtype=torch.float32)
        for start in range(0, hidden.shape[1], chunk_size):
            args = (hidden[:, start:start + chunk_size], self.embed.weight,
                    targets[:, start:start + chunk_size])
            value = activation_checkpoint(score, *args, use_reentrant=False) if torch.is_grad_enabled() else score(*args)
            logps = logps + value
        return None, logps
    model.forward = types.MethodType(forward, model)


def preference_metrics(policy, reference, counts, beta, objective):
    """Pair vectors, retaining gradients only through policy log probabilities."""
    reference = reference.detach().to(policy.dtype)
    if objective == 'length-normalized-dpo':
        policy = policy / counts.clamp_min(1)
        reference = reference / counts.clamp_min(1)
    elif objective != 'summed-dpo':
        raise ValueError('Unsupported DPO objective')
    rewards = beta * (policy - reference)
    margin = rewards[:, 0] - rewards[:, 1]
    return -F.logsigmoid(margin), margin, (margin > 0).to(policy.dtype)


def stop_requested(args, device, world):
    value = STOP or bool(args.stop_file and Path(args.stop_file).exists())
    if world > 1:
        flag = torch.tensor(int(value), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        value = bool(flag.item())
    return value


def load_model(checkpoint, settings, device, tokenizer):
    from warpstate import WarpStateConfig, WarpStateLM
    from run_40b import compile_training_cores
    from ops.residual_experiment import apply_bf16_residuals
    cfg = WarpStateConfig(**checkpoint['config'])
    cfg.gradient_checkpointing = settings['gradient_checkpointing']
    # DPO compares a deterministic policy/reference scoring rule, with dropout disabled.
    cfg.dropout = 0.
    cfg.validate()
    if cfg.vocab_size != tokenizer.vocab_size() or settings['sequence_length'] % cfg.chunk_size:
        raise ValueError('Model/tokenizer vocabulary or sequence/chunk mismatch')
    with torch.device('meta'):
        raw = WarpStateLM(cfg)
    raw.load_state_dict(checkpoint['model'], strict=True, assign=True)
    # Clone on CPU as well: mmap initialization is immutable and must not alias training.
    raw.to(device=device, dtype=torch.float32)
    if device.type == 'cpu':
        for parameter in raw.parameters():
            parameter.data = parameter.data.clone()
    if settings['compile_core']:
        compile_training_cores(raw)
    apply_sequence_logps(raw, settings['head_chunk_size'])
    if settings['bf16_residuals']:
        apply_bf16_residuals(raw)
    return raw, cfg


def atomic_array(path, array):
    temporary = Path(str(path) + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare_reference(args, identity, records, settings, device, rank, world, tokenizer):
    """Create/reuse distributed resumable score shards, then seal a checked cache."""
    directory = Path(args.reference_cache_dir)
    if (directory / 'manifest.json').exists():
        return validate_cache(directory, identity, [len(rows) for rows in records])
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / 'identity.json'
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError('Partial reference cache identity differs')
    if rank == 0:
        atomic_json(marker, identity)
    if world > 1:
        dist.barrier()
    checkpoint = torch.load(args.weights, map_location='cpu', weights_only=False, mmap=True)
    raw, _ = load_model(checkpoint, settings, device, tokenizer)
    raw.requires_grad_(False).eval()
    del checkpoint
    pad_id = tokenizer.pad_id() if tokenizer.pad_id() >= 0 else tokenizer.eos_id()
    started = time.monotonic()
    try:
        for split, rows in zip(('train', 'val'), records):
            indices = list(range(rank * settings['micro_batch'], len(rows), world * settings['micro_batch']))
            shard = directory / f'{split}-rank-{rank}.npz'
            done, scored_indices, scores = 0, [], []
            if shard.exists():
                with np.load(shard, allow_pickle=False) as old:
                    if str(old['identity']) != identity_digest(identity):
                        raise ValueError('Partial reference shard identity differs')
                    done = int(old['batches'])
                    scored_indices = old['indices'].tolist()
                    scores = old['scores'].tolist()
                expected_indices = [i for start in indices[:done]
                                    for i in range(start, min(start + settings['micro_batch'], len(rows)))]
                if scored_indices != expected_indices or len(scores) != len(scored_indices):
                    raise ValueError('Partial reference shard cursor differs')

            def save_shard():
                temporary = Path(str(shard) + '.tmp')
                with temporary.open('wb') as stream:
                    np.savez(stream, identity=identity_digest(identity), batches=done,
                             indices=np.asarray(scored_indices, dtype=np.int64),
                             scores=np.asarray(scores, dtype=np.float64).reshape(-1, 2))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, shard)

            # Every rank executes equal stop/barrier counts, even on uneven final shards.
            max_batches = math.ceil(len(rows) / (world * settings['micro_batch']))
            for batch in range(max_batches):
                if stop_requested(args, device, world):
                    save_shard()
                    return None
                if batch < done or batch >= len(indices):
                    continue
                start = indices[batch]
                selected = list(range(start, min(start + settings['micro_batch'], len(rows))))
                padded = [rows[i] for i in selected] + [None] * (settings['micro_batch'] - len(selected))
                x, y, _ = collate_pairs(padded, settings['sequence_length'], pad_id)
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False):
                    _, logps = raw(x.to(device), y.to(device), return_logits=False)
                values = logps.reshape(-1, 2).double().cpu().numpy()[:len(selected)]
                if not np.isfinite(values).all():
                    raise RuntimeError('Non-finite frozen reference scores')
                scored_indices.extend(selected)
                scores.extend(values.tolist())
                done = batch + 1
                if done % 128 == 0 or done == len(indices):
                    save_shard()
                    print(json.dumps(dict(event='reference_progress', rank=rank, split=split,
                        batches=done, total_batches=len(indices), scored_pairs=len(scores),
                        elapsed_seconds=time.monotonic() - started)), flush=True)
            save_shard()
        if world > 1:
            dist.barrier()
        if rank == 0:
            files = {}
            for split, rows in zip(('train', 'val'), records):
                combined = np.full((len(rows), 2), np.nan, dtype=np.float64)
                seen = np.zeros(len(rows), dtype=np.bool_)
                for owner in range(world):
                    with np.load(directory / f'{split}-rank-{owner}.npz', allow_pickle=False) as shard:
                        if str(shard['identity']) != identity_digest(identity):
                            raise ValueError('Reference shard identity differs')
                        index = shard['indices']
                        if seen[index].any():
                            raise ValueError('Duplicate reference indices')
                        combined[index] = shard['scores']
                        seen[index] = True
                if not seen.all() or not np.isfinite(combined).all():
                    raise ValueError('Reference cache is incomplete')
                path = directory / f'{split}.npy'
                atomic_array(path, combined)
                files[path.name] = dict(sha256=digest(path), rows=len(rows))
            atomic_json(directory / 'manifest.json', dict(format=CACHE_FORMAT, complete=True,
                identity=identity, identity_sha256=identity_digest(identity), files=files))
        if world > 1:
            dist.barrier()
        return validate_cache(directory, identity, [len(rows) for rows in records])
    finally:
        del raw
        # Patched bound forwards and functional BF16 wrappers form cycles;
        # collect them before allocating the policy and FP32 Adam state.
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def training_update(model, optimizer, microbatches, global_pairs, world, device, settings):
    optimizer.zero_grad(set_to_none=True)
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    for index, (x, y, active, reference) in enumerate(microbatches):
        sync = model.no_sync() if world > 1 and index < len(microbatches) - 1 else contextlib.nullcontext()
        x, y, active, reference = (value.to(device) for value in (x, y, active, reference))
        counts = y.ne(-100).sum(dim=1).reshape(-1, 2)
        with sync:
            with torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False):
                _, logps = model(x, y, return_logits=False)
            logps = logps.reshape(-1, 2)
            losses, margins, accuracy = preference_metrics(logps, reference, counts,
                                                           settings['beta'], settings['objective'])
            if not torch.isfinite(losses).all():
                raise RuntimeError('Non-finite DPO training loss')
            # DDP averages rank gradients; this is the true global pair mean,
            # including a partial final update and entirely empty rank slots.
            (losses * active).sum().mul(world / global_pairs).backward()
        totals += torch.stack(((losses.detach() * active).double().sum(),
                               (margins.detach() * active).double().sum(),
                               (accuracy.detach() * active).double().sum(),
                               ((-logps[:, 0].detach() / counts[:, 0].clamp_min(1)) * active).double().sum(),
                               active.double().sum()))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip_grad_norm'], error_if_nonfinite=True)
    optimizer.step()
    if world > 1:
        dist.all_reduce(totals)
    if int(totals[-1]) != global_pairs:
        raise RuntimeError('Distributed DPO pair count differs')
    values = (totals[:4] / totals[-1]).tolist()
    return dict(zip(('loss', 'reward_margin', 'reward_accuracy', 'chosen_nll'), values)), float(norm)


def evaluate(model, records, reference, settings, device, pad_id, *, full=False):
    state = rng_state(device)
    was_training = model.training
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    count = 0
    limit = len(records) if full else min(len(records), settings['eval_batches'] * settings['micro_batch'])
    try:
        with torch.no_grad():
            for start in range(0, limit, settings['micro_batch']):
                selected = records[start:min(start + settings['micro_batch'], limit)]
                n = len(selected)
                selected += [None] * (settings['micro_batch'] - n)
                x, y, active = collate_pairs(selected, settings['sequence_length'], pad_id)
                x, y, active = x.to(device), y.to(device), active.to(device)
                ref = torch.zeros((settings['micro_batch'], 2), device=device)
                ref[:n] = torch.tensor(np.array(reference[start:start + n]), device=device)
                with torch.autocast(device.type, dtype=torch.bfloat16, cache_enabled=False):
                    _, logps = model(x, y, return_logits=False)
                logps = logps.reshape(-1, 2)
                counts = y.ne(-100).sum(dim=1).reshape(-1, 2)
                losses, margins, accuracy = preference_metrics(logps, ref, counts,
                                                               settings['beta'], settings['objective'])
                totals += torch.stack(((losses * active).double().sum(), (margins * active).double().sum(),
                    (accuracy * active).double().sum(),
                    ((-logps[:, 0] / counts[:, 0].clamp_min(1)) * active).double().sum()))
                count += n
    finally:
        model.train(was_training)
        restore_rng(state, device)
    values = (totals / count).tolist()
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError('Non-finite DPO validation')
    return dict(zip(('val_loss', 'reward_margin', 'reward_accuracy', 'chosen_nll'), values)) | dict(evaluated_samples=count)


def validate_resume(checkpoint, signature, world):
    if checkpoint.get('format') != FORMAT or checkpoint.get('signature') != signature:
        raise ValueError('DPO resume signature differs; use a new output directory')
    if len(checkpoint.get('rng_by_rank', [])) != world:
        raise ValueError('DPO checkpoint lacks exact per-rank RNG state')


def validate_reference_path(args):
    cache = Path(args.reference_cache_dir).resolve()
    paths = [Path(path).resolve() for path in (args.work_dir, args.data_dir, args.weights.parent)]
    if not cache.is_relative_to(WORKSPACE) or cache == WORKSPACE:
        raise ValueError('Reference cache must stay inside the workspace')
    if any(cache == path or cache.is_relative_to(path) or path.is_relative_to(cache) for path in paths):
        raise ValueError('Reference cache must be separate from training outputs and inputs')
    return cache


def train(args):
    import sentencepiece as spm
    from ops.checkpoint_writer import DistributedCheckpointWriter
    work, data_dir, weights, tokenizer_path, config = validate_paths(
        args.work_dir, args.data_dir, args.weights, args.tokenizer, args.config)
    cache = validate_reference_path(args)
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
    tokenizer_hash, base_hash = digest(tokenizer_path), digest(weights)
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    train_data, val_data, manifest, manifest_hash = load_data(data_dir, sp, tokenizer_hash, settings['sequence_length'])
    source = torch.load(weights, map_location='cpu', weights_only=False, mmap=True)
    if (source.get('format') != 'warpstate-chat-v1' or source.get('inference_only') is not True or
            source.get('stage') != 'sft' or source.get('tokenizer_sha256') != tokenizer_hash):
        raise ValueError('Initialize DPO from an independent, tokenizer-matched SFT model export')
    expected = plan.get('sft', plan.get('base', {}))
    for key in ('step', 'tokens_seen', 'tokenizer_sha256'):
        if key in expected and expected[key] != source.get(key):
            raise ValueError(f'SFT source {key} differs from plan')
    ancestor = source.get('origin', {})
    for key in ('step', 'tokens_seen'):
        expected_key = 'base_' + key
        value = ancestor.get(key, ancestor.get(expected_key))
        if expected_key in expected and expected[expected_key] != value:
            raise ValueError(f'SFT source {expected_key} differs from plan')
    origin = {key: value for key, value in source.items() if key != 'model'}
    origin.update(weights_sha256=base_hash, weights_path=str(weights))
    del source
    identity = reference_identity(base_hash, manifest_hash, tokenizer_hash, settings, device, world)
    if world > 1:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
    try:
        reference = prepare_reference(args, identity, (train_data, val_data), settings, device, rank, world, sp)
        if reference is None:
            print(json.dumps(dict(event='reference_paused', rank=rank)), flush=True)
            return
        reference_train, reference_val, reference_hash = reference
        if getattr(args, 'reference_only', False):
            print(json.dumps(dict(event='reference_complete', reference_sha256=reference_hash)), flush=True)
            return
        signature = dict(plan=plan, settings=settings, data_sha256=manifest_hash,
            tokenizer_sha256=tokenizer_hash, renderer=RENDER_FORMAT, mask=MASK_FORMAT,
            base_sha256=base_hash, reference_sha256=reference_hash,
            world_size=world, device_type=device.type, torch_version=torch.__version__, sources=source_identity())
        latest = work / 'latest.pt'
        resuming = latest.exists()
        saved = torch.load(latest if resuming else weights, map_location='cpu', weights_only=False, mmap=True)
        if resuming:
            validate_resume(saved, signature, world)
            origin = saved['origin']
        raw, cfg = load_model(saved, settings, device, sp)
        raw.requires_grad_(True)
        optimizer = torch.optim.AdamW(raw.parameters(), lr=settings['lr'], betas=(.9, .95),
            eps=1e-8, weight_decay=settings['weight_decay'], fused=device.type == 'cuda')
        step = cursor = tokens_seen = response_tokens_seen = 0
        if resuming:
            optimizer.load_state_dict(saved['optimizer'])
            step, cursor = saved['step'], saved['sample_cursor']
            tokens_seen, response_tokens_seen = saved['tokens_seen'], saved['response_tokens_seen']
        resume_rng = saved['rng_by_rank'][rank] if resuming else None
        del saved
        if world > 1:
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
            atomic_json(work / 'settings.json', dict(signature=signature, origin=origin,
                total_steps=total_steps, train_rows=len(train_data), val_rows=len(val_data)))
            if step < total_steps:
                (work / 'completion.json').unlink(missing_ok=True)
        pad_id = sp.pad_id() if sp.pad_id() >= 0 else sp.eos_id()
        last_validation = None
        last_validation_step = None

        def validation(full=False):
            values = [evaluate(raw, val_data, reference_val, settings, device, pad_id, full=full) if rank == 0 else None]
            if world > 1:
                dist.broadcast_object_list(values, src=0, device=device)
            return values[0]

        def metric(values):
            if rank == 0:
                values.update(timestamp=time.time(), stage='dpo', step=step, tokens_seen=tokens_seen,
                    response_tokens_seen=response_tokens_seen, sample_cursor=cursor,
                    objective=settings['objective'])
                with (work / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(values, allow_nan=False) + '\n')
                if values.get('event') == 'validation':
                    atomic_json(work / 'evaluation.json', dict(values, data_sha256=manifest_hash,
                        base_sha256=base_hash, reference_sha256=reference_hash,
                        tokenizer_sha256=tokenizer_hash, mask_format=MASK_FORMAT,
                        available_validation_samples=len(val_data)))
                print(json.dumps(values, allow_nan=False), flush=True)

        def save(writer):
            state = rng_state(device)
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, state)
            else:
                states = [state]
            writer.submit(state_factory=lambda: dict(format=FORMAT, signature=signature, origin=origin,
                config=vars(cfg), model=raw.state_dict(), optimizer=optimizer.state_dict(), step=step,
                sample_cursor=cursor, tokens_seen=tokens_seen, response_tokens_seen=response_tokens_seen,
                val_loss=last_validation['val_loss'], validation=last_validation,
                val_step=last_validation_step, rng_by_rank=states, completed=step >= total_steps,
                total_steps=total_steps))

        with DistributedCheckpointWriter(latest, rank=rank, device=device, distributed=world > 1) as writer:
            last_validation = validation(full=True)
            last_validation_step = step
            metric(dict(event='validation', full_holdout=True, **last_validation))
            last_saved_step = step if resuming else -1
            model.train()
            while step < end_step and not stop_requested(args, device, world):
                writer.check()
                indices, next_cursor = update_indices(len(train_data), settings['seed'], cursor,
                                                      settings['epochs'], global_batch)
                if not indices:
                    break
                microbatches = []
                for micro in range(settings['grad_accum']):
                    offset = (micro * world + rank) * settings['micro_batch']
                    selected = indices[offset:offset + settings['micro_batch']]
                    padded = [train_data[i] for i in selected] + [None] * (settings['micro_batch'] - len(selected))
                    x, y, active = collate_pairs(padded, settings['sequence_length'], pad_id)
                    ref = torch.zeros((settings['micro_batch'], 2), dtype=torch.float32)
                    if selected:
                        ref[:len(selected)] = torch.tensor(np.array(reference_train[selected]), dtype=torch.float32)
                    microbatches.append((x, y, active, ref))
                lr = learning_rate(step, total_steps, settings)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                started = time.perf_counter()
                values, norm = training_update(model, optimizer, microbatches, len(indices), world, device, settings)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - started
                step += 1
                cursor = next_cursor
                actual_tokens = sum(len(train_data[i][branch]['input_ids']) - 1
                                    for i in indices for branch in ('chosen', 'rejected'))
                tokens_seen += actual_tokens
                response_tokens_seen += sum(sum(value != -100 for value in train_data[i][branch]['labels'][1:])
                                            for i in indices for branch in ('chosen', 'rejected'))
                metric(dict(event='train', **values, lr=lr, grad_norm=norm, update_seconds=seconds,
                    tokens_per_second=actual_tokens / seconds, update_samples=len(indices)))
                if step % settings['eval_every'] == 0 or step == end_step:
                    full = step >= total_steps
                    last_validation = validation(full=full)
                    last_validation_step = step
                    metric(dict(event='validation', full_holdout=full, **last_validation))
                if step % settings['save_every'] == 0 or step == end_step:
                    save(writer)
                    last_saved_step = step
            if last_saved_step != step:
                save(writer)
            writer.flush()
            if rank == 0 and step >= total_steps:
                atomic_json(work / 'completion.json', dict(format='warpstate-dpo-completion-v1',
                    stage='dpo', completed=True, paused_for_evaluation=True, step=step,
                    total_steps=total_steps, sample_cursor=cursor, tokens_seen=tokens_seen,
                    response_tokens_seen=response_tokens_seen, val_loss=last_validation['val_loss'],
                    evaluation_file=str(work / 'evaluation.json'), checkpoint=str(latest),
                    reference_sha256=reference_hash, timestamp=time.time()))
            metric(dict(event='complete' if step >= total_steps else 'paused', total_steps=total_steps,
                        checkpoint=str(latest)))
        validate_cache(cache, identity, (len(train_data), len(val_data)))
        if digest(weights) != base_hash:
            raise RuntimeError('Frozen SFT initialization changed during DPO')
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'work-dir', 'data-dir', 'weights', 'tokenizer', 'reference-cache-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.stop_file is None:
        args.stop_file = args.work_dir / 'control/training-stop-request'
    if not args.stop_file.resolve().is_relative_to(WORKSPACE):
        parser.error('--stop-file must stay inside the workspace')
    def stop(signum, frame):
        global STOP
        STOP = True
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(signum, stop)
    from run_40b import process_lock
    validate_paths(args.work_dir, args.data_dir, args.weights, args.tokenizer, args.config)
    validate_reference_path(args)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.reference_cache_dir.mkdir(parents=True, exist_ok=True)
    plan, _ = load_plan(args.config)
    rank = int(os.environ.get('RANK', '0'))
    with process_lock(args.work_dir / f'rank-{rank}.lock'), \
            process_lock(args.reference_cache_dir / f'rank-{rank}.lock'), base_pipeline_lock(plan):
        train(args)


if __name__ == '__main__':
    main()
