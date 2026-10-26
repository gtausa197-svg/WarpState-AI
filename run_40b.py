#!/usr/bin/env python3
"""One entry point: isolated dependencies, resumable 40B corpus, full GPU training."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STOP = False
RECIPE = 1


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument('--weights', type=Path, help='Your existing trusted WarpState checkpoint; used only to initialize a new run')
    initialization.add_argument('--from-scratch', action='store_true', help='Initialize a new model from --config and --seed; existing runs still resume')
    p.add_argument('--work-dir', type=Path, default=ROOT / 'work')
    p.add_argument('--tokenizer', type=Path, default=ROOT / 'tokenizer.model')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/warpstate_2b.json')
    p.add_argument('--stage', choices=['all', 'prepare', 'train'], default='all')
    p.add_argument('--verify-data', action='store_true', help='Read the entire corpus and check SHA256 before training')
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-100BT')
    p.add_argument('--revision', default='main', help='Resolved to an immutable Hub commit on first preparation')
    p.add_argument('--train-tokens', type=int, default=40_000_000_000)
    p.add_argument('--val-tokens', type=int, default=1_048_576)
    p.add_argument('--workers', type=int, default=min(16, os.cpu_count() or 4))
    p.add_argument('--documents-per-batch', type=int, default=256)
    p.add_argument('--seq-len', type=int, default=2048)
    p.add_argument('--micro-batch', type=int, default=8)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--lr', type=float, default=1.5e-4)
    p.add_argument('--min-lr', type=float, default=1.5e-5)
    p.add_argument('--warmup-steps', type=int, default=1000)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--autocast-cache', action=argparse.BooleanOptionalAction, default=True,
                   help='Disable on a shared GPU to avoid retaining BF16 copies of all weights')
    p.add_argument('--vram-limit-gib', type=float, help='Cap the PyTorch CUDA allocator, excluding CUDA context/library memory')
    p.add_argument('--vram-headroom-gib', type=float, default=3.0,
                   help='Require this much additional free device memory before applying a VRAM limit')
    p.add_argument('--loss-chunk-size', type=int, default=1024)
    p.add_argument('--loss-mode', choices=['checkpointed', 'full', 'compiled-full'], default='checkpointed',
                   help='Full loss retains vocabulary logits to avoid head recomputation; requires additional VRAM')
    p.add_argument('--bf16-residuals', action=argparse.BooleanOptionalAction, default=False,
                   help='Use BF16 residual activations while retaining FP32 parameters, gradients and optimizer state')
    p.add_argument('--activation-mode', choices=['baseline', 'swiglu', 'rmsnorm', 'all'], default='baseline',
                   help='Optional BF16 activation/normalization kernels; requires explicit reconfiguration on resume')
    p.add_argument('--matmul-plan', type=Path,
                   help='Explicit BF16 shape/tile recipe; missing shapes retain ATen')
    p.add_argument('--gradient-mode', choices=['eager', 'compiled-autograd', 'compiled-autograd-overlap'], default='eager',
                   help='Optional compiled backward and FP32 gradient accumulation')
    p.add_argument('--backward-layout', choices=['baseline', 'inspect', 'paired'], default='baseline',
                   help='Opt-in production-shape packed attention backward graph experiment')
    p.add_argument('--overlap-swap-collectives', action='store_true',
                   help='Allow wait/copy groups past later collective launches, preserving launch order')
    p.add_argument('--overlap-runtime-estimates', action='store_true',
                   help='Use measured, rank-aligned GEMM estimates for communication scheduling')
    p.add_argument('--overlap-producer-schedule', action='store_true',
                   help='Move ready gradient producers with collectives under zero estimated memory growth')
    p.add_argument('--overlap-memory-budget', type=float, default=0.0,
                   help='Allowed fractional increase in estimated scheduler peak memory (0 to 0.25)')
    p.add_argument('--cache-core-weight-casts', action=argparse.BooleanOptionalAction, default=False,
                   help='Reuse differentiable BF16 core weight views across one accumulation update')
    p.add_argument('--ddp-bucket-cap-mb', type=float, default=128,
                   help='Gradient communication bucket capacity in MiB')
    p.add_argument('--ddp-batched-grad-copy', action=argparse.BooleanOptionalAction, default=False,
                   help='Batch FP32 gradient-to-bucket copying and scaling')
    p.add_argument('--compile', action='store_true', help='Optional Linux Inductor; initial compilation can take minutes')
    p.add_argument('--compile-core', action='store_true', help='Compile individual training blocks while preserving canonical checkpoint keys')
    p.add_argument('--compile-mode', choices=['default', 'reduce-overhead', 'max-autotune-no-cudagraphs'],
                   default='default', help='Compilation mode for --compile-core or --compile')
    p.add_argument('--reconfigure', action='store_true', help='Allow execution changes on resume while preserving context, global batch, data and optimizer schedule')
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--eval-batches', type=int, default=8)
    p.add_argument('--eval-micro-batch', type=int, default=1, help='Validation batch size independent of training microbatch')
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--async-checkpoint', action=argparse.BooleanOptionalAction, default=False,
                   help='Stage an immutable full CPU snapshot on rank zero, then overlap disk writing with training')
    p.add_argument('--max-steps', type=int, help='Stop and save after N additional steps, keeping the 40B schedule')
    p.add_argument('--stop-at-step', type=int,
                   help='Evaluate, save and stop at this absolute optimizer step, keeping the 40B schedule')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--no-bootstrap', action='store_true', help='Use an already provisioned environment')
    p.add_argument('--test-small-gpu', action='store_true', help='Tests only: bypass the 80 GiB device requirement')
    a = p.parse_args()
    for key in ('train_tokens','val_tokens','workers','documents_per_batch','seq_len','micro_batch','grad_accum',
                'loss_chunk_size','eval_every','eval_batches','eval_micro_batch','save_every','warmup_steps'):
        if getattr(a, key) < 1:
            p.error(key + ' must be positive')
    if a.max_steps is not None and a.max_steps < 1:
        p.error('--max-steps must be positive')
    if a.stop_at_step is not None and a.stop_at_step < 1:
        p.error('--stop-at-step must be positive')
    if a.vram_limit_gib is not None and (not math.isfinite(a.vram_limit_gib) or a.vram_limit_gib <= 0):
        p.error('--vram-limit-gib must be finite and positive')
    if not math.isfinite(a.vram_headroom_gib) or a.vram_headroom_gib < 0:
        p.error('--vram-headroom-gib must be finite and nonnegative')
    if not 0 < a.min_lr <= a.lr or a.weight_decay < 0:
        p.error('Invalid learning rate or weight decay')
    if a.train_tokens < 2 or a.val_tokens <= a.seq_len:
        p.error('Need at least 2 training tokens and more validation tokens than seq-len')
    if a.compile and a.compile_core:
        p.error('Choose --compile or --compile-core')
    if a.compile_mode != 'default' and not (a.compile or a.compile_core):
        p.error('A nondefault --compile-mode requires --compile-core or --compile')
    if not math.isfinite(a.ddp_bucket_cap_mb) or a.ddp_bucket_cap_mb * 1024**2 < 1:
        p.error('--ddp-bucket-cap-mb must be finite and at least one byte in MiB')
    if not math.isfinite(a.overlap_memory_budget) or not 0 <= a.overlap_memory_budget <= 0.25:
        p.error('--overlap-memory-budget must be finite and between 0 and 0.25')
    if a.gradient_mode != 'compiled-autograd-overlap' and (
            a.overlap_swap_collectives or a.overlap_runtime_estimates or a.overlap_memory_budget
            or a.overlap_producer_schedule):
        p.error('Overlap scheduling options require --gradient-mode compiled-autograd-overlap')
    if a.overlap_producer_schedule and (
            a.overlap_swap_collectives or a.overlap_runtime_estimates or a.overlap_memory_budget):
        p.error('--overlap-producer-schedule must be tested separately from other overlap scheduling options')
    if a.gradient_mode != 'eager' and a.ddp_batched_grad_copy:
        p.error('--ddp-batched-grad-copy uses the C++ reducer and cannot be combined with compiled autograd')
    if a.backward_layout != 'baseline':
        if (a.gradient_mode != 'compiled-autograd-overlap' or not a.compile_core
                or a.activation_checkpointing or not a.bf16_residuals
                or a.micro_batch != 4 or a.seq_len != 2048
                or a.activation_mode != 'baseline' or a.matmul_plan is not None
                or a.compile_mode != 'default'):
            p.error('--backward-layout requires compiled-autograd-overlap, --compile-core, '
                    '--no-activation-checkpointing, --bf16-residuals, microbatch 4, sequence 2048, '
                    'default compile mode, baseline activations and no matmul plan')
    if a.gradient_mode != 'eager' and a.activation_mode != 'baseline':
        p.error('Standalone activation modes require eager gradients; use baseline activations '
                'for compiled autograd, including matmul-plan epilogues')
    if a.cache_core_weight_casts and a.activation_checkpointing:
        p.error('--cache-core-weight-casts requires --no-activation-checkpointing')
    if a.cache_core_weight_casts and a.gradient_mode != 'eager':
        p.error('Core weight cast caching and compiled autograd must be tested separately')
    if a.compile_mode == 'reduce-overhead' and (a.cache_core_weight_casts or a.gradient_mode != 'eager'):
        p.error('Weight cast caching and compiled autograd require a compile mode without CUDA graphs')
    if a.matmul_plan is not None and a.cache_core_weight_casts:
        p.error('--matmul-plan cannot be combined with external core weight cast caching')
    if a.matmul_plan is not None and a.compile_mode == 'reduce-overhead':
        p.error('--matmul-plan currently requires a compile mode without CUDA graphs')
    if a.matmul_plan is not None and a.gradient_mode != 'eager' and a.activation_checkpointing:
        p.error('--matmul-plan with compiled autograd requires --no-activation-checkpointing')
    a.rank = int(os.environ.get('RANK', '0'))
    a.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    a.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if a.world_size > 1 and a.stage != 'train':
        p.error('torchrun supports --stage train only; prepare the corpus once beforehand')
    for name in ('work_dir','tokenizer','config','weights','matmul_plan'):
        if getattr(a,name) is not None:
            setattr(a,name,getattr(a,name).expanduser().resolve())
    return a


def bootstrap(a):
    if a.no_bootstrap:
        return
    if sys.platform != 'linux':
        raise RuntimeError('Automatic installation targets Linux. Tests may use --no-bootstrap.')
    if not (3, 12) <= sys.version_info[:2] <= (3, 14):
        raise RuntimeError('Use Python 3.12–3.14; Python 3.12 is recommended.')
    env = ROOT / '.server-venv'
    python = env / 'bin/python'
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', str(env)], check=True)
    marker = env / 'warpstate-deps-v1.json'
    if not marker.exists():
        subprocess.run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', 'torch==2.14.0',
                        '--index-url', 'https://download.pytorch.org/whl/cu130'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', 'numpy==2.5.3', 'sentencepiece==0.2.2',
                        'datasets==5.0.1', 'huggingface-hub==1.30.0'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
        freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
        (env / 'requirements-resolved.txt').write_text(freeze)
        marker.write_text(json.dumps({'installed': time.time()}))
    os.execv(str(python), [str(python), '-u', str(Path(__file__).resolve()), *sys.argv[1:], '--no-bootstrap'])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, writer):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


@contextlib.contextmanager
def process_lock(path):
    with Path(path).open('a+b') as f:
        if os.name == 'nt':
            import msvcrt
            f.seek(0)
            f.write(b'0')
            f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def data_signature(a):
    return {'recipe': RECIPE, 'dataset': a.dataset, 'config': a.dataset_config, 'requested_revision': a.revision,
            'train_tokens': a.train_tokens, 'val_tokens': a.val_tokens, 'tokenizer_sha256': digest(a.tokenizer),
            'dtype': '<u2', 'holdout': 'sha256(raw-text) modulo 1000 == 0; train excludes all holdout documents'}


def is_holdout(text):
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'little') % 1000 == 0


def check_manifest(a):
    directory = a.work_dir / 'data'
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['signature'] != data_signature(a) or not manifest['complete']:
        raise ValueError('Prepared dataset does not match requested settings/tokenizer')
    for split, count in [('train', a.train_tokens), ('val', a.val_tokens)]:
        if (directory / (split + '.bin')).stat().st_size != count * 2:
            raise ValueError('Invalid token file size: ' + split)
        if getattr(a, 'verify_data', False) and digest(directory / (split+'.bin')) != manifest['sha256'][split]:
            raise ValueError('Dataset checksum mismatch: '+split)
    return manifest


def prepare(a):
    import numpy as np
    import sentencepiece as spm
    import datasets
    from huggingface_hub import HfApi
    directory = a.work_dir / 'data'
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'manifest.json').exists():
        check_manifest(a)
        print('Dataset already complete and matches this tokenizer.', flush=True)
        return True
    signature = data_signature(a)
    sp = spm.SentencePieceProcessor(model_file=str(a.tokenizer))
    if sp.vocab_size() > 65536 or sp.eos_id() < 0:
        raise ValueError('Need a uint16-compatible tokenizer with EOS')
    progress = directory / 'prepare-state.pkl'
    if progress.exists():
        with progress.open('rb') as f:
            state = pickle.load(f)
        if state['signature'] != signature or state['datasets_version'] != datasets.__version__:
            raise ValueError('Preparation settings/version changed; use the original environment or a different work directory')
    else:
        if any((directory / (s+'.bin')).exists() for s in ('train','val')):
            raise RuntimeError('Token files without preparation state: refusing to overwrite')
        state = {'signature': signature, 'datasets_version': datasets.__version__, 'train': 0, 'val': 0,
                 'revision': HfApi().dataset_info(a.dataset, revision=a.revision).sha, 'iterator': None, 'documents': 0}
        atomic(progress, lambda f: pickle.dump(state, f))
    remaining_bytes = 2*(a.train_tokens+a.val_tokens-state['train']-state['val'])
    reserve = 70*1024**3 if not a.test_small_gpu else 0
    if shutil.disk_usage(directory).free < remaining_bytes + reserve:
        raise RuntimeError(f'Need {(remaining_bytes+reserve)/1024**3:.1f} GiB free for remaining data and checkpoints')
    ds = datasets.load_dataset(a.dataset, name=a.dataset_config, split='train', streaming=True, revision=state['revision'])
    print(f"Preparing {a.dataset}/{a.dataset_config} at {state['revision']}; "
          f"resuming train={state['train']:,}, val={state['val']:,}", flush=True)
    if state['iterator'] is not None:
        ds.load_state_dict(state['iterator'])
    iterator = iter(ds)
    with contextlib.ExitStack() as stack:
        stack.callback(lambda: getattr(iterator, 'close', lambda: None)())
        files = {}
        for split in ('train', 'val'):
            path = directory / (split+'.bin')
            if not path.exists() and state[split]:
                raise RuntimeError('Missing committed data: '+split)
            f = stack.enter_context(path.open('r+b' if path.exists() else 'w+b'))
            if path.stat().st_size < state[split]*2:
                raise RuntimeError('Truncated committed data: '+split)
            f.truncate(state[split]*2)  # Discard only uncommitted output after interruption.
            f.seek(state[split]*2)
            files[split] = f
        def commit():
            for f in files.values():
                f.flush()
                os.fsync(f.fileno())
            state['iterator'] = ds.state_dict()
            atomic(progress, lambda f: pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL))
        last_commit = state['train']+state['val']
        started = time.monotonic()
        start_count = last_commit
        while state['train'] < a.train_tokens or state['val'] < a.val_tokens:
            rows = list(itertools.islice(iterator, a.documents_per_batch))
            if not rows:
                commit()
                raise RuntimeError('Source exhausted before the requested token budget; do not repeat it silently')
            texts = [row['text'] for row in rows]
            encoded = sp.encode(texts, out_type=int, num_threads=a.workers)
            for text, ids in zip(texts, encoded):
                state['documents'] += 1
                if not text.strip():
                    continue
                split = 'val' if is_holdout(text) else 'train'
                target = a.val_tokens if split == 'val' else a.train_tokens
                room = target - state[split]
                if room <= 0:
                    continue
                tokens = np.asarray((ids+[sp.eos_id()])[:room], dtype='<u2')
                files[split].write(tokens.tobytes())
                state[split] += len(tokens)
            count = state['train']+state['val']
            if STOP or count-last_commit >= 8_000_000:
                commit()
                last_commit = count
                print(f"prepared train={state['train']:,}/{a.train_tokens:,}, val={state['val']:,}; "
                      f"{(count-start_count)/max(time.monotonic()-started,1):,.0f} tokens/s", flush=True)
            if STOP:
                return False
        commit()
    manifest = {'signature': signature, 'source_revision': state['revision'], 'complete': True,
                'documents_read': state['documents'], 'datasets_version': datasets.__version__,
                'sha256': {s:digest(directory/(s+'.bin')) for s in ('train','val')},
                'source_url': 'https://huggingface.co/datasets/'+a.dataset}
    atomic(directory / 'manifest.json', lambda f: f.write(json.dumps(manifest, indent=2).encode()))
    check_manifest(a)
    print('Prepared the exact requested token counts.', flush=True)
    return True


def batches(data, cursor, batch, length, valid, device):
    import numpy as np
    import torch
    # A cyclic contiguous pass: no random sampling with replacement over 40B tokens.
    indexes = (np.arange(batch*length+1, dtype=np.int64)+cursor) % len(data)
    tokens = np.asarray(data[indexes], dtype=np.int64)
    x = torch.from_numpy(tokens[:-1].reshape(batch, length).copy())
    y = torch.from_numpy(tokens[1:].reshape(batch, length).copy())
    y.view(-1)[valid:] = -100
    if device.type == 'cuda':
        x, y = x.pin_memory(), y.pin_memory()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def train_signature(a, manifest):
    keys = ('seq_len','micro_batch','grad_accum','train_tokens','lr','min_lr','warmup_steps','weight_decay',
            'seed','activation_checkpointing','loss_chunk_size','compile')
    settings = {k: getattr(a,k) for k in keys}
    settings.update(world_size=getattr(a, 'world_size', 1), compile_core=getattr(a, 'compile_core', False),
                    loss_mode=getattr(a, 'loss_mode', 'checkpointed'),
                    compile_mode=getattr(a, 'compile_mode', 'default'),
                    bf16_residuals=getattr(a, 'bf16_residuals', False),
                    activation_mode=getattr(a, 'activation_mode', 'baseline'),
                    gradient_mode=getattr(a, 'gradient_mode', 'eager'),
                    overlap_swap_collectives=getattr(a, 'overlap_swap_collectives', False),
                    overlap_runtime_estimates=getattr(a, 'overlap_runtime_estimates', False),
                    overlap_memory_budget=getattr(a, 'overlap_memory_budget', 0.0),
                    overlap_producer_schedule=getattr(a, 'overlap_producer_schedule', False),
                    matmul_execution=getattr(a, 'matmul_metadata', None),
                    backward_layout_execution=getattr(a, 'backward_layout_metadata', None),
                    cache_core_weight_casts=getattr(a, 'cache_core_weight_casts', False),
                    ddp_bucket_cap_mb=getattr(a, 'ddp_bucket_cap_mb', 128),
                    ddp_batched_grad_copy=getattr(a, 'ddp_batched_grad_copy', False))
    return {'settings': settings, 'data': manifest}


def check_resume_signature(old, new, reconfigure=False):
    """Execution can change explicitly; the objective and global schedule cannot."""
    def normalized(signature):
        settings = dict(signature['settings'])
        settings.setdefault('world_size', 1)
        settings.setdefault('compile_core', False)
        settings.setdefault('loss_mode', 'checkpointed')
        settings.setdefault('compile_mode', 'default')
        settings.setdefault('bf16_residuals', False)
        settings.setdefault('activation_mode', 'baseline')
        settings.setdefault('gradient_mode', 'eager')
        settings.setdefault('overlap_swap_collectives', False)
        settings.setdefault('overlap_runtime_estimates', False)
        settings.setdefault('overlap_memory_budget', 0.0)
        settings.setdefault('overlap_producer_schedule', False)
        settings.setdefault('matmul_execution', None)
        settings.setdefault('backward_layout_execution', None)
        settings.setdefault('cache_core_weight_casts', False)
        settings.setdefault('ddp_bucket_cap_mb', 128)
        settings.setdefault('ddp_batched_grad_copy', False)
        return {'settings': settings, 'data': signature['data']}
    old, new = normalized(old), normalized(new)
    if old == new:
        return False
    message = 'Resume settings/data differ; keep original settings or use --reconfigure for execution-only changes'
    if not reconfigure or old['data'] != new['data']:
        raise ValueError(message)
    execution = {'micro_batch', 'grad_accum', 'world_size', 'activation_checkpointing',
                 'loss_chunk_size', 'loss_mode', 'compile', 'compile_core', 'compile_mode',
                 'bf16_residuals', 'activation_mode', 'gradient_mode', 'cache_core_weight_casts', 'matmul_execution',
                 'overlap_swap_collectives', 'overlap_runtime_estimates', 'overlap_memory_budget',
                 'overlap_producer_schedule',
                 'ddp_bucket_cap_mb', 'ddp_batched_grad_copy', 'backward_layout_execution'}
    objective = lambda s: {k: v for k, v in s.items() if k not in execution}
    batch = lambda s: s['seq_len']*s['micro_batch']*s['grad_accum']*s['world_size']
    if objective(old['settings']) != objective(new['settings']) or batch(old['settings']) != batch(new['settings']):
        raise ValueError(message + '; objective and global tokens per update must stay fixed')
    return True


def update_slices(target, micro_tokens, world_size, rank):
    """Disjoint contiguous slices; empty ranks still participate in the final backward."""
    for iteration in range(math.ceil(target / (world_size*micro_tokens))):
        offset = (iteration*world_size + rank)*micro_tokens
        yield offset, max(0, min(micro_tokens, target-offset))


def compile_training_cores(model, mode='default'):
    import torch
    for core in model.cores:
        eager = core.forward
        compiled = torch.compile(eager, fullgraph=True, dynamic=False, mode=mode)
        def forward(*args, _core=core, _eager=eager, _compiled=compiled, **kwargs):
            # Validation does not need another compiled graph for a different batch size.
            fn = _compiled if _core.training and torch.is_grad_enabled() else _eager
            return fn(*args, **kwargs)
        core.forward = forward


def configure_memory(a):
    """Apply the cap before model allocation; this does not reserve another job's memory."""
    if a.vram_limit_gib is None:
        return
    import torch
    device = torch.device('cuda', getattr(a, 'local_rank', 0))
    free, total = torch.cuda.mem_get_info(device)
    limit = int(a.vram_limit_gib * 1024**3)
    required = limit + int(a.vram_headroom_gib * 1024**3)
    if limit > total or free < required:
        raise RuntimeError(f'Only {free/1024**3:.2f} GiB free; need {required/1024**3:.2f} GiB '
                           'for the requested allocator cap and headroom')
    torch.cuda.set_per_process_memory_fraction(limit / total, device)
    print(f'CUDA allocator cap {a.vram_limit_gib:.2f} GiB; free before model '
          f'{free/1024**3:.2f} GiB; autocast cache={a.autocast_cache}', flush=True)


def train(a):
    import numpy as np
    import sentencepiece as spm
    import torch
    import torch.distributed as dist
    from warpstate import WarpStateConfig, WarpStateLM
    manifest = check_manifest(a)
    matmul_plan = None
    if a.matmul_plan is not None:
        from ops.matmul_execution import load_matmul_plan
        matmul_plan = load_matmul_plan(a.matmul_plan)
    a.matmul_metadata = matmul_plan.metadata if matmul_plan else None
    a.backward_layout_metadata = None
    if a.backward_layout != 'baseline':
        from ops.backward_layout_execution import configure_backward_layout
        a.backward_layout_metadata = configure_backward_layout(a.backward_layout)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA GPU required')
    device = torch.device('cuda', a.local_rank)
    distributed = a.world_size > 1
    props = torch.cuda.get_device_properties(device)
    if not a.test_small_gpu and props.total_memory < 80*1024**3:
        raise RuntimeError('This preset requires the 96 GB RTX PRO 6000; use --test-small-gpu only for tiny tests')
    torch.set_num_threads(min(a.workers, 8))
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed(a.seed + a.rank)
    run = a.work_dir / 'run'
    run.mkdir(exist_ok=True)
    latest = run / 'latest.pt'
    resuming = latest.exists()
    source = latest if resuming else a.weights
    if not resuming and a.from_scratch:
        checkpoint = {}
    elif source is None or not source.is_file():
        raise ValueError('First launch requires --weights /path/to/your/checkpoint.pt')
    else:
        checkpoint = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    cfg = WarpStateConfig(**checkpoint.get('config', json.loads(a.config.read_text())))
    cfg.validate()
    if a.backward_layout != 'baseline':
        from ops.backward_layout_execution import validate_backward_layout_config
        validate_backward_layout_config(cfg)
    if a.seq_len % cfg.chunk_size:
        raise ValueError('seq-len must be divisible by model chunk_size')
    sp = spm.SentencePieceProcessor(model_file=str(a.tokenizer))
    if cfg.vocab_size != sp.vocab_size():
        raise ValueError('Tokenizer/model vocabulary mismatch; use the exact original tokenizer')
    signature = train_signature(a, manifest)
    changed = False
    if resuming:
        if checkpoint.get('format') != 'warpstate-server-v1':
            raise ValueError('Resume settings/data differ; unsupported checkpoint format')
        changed = check_resume_signature(checkpoint['signature'], signature, a.reconfigure)
    if not resuming and a.from_scratch:
        print(f'Initializing fresh FP32 weights on CPU with seed {a.seed}.', flush=True)
        model = WarpStateLM(cfg)
    else:
        with torch.device('meta'):
            model = WarpStateLM(cfg)
        model.load_state_dict(checkpoint['model'], assign=True)
    model.to(device=device, dtype=torch.float32).requires_grad_(True)
    # Prefer unrounded FP32 masters when importing the laptop's CPU-offload checkpoint.
    old_opt = checkpoint.get('optimizer', {})
    if not resuming and old_opt.get('kind') == 'cpu_offload_adamw':
        with torch.no_grad():
            for parameter, master in zip(model.parameters(), old_opt['masters'], strict=True):
                if parameter.shape != master.shape:
                    raise ValueError('Master weight shape mismatch')
                parameter.copy_(master)
    cfg.gradient_checkpointing = a.activation_checkpointing
    model.loss_chunk_size = a.loss_chunk_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9,0.95), eps=1e-8,
                                  weight_decay=a.weight_decay, fused=True)
    seen, step = 0, 0
    if resuming:
        optimizer.load_state_dict(checkpoint['optimizer'])
        seen, step = checkpoint['tokens_seen'], checkpoint['step']
        rank_states = checkpoint.get('rng_by_rank')
        if rank_states is None:
            rank_states = [{'torch': checkpoint['rng_torch'], 'cuda': checkpoint['rng_cuda'][0],
                            'numpy': checkpoint['rng_numpy'], 'python': checkpoint['rng_python']}]
        if a.rank < len(rank_states):
            state = rank_states[a.rank]
            torch.set_rng_state(state['torch'])
            torch.cuda.set_rng_state(state['cuda'], device)
            np.random.set_state(state['numpy'])
            random.setstate(state['python'])
        else:
            # A newly added rank gets a reproducible independent stream.
            random.seed(a.seed + a.rank)
            np.random.seed(a.seed + a.rank)
            torch.manual_seed(a.seed + a.rank)
            torch.cuda.manual_seed(a.seed + a.rank)
    origin = checkpoint.get('origin',
        {'initialization': 'random', 'seed': a.seed, 'previous_training_tokens': 0}
        if not resuming and a.from_scratch else
        {'path': str(source), 'previous_training_tokens': checkpoint.get('tokens_seen')})
    history = list(checkpoint.get('execution_history', []))
    if changed:
        history.append({'step': step, 'tokens': seen, 'timestamp': time.time(),
                        'from': checkpoint['signature']['settings'], 'to': signature['settings']})
        if a.rank == 0:
            print(f'Reconfiguring execution at step {step}, tokens={seen:,}; preserving model, optimizer and schedule.', flush=True)
    del old_opt, checkpoint
    count = model.num_parameters()
    if shutil.disk_usage(run).free < count*28:
        raise RuntimeError('Need free space for checkpoint plus atomic replacement (about 53 GiB for 2B)')
    raw = model
    from ops.gradient_execution import OverlapSettings, make_backward, scale_loss
    from ops.distributed_execution import ddp_kwargs, ddp_runtime_settings
    overlap_settings = (OverlapSettings(swap_with_collectives=a.overlap_swap_collectives,
                                        use_runtime_estimations=a.overlap_runtime_estimates,
                                        producer_schedule=a.overlap_producer_schedule,
                                        peak_memory_budget=a.overlap_memory_budget)
                        if a.gradient_mode == 'compiled-autograd-overlap' else None)
    backward = make_backward(a.gradient_mode, distributed=distributed, overlap_settings=overlap_settings)
    ddp_settings = ddp_runtime_settings(bucket_cap_mb=a.ddp_bucket_cap_mb,
                                        batched_grad_copy=a.ddp_batched_grad_copy)
    if a.activation_mode != 'baseline':
        from ops.activation_execution import apply_activation_execution
        apply_activation_execution(raw, a.activation_mode)
    if matmul_plan is not None:
        from ops.matmul_execution import apply_matmul_execution
        apply_matmul_execution(raw, matmul_plan)
    if a.compile_core:
        compile_training_cores(raw, mode=a.compile_mode)
    if a.loss_mode != 'checkpointed':
        from ops.loss_experiment import apply_loss_execution
        apply_loss_execution(raw, a.loss_mode)
    if a.bf16_residuals:
        from ops.residual_experiment import apply_bf16_residuals
        apply_bf16_residuals(raw)
    weight_cast_cache = None
    if a.cache_core_weight_casts:
        from ops.weight_cast_execution import apply_weight_cast_cache
        weight_cast_cache = apply_weight_cast_cache(raw)
    if a.compile:
        model = torch.compile(model, mode=a.compile_mode)  # Optional: GPU-only optimizer, no CPU offload hooks.
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[a.local_rank],
            output_device=a.local_rank, **ddp_kwargs(ddp_settings))
    graph_gradients = (a.compile or a.compile_core) and a.compile_mode == 'reduce-overhead'
    if graph_gradients:
        # Replay must accumulate into stable buffers outside the graph's private pool.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
    train_data = np.memmap(a.work_dir/'data/train.bin', dtype='<u2', mode='r')
    val_data = np.memmap(a.work_dir/'data/val.bin', dtype='<u2', mode='r')
    if a.rank == 0:
        print(f'{a.world_size} x {props.name}: {props.total_memory/1024**3:.1f} GiB/device; {count:,} parameters; '
              f'FP32 GPU AdamW + BF16 autocast; start tokens={seen:,}', flush=True)
    batch_tokens = a.micro_batch*a.seq_len
    update_tokens = batch_tokens*a.grad_accum*a.world_size
    total_steps = math.ceil(a.train_tokens/update_tokens)
    end_step = total_steps
    if a.max_steps is not None:
        end_step = min(end_step, step + a.max_steps)
    if a.stop_at_step is not None:
        end_step = min(end_step, a.stop_at_step)
    if a.rank == 0:
        if end_step < total_steps:
            print(f'Pause boundary: optimizer step {end_step}; loaded step {step}; '
                  f'learning-rate schedule remains {total_steps} steps.', flush=True)
            if step >= end_step:
                print('Pause boundary already reached; no further optimizer updates.', flush=True)
        (run/'settings.json').write_text(json.dumps({'signature':signature, 'gpu':props.name,
            'torch':torch.__version__, 'cuda':torch.version.cuda, 'parameters':count, 'origin':origin,
            'execution_history': history,
            'runtime': {'stop_at_step': a.stop_at_step, 'max_steps': a.max_steps, 'end_step': end_step,
                        'vram_limit_gib': a.vram_limit_gib, 'vram_headroom_gib': a.vram_headroom_gib,
                        'autocast_cache': a.autocast_cache, 'eval_micro_batch': a.eval_micro_batch,
                        'eval_every': a.eval_every, 'save_every': a.save_every,
                        'async_checkpoint': a.async_checkpoint,
                        'bf16_residuals': a.bf16_residuals,
                        'activation_mode': a.activation_mode, 'gradient_mode': a.gradient_mode,
                        'overlap_swap_collectives': a.overlap_swap_collectives,
                        'overlap_runtime_estimates': a.overlap_runtime_estimates,
                        'overlap_memory_budget': a.overlap_memory_budget,
                        'overlap_producer_schedule': a.overlap_producer_schedule,
                        'matmul_execution': a.matmul_metadata,
                        'backward_layout_execution': a.backward_layout_metadata,
                        'cache_core_weight_casts': a.cache_core_weight_casts,
                        **ddp_settings,
                        'blas_preference': str(torch.backends.cuda.preferred_blas_library()),
                        'cublaslt_workspace_size_env': os.getenv('CUBLASLT_WORKSPACE_SIZE'),
                        'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES')}}, indent=2))
    def evaluate():
        model.eval()
        values = []
        if a.rank == 0:
            validation_tokens = a.eval_micro_batch*a.seq_len
            with torch.no_grad():
                for cursor in np.linspace(0, max(0,len(val_data)-validation_tokens-1), a.eval_batches, dtype=np.int64):
                    x,y = batches(val_data,int(cursor),a.eval_micro_batch,a.seq_len,validation_tokens,device)
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=a.autocast_cache):
                        _, loss = raw(x,y,return_logits=False)
                    values.append(float(loss))
        model.train()
        value = sum(values)/len(values) if values else 0.0
        if distributed:
            tensor = torch.tensor(value, dtype=torch.float64, device=device)
            dist.broadcast(tensor, src=0)
            value = float(tensor)
        if not math.isfinite(value):
            raise RuntimeError('Non-finite validation loss')
        return value
    def save(validation):
        state = {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device),
                 'numpy': np.random.get_state(), 'python': random.getstate()}
        rank_states = [state]
        if distributed:
            rank_states = [None]*a.world_size
            dist.all_gather_object(rank_states, state)
        def build_state():
            print('Staging full CPU checkpoint...' if a.async_checkpoint else
                  'Saving full GPU-optimizer checkpoint...', flush=True)
            return {'format':'warpstate-server-v1', 'signature':signature, 'origin':origin,
                   'execution_history': history,
                   'config':vars(cfg), 'model':raw.state_dict(), 'optimizer':optimizer.state_dict(),
                   'step':step, 'tokens_seen':seen, 'val_loss':validation,
                   'val_step':last_val_step, 'val_tokens':last_val_tokens, 'rng_by_rank': rank_states,
                   'rng_torch':state['torch'], 'rng_cuda':[state['cuda']],
                   'rng_numpy':state['numpy'], 'rng_python':state['python']}
        if checkpoint_writer is not None:
            # Every rank waits for the frozen CPU snapshot or a coordinated error.
            # Only rank zero writes; subsequent updates overlap its CPU thread.
            staging_seconds = checkpoint_writer.submit(state_factory=build_state)
            if a.rank == 0:
                print(f'Staged step {step}, tokens={seen:,} for asynchronous checkpoint; '
                      f'synchronous CPU staging {staging_seconds:.2f}s.', flush=True)
        else:
            if a.rank == 0:
                obj = build_state()
                atomic(latest, lambda f: torch.save(obj,f))
                print('Saved '+str(latest), flush=True)
            if distributed:
                dist.barrier()
    def stop_requested():
        stop_file = os.environ.get('WARPSTATE_STOP_FILE')
        requested = STOP or bool(stop_file and Path(stop_file).is_file())
        if not distributed:
            return requested
        flag = torch.tensor(int(requested), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())
    checkpoint_writer = None
    if a.async_checkpoint:
        from ops.checkpoint_writer import DistributedCheckpointWriter
        checkpoint_writer = DistributedCheckpointWriter(latest, rank=a.rank,
                                                        device=device, distributed=distributed)
    with checkpoint_writer if checkpoint_writer is not None else contextlib.nullcontext():
        last_val = evaluate()
        last_val_step, last_val_tokens = step, seen
        if a.rank == 0:
            print(f'initial validation loss: {last_val:.5f}', flush=True)
        while seen < a.train_tokens and step < end_step:
            if checkpoint_writer is not None:
                checkpoint_writer.check()
            if stop_requested():
                save(last_val)
                break
            torch.cuda.synchronize()
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=not graph_gradients)
            target = min(update_tokens,a.train_tokens-seen)
            warm = min(a.warmup_steps,total_steps)
            rate = (a.lr*(step+1)/warm if step < warm else a.min_lr+(a.lr-a.min_lr)*0.5*
                    (1+math.cos(math.pi*(step-warm)/max(1,total_steps-warm))))
            for group in optimizer.param_groups:
                group['lr'] = rate
            loss_sum = torch.zeros((),device=device)
            slices = list(update_slices(target, batch_tokens, a.world_size, a.rank))
            if a.backward_layout == 'paired':
                from ops.backward_layout_execution import backward_layout_stats, require_backward_layout_dispatch
                layout_calls_before = backward_layout_stats()['runtime_calls']
            with weight_cast_cache.update() if weight_cast_cache else contextlib.nullcontext():
                for index, (offset, valid) in enumerate(slices):
                    x,y = batches(train_data,seen+offset,a.micro_batch,a.seq_len,valid,device)
                    sync = model.no_sync() if distributed and index < len(slices)-1 else contextlib.nullcontext()
                    with sync:
                        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=a.autocast_cache):
                            _, loss = model(x,y,return_logits=False)
                        scaled = scale_loss(loss, valid*a.world_size/target, a.gradient_mode)
                        backward(scaled)
                    loss_sum += loss.detach()*(valid/target)
            if a.backward_layout == 'paired':
                require_backward_layout_dispatch(min_calls=layout_calls_before + cfg.logical_depth * len(slices))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            optimizer.step()
            if distributed:
                dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begin
            gpu_stats = [{'allocated': torch.cuda.max_memory_allocated(device)/1024**3,
                          'reserved': torch.cuda.max_memory_reserved(device)/1024**3,
                          'free': torch.cuda.mem_get_info(device)[0]/1024**3, 'seconds': elapsed}]
            if distributed:
                local_stats = gpu_stats[0]
                gpu_stats = [None]*a.world_size
                dist.all_gather_object(gpu_stats, local_stats)
                elapsed = max(item['seconds'] for item in gpu_stats)
            seen += target
            step += 1
            record = {'step':step,'tokens':seen,'loss':float(loss_sum),'lr':rate,'grad_norm':float(norm),
                      'tokens_per_second':target/elapsed,'seconds':elapsed,'timestamp':time.time(),
                      'world_size': a.world_size, 'gpu_memory': gpu_stats,
                      'peak_vram_gib':max(item['allocated'] for item in gpu_stats),
                      'peak_reserved_vram_gib':max(item['reserved'] for item in gpu_stats),
                      'device_free_vram_gib':min(item['free'] for item in gpu_stats)}
            torch.cuda.reset_peak_memory_stats()
            ending = stop_requested() or seen == a.train_tokens or step >= end_step
            if step % a.eval_every == 0 or ending:
                last_val = evaluate()
                last_val_step, last_val_tokens = step, seen
                if a.rank == 0:
                    with (run/'validation.jsonl').open('a') as f:
                        f.write(json.dumps({'step':step,'tokens':seen,'loss':last_val})+'\n')
            record.update(val_loss=last_val, val_step=last_val_step, val_tokens=last_val_tokens)
            if a.backward_layout != 'baseline':
                from ops.backward_layout_execution import backward_layout_stats
                record['backward_layout'] = backward_layout_stats()
            if a.rank == 0:
                validation_log = f'val {last_val:.5f} (step {last_val_step}) | ' if last_val_step == step else ''
                print(f"step {step}/{total_steps} | {seen:,}/{a.train_tokens:,} tokens | loss {record['loss']:.4f} | "
                      f"{validation_log}"
                      f"{record['tokens_per_second']:,.0f} tok/s | peak {record['peak_vram_gib']:.2f} GiB allocated / "
                      f"{record['peak_reserved_vram_gib']:.2f} GiB reserved | free {record['device_free_vram_gib']:.2f} GiB",flush=True)
                with (run/'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(record)+'\n')
            if step % a.save_every == 0 or ending:
                save(last_val)
            if ending:
                break


def request_stop(signum, frame):
    global STOP
    STOP = True
    print('Stop requested; committing preparation or finishing the current optimizer update.', flush=True)


def preflight(a):
    """Reject incompatible weights/hardware before a multi-day corpus preparation."""
    import torch
    import sentencepiece as spm
    from warpstate import WarpStateConfig, WarpStateLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Install a CUDA 13.0-compatible NVIDIA driver; BF16 CUDA is required')
    if not a.test_small_gpu and torch.cuda.get_device_properties(a.local_rank).total_memory < 80*1024**3:
        raise RuntimeError('The normal preset requires the 96 GB GPU')
    latest = a.work_dir/'run/latest.pt'
    source = latest if latest.exists() else a.weights
    fresh = not latest.exists() and a.from_scratch
    checkpoint = {} if fresh else torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    if not fresh and 'model' not in checkpoint:
        raise ValueError('Expected a WarpState checkpoint with a model state dictionary, not arbitrary model weights')
    cfg = WarpStateConfig(**checkpoint.get('config',json.loads(a.config.read_text())))
    cfg.validate()
    if cfg.vocab_size != spm.SentencePieceProcessor(model_file=str(a.tokenizer)).vocab_size():
        raise ValueError('Tokenizer/model vocabulary mismatch')
    if a.seq_len % cfg.chunk_size:
        raise ValueError('Sequence length must be divisible by the local tile size')
    with torch.device('meta'):
        probe = WarpStateLM(cfg)
    if not fresh:
        probe.load_state_dict(checkpoint['model'], assign=True, strict=True)
    if not a.test_small_gpu and not 2_000_000_000 <= probe.num_parameters() <= 2_020_000_000:
        raise ValueError('This package expects the approximately 2B WarpState architecture')
    optimizer_state = checkpoint.get('optimizer', {})
    if optimizer_state.get('kind') == 'cpu_offload_adamw':
        for parameter, master in zip(probe.parameters(), optimizer_state['masters'], strict=True):
            if parameter.shape != master.shape:
                raise ValueError('FP32 master weight shape mismatch')
    if latest.exists() and checkpoint.get('format') != 'warpstate-server-v1':
        raise ValueError('Use --weights to initialize a new server run; do not place a laptop checkpoint in work/run')
    print(f'Preflight: compatible WarpState {"configuration for fresh initialization" if fresh else "weights"} '
          f'({probe.num_parameters():,} parameters) and CUDA GPU.', flush=True)


def main():
    a = arguments()
    if not a.tokenizer.is_file():
        raise FileNotFoundError(a.tokenizer)
    if (a.stage != 'prepare' and not (a.work_dir/'run/latest.pt').exists() and not a.from_scratch
            and (a.weights is None or not a.weights.is_file())):
        raise ValueError('Supply --weights or explicitly choose --from-scratch before training')
    bootstrap(a)
    a.work_dir.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('HF_HOME',str(a.work_dir/'hf-cache'))
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(a.work_dir/'compile-cache'))
    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, 'SIGUSR1'):
        signals.append(signal.SIGUSR1)
    for sig in signals:
        signal.signal(sig,request_stop)
    distributed = a.world_size > 1
    if a.stage != 'prepare':
        import torch
        torch.cuda.set_device(a.local_rank)
        configure_memory(a)
        if distributed:
            import torch.distributed as dist
            dist.init_process_group('nccl', device_id=torch.device('cuda', a.local_rank))
    try:
        with contextlib.ExitStack() as stack:
            if a.rank == 0:
                stack.enter_context(process_lock(a.work_dir/'pipeline.lock'))
            if distributed:
                dist.barrier()
            if a.stage != 'prepare':
                preflight(a)
            if a.stage in ('all','prepare') and not prepare(a):
                return
            if a.stage in ('all','train'):
                train(a)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
