#!/usr/bin/env python3
"""Measure disposable full-model AdamW updates without changing training state.

Source ops/training-env.sh before launching. For two GPUs, use torchrun
--standalone --nproc-per-node=2 with the same arguments. Each configuration must
use a new output directory under work/benchmark-*. No checkpoints are written.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent
GIB = 1024**3
sys.path.insert(0, str(PACKAGE))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    run_id = os.environ.get('TORCHELASTIC_RUN_ID') or dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%S')
    parser.add_argument('--checkpoint', type=Path, default=PACKAGE / 'work/run/latest.pt',
                        help='Trusted local checkpoint; mmap-loaded read-only and copied to CUDA')
    parser.add_argument('--data', type=Path, default=PACKAGE / 'work/data/train.bin')
    parser.add_argument('--output-dir', type=Path,
                        default=PACKAGE / 'work' / ('benchmark-' + run_id))
    parser.add_argument('--micro-batch', type=int, required=True)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--global-tokens-per-update', type=int, default=131072)
    parser.add_argument('--activation-checkpointing', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--autocast-cache', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--loss-chunk-size', type=int, default=1024)
    parser.add_argument('--loss-mode', choices=['checkpointed', 'full', 'compiled-full'], default='checkpointed',
                        help='Disposable full-vocabulary loss execution experiment; model/objective stay the same')
    parser.add_argument('--bf16-residuals', action='store_true',
                        help='Experimental BF16 residual stream with unchanged FP32 master parameters')
    parser.add_argument('--compile-cores', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--compile-mode', choices=['default', 'reduce-overhead', 'max-autotune-no-cudagraphs'],
                        default='default')
    parser.add_argument('--blas-backend', choices=['default', 'cublas', 'cublaslt'], default=None,
                        help='Explicit PyTorch BLAS preference for a controlled comparison; '
                             'when omitted, retain the environment/library preference')
    parser.add_argument('--fused-rms-norm', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--activation-mode', choices=['baseline', 'swiglu', 'rmsnorm', 'all'],
                        default='baseline', help='Opt-in activation/normalization kernels')
    parser.add_argument('--matmul-plan', type=Path)
    parser.add_argument('--gradient-mode', choices=['eager', 'compiled-autograd', 'compiled-autograd-overlap'], default='eager',
                        help='Capture backward and gradient accumulation with compiled autograd')
    parser.add_argument('--backward-layout', choices=['baseline', 'inspect', 'paired'], default='baseline',
                        help='Opt-in production-shape packed attention backward graph experiment')
    parser.add_argument('--overlap-swap-collectives', action='store_true',
                        help='Allow wait/copy groups past later collective launches, preserving launch order')
    parser.add_argument('--overlap-runtime-estimates', action='store_true',
                        help='Use measured, rank-aligned GEMM estimates for communication scheduling')
    parser.add_argument('--overlap-producer-schedule', action='store_true',
                        help='Move ready gradient producers with collectives under zero estimated memory growth')
    parser.add_argument('--overlap-memory-budget', type=float, default=0.0,
                        help='Allowed fractional increase in estimated scheduler peak memory (0 to 0.25)')
    parser.add_argument('--cache-core-weight-casts', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ddp-batched-grad-copy', action=argparse.BooleanOptionalAction, default=False,
                        help='Batch gradient-to-bucket copies and scaling in the C++ DDP reducer')
    parser.add_argument('--warmup-updates', type=int, default=1)
    parser.add_argument('--measured-updates', type=int, default=2)
    parser.add_argument('--profile', action='store_true',
                        help='Trace only the first measured full update after warmup; timings include profiler overhead')
    parser.add_argument('--diagnose-stalls', action='store_true',
                        help='Record host stage timings without extra CUDA synchronization; dump Python stacks '
                             'every 10 seconds during measured updates')
    parser.add_argument('--cudagraph-diagnostics', action='store_true',
                        help='Count successful nonempty CUDA graph replays using installed PyTorch internals; '
                             'adds host instrumentation only, for disposable comparisons')
    parser.add_argument('--optimizer-state', choices=['resume', 'fresh'], default='resume')
    parser.add_argument('--vram-limit-gib', type=float)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--ddp-bucket-cap-mb', type=int, default=128)
    args = parser.parse_args()
    for name in ('micro_batch', 'seq_len', 'global_tokens_per_update', 'loss_chunk_size',
                 'measured_updates', 'threads', 'ddp_bucket_cap_mb'):
        if getattr(args, name) < 1:
            parser.error(name + ' must be positive')
    if args.warmup_updates < 0:
        parser.error('--warmup-updates must be nonnegative')
    if args.vram_limit_gib is not None and (not math.isfinite(args.vram_limit_gib) or args.vram_limit_gib <= 0):
        parser.error('--vram-limit-gib must be finite and positive')
    if args.gradient_mode != 'eager' and args.ddp_batched_grad_copy:
        parser.error('Batched gradient copy requires the C++ reducer; test it separately from compiled autograd')
    if args.backward_layout != 'baseline':
        if (args.gradient_mode != 'compiled-autograd-overlap' or not args.compile_cores
                or args.activation_checkpointing or not args.bf16_residuals
                or args.micro_batch != 4 or args.seq_len != 2048
                or args.activation_mode != 'baseline' or args.matmul_plan is not None
                or args.compile_mode != 'default' or args.fused_rms_norm):
            parser.error('--backward-layout requires compiled-autograd-overlap, --compile-cores, '
                         '--no-activation-checkpointing, --bf16-residuals, microbatch 4, sequence 2048, '
                         'default compile mode, baseline activations/norm and no matmul plan')
    if not math.isfinite(args.overlap_memory_budget) or not 0 <= args.overlap_memory_budget <= 0.25:
        parser.error('--overlap-memory-budget must be finite and between 0 and 0.25')
    if args.gradient_mode != 'compiled-autograd-overlap' and (
            args.overlap_swap_collectives or args.overlap_runtime_estimates or args.overlap_memory_budget
            or args.overlap_producer_schedule):
        parser.error('Overlap scheduling options require --gradient-mode compiled-autograd-overlap')
    if args.overlap_producer_schedule and (
            args.overlap_swap_collectives or args.overlap_runtime_estimates or args.overlap_memory_budget):
        parser.error('--overlap-producer-schedule must be tested separately from other overlap scheduling options')
    if args.gradient_mode != 'eager' and args.activation_mode != 'baseline':
        parser.error('Standalone activation modes require eager gradients; use baseline activations '
                     'for compiled autograd, including matmul-plan epilogues')
    if args.activation_mode != 'baseline' and args.fused_rms_norm:
        parser.error('Choose an activation mode or the ATen fused RMSNorm experiment')
    if args.cache_core_weight_casts and args.activation_checkpointing:
        parser.error('--cache-core-weight-casts requires --no-activation-checkpointing')
    if args.cache_core_weight_casts and args.gradient_mode != 'eager':
        parser.error('Core weight cast caching and compiled autograd must be tested separately')
    if args.compile_mode == 'reduce-overhead' and (args.cache_core_weight_casts or args.gradient_mode != 'eager'):
        parser.error('Weight cast caching and compiled autograd require a compile mode without CUDA graphs')
    if args.matmul_plan is not None:
        args.matmul_plan = args.matmul_plan.expanduser().resolve()
        if not args.matmul_plan.is_relative_to(WORKSPACE):
            parser.error('--matmul-plan must stay inside the WarpState workspace')
        if args.cache_core_weight_casts or args.compile_mode == 'reduce-overhead':
            parser.error('--matmul-plan cannot use external core cast caching or CUDA graphs')
        if args.gradient_mode != 'eager' and args.activation_checkpointing:
            parser.error('--matmul-plan with compiled autograd requires --no-activation-checkpointing')
    for name in ('checkpoint', 'data', 'output_dir'):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_relative_to(WORKSPACE):
            parser.error(name + ' must be inside ' + str(WORKSPACE))
        setattr(args, name, path)
    relative_output = args.output_dir.relative_to(PACKAGE / 'work') if args.output_dir.is_relative_to(PACKAGE / 'work') else None
    if relative_output is None or not relative_output.parts or not relative_output.parts[0].startswith('benchmark-'):
        parser.error('--output-dir must be below rtx6000_40b/work/benchmark-*')
    for name in ('TMPDIR', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR'):
        if not os.environ.get(name) or not Path(os.environ[name]).resolve().is_relative_to(WORKSPACE):
            parser.error('Source ops/training-env.sh first; ' + name + ' must stay inside the workspace')
    return args


def file_identity(path):
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns}


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_optimizer_state(value, device):
    """Detach every tensor from the checkpoint mapping, including Adam step counters."""
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: copy_optimizer_state(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_optimizer_state(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_optimizer_state(item, device) for item in value)
    return value


def timed_stage_markers(totals, annotate):
    """Measure host wall time, including implicit waits, without synchronizing CUDA."""
    @contextlib.contextmanager
    def mark(name):
        with annotate(name):
            begin = time.perf_counter()
            try:
                yield
            finally:
                key = name.removeprefix('warpstate::')
                totals[key] = totals.get(key, 0.0) + time.perf_counter() - begin
    return mark


@contextlib.contextmanager
def stalled_update_tracebacks(enabled):
    """Keep the diagnostic watchdog scoped to one measured benchmark update."""
    if not enabled:
        yield
        return
    import faulthandler
    faulthandler.dump_traceback_later(10, repeat=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


def kernel_category(name):
    """Coarse name-based labels; preserve raw kernels for manual attribution."""
    name = name.lower()
    if 'nccl' in name:
        return 'communication'
    if 'memcpy' in name or 'memset' in name:
        return 'copies_and_fills'
    if 'adam' in name:
        return 'optimizer'
    if any(part in name for part in ('flash', 'fmha', 'efficient_attention', 'scaled_dot_product')):
        return 'attention'
    if any(part in name for part in ('gemm', 'xmma', 'matmul', 'bmm', 'persistent_tma', 'swiglu_mm')):
        return 'matrix_multiply'
    if 'triton_' in name:
        return 'compiled_pointwise_or_reduction'
    return 'other'


def cuda_activity_summary(trace_events):
    """Count real CUDA activities; GPU annotation ranges contain the same work."""
    kernels = {}
    intervals = []
    excluded_annotations = 0
    for event in trace_events:
        if event.get('ph') != 'X':
            continue
        if event.get('cat') == 'gpu_user_annotation':
            excluded_annotations += 1
        if event.get('cat') not in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            continue
        name = event['name']
        device_index = event.get('args', {}).get('device', event.get('pid'))
        key = (name, device_index)
        row = kernels.setdefault(key, {'name': name, 'device_index': device_index,
                                       'category': kernel_category(name), 'count': 0, 'self_cuda_us': 0.0})
        row['count'] += 1
        row['self_cuda_us'] += event['dur']
        intervals.append((event['ts'], event['ts'] + event['dur']))
    kernels = sorted(kernels.values(), key=lambda row: row['self_cuda_us'], reverse=True)
    total_kernel_us = sum(row['self_cuda_us'] for row in kernels)
    categories = {}
    for row in kernels:
        category = categories.setdefault(row['category'], {'self_cuda_us': 0.0, 'count': 0})
        category['self_cuda_us'] += row['self_cuda_us']
        category['count'] += row['count']
    for category in categories.values():
        category['fraction_of_summed_cuda_time'] = category['self_cuda_us'] / total_kernel_us if total_kernel_us else 0.0
    # A union avoids counting overlapping streams twice when estimating CUDA occupancy.
    active_us = 0.0
    previous_end = None
    for begin, end in sorted(intervals):
        active_us += max(0.0, end - max(begin, previous_end if previous_end is not None else begin))
        previous_end = max(previous_end if previous_end is not None else end, end)
    return {'kernels': kernels, 'kernel_name_count': len(kernels), 'categories': categories,
            'summed_cuda_event_us': total_kernel_us, 'cuda_event_union_us': active_us,
            'cuda_activity_filter': ['kernel', 'gpu_memcpy', 'gpu_memset'],
            'excluded_gpu_annotation_ranges': excluded_annotations}


def profile_operator_table(operators):
    lines = ['self_cuda_ms\ttotal_cuda_ms\tself_cpu_ms\tcount\toperator\tinput_shapes']
    for row in operators[:100]:
        lines.append(f"{row['self_cuda_us']/1000:.3f}\t{row['total_cuda_us']/1000:.3f}\t"
                     f"{row['self_cpu_us']/1000:.3f}\t{row['count']}\t{row['name']}\t"
                     + json.dumps(row['input_shapes']))
    return '\n'.join(lines) + '\n'


def export_profile(profiler, output_dir, rank, update, elapsed):
    """Export both operator attribution and CUDA activities without mixing totals."""
    import torch
    prefix = output_dir / f'profile-rank-{rank}'
    trace_path = prefix.with_name(prefix.name + '-trace.json')
    profiler.export_chrome_trace(str(trace_path))
    averages = profiler.key_averages(group_by_input_shape=True)
    operators = []
    for event in averages:
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        operators.append({'name': event.key, 'input_shapes': event.input_shapes, 'count': event.count,
                          'self_cuda_us': event.self_device_time_total,
                          'total_cuda_us': event.device_time_total,
                          'self_cpu_us': event.self_cpu_time_total, 'total_cpu_us': event.cpu_time_total})
    operators.sort(key=lambda row: row['self_cuda_us'], reverse=True)
    activity = cuda_activity_summary(json.loads(trace_path.read_text()).get('traceEvents', []))
    summary = {'rank': rank, 'update': update, 'profiled_update_seconds': elapsed,
               'trace': str(trace_path), 'operator_count': len(operators), 'operators': operators,
               'aggregation_source_sha256': source_hash(Path(__file__)), **activity,
               'limitations': ['Profiling includes instrumentation overhead; use unprofiled runs for throughput.',
                               'CUDA event times can overlap across streams; summed category time is not wall time.',
                               'Operator and kernel CUDA times describe the same work; do not add them.',
                               'Kernel categories are name-based heuristics; raw names and operator shapes are retained.']}
    summary_path = prefix.with_name(prefix.name + '-summary.json')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    table_path = prefix.with_name(prefix.name + '-operators.txt')
    table_path.write_text(profile_operator_table(operators))
    kernel_lines = ['self_cuda_ms\tcount\tcategory\tkernel']
    kernel_lines.extend(f"{row['self_cuda_us']/1000:.3f}\t{row['count']}\t{row['category']}\t{row['name']}"
                        for row in activity['kernels'])
    prefix.with_name(prefix.name + '-kernels.tsv').write_text('\n'.join(kernel_lines) + '\n')
    return {'summary': str(summary_path), 'trace': str(trace_path), 'operators': str(table_path),
            **{key: value for key, value in activity.items() if key != 'kernels'}}


def benchmark(args, cudagraph_diagnostics=None):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from run_40b import batches
    from warpstate import WarpStateConfig, WarpStateLM

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    batch_tokens = args.micro_batch * args.seq_len
    divisor = batch_tokens * world_size
    if args.global_tokens_per_update % divisor:
        raise ValueError('Global tokens/update must be divisible by micro_batch * seq_len * world_size')
    accumulation = args.global_tokens_per_update // divisor
    matmul_plan = None
    if args.matmul_plan is not None:
        from ops.matmul_execution import load_matmul_plan
        matmul_plan = load_matmul_plan(args.matmul_plan)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA hardware is required')
    torch.cuda.set_device(local_rank)
    if args.blas_backend is not None:
        torch.backends.cuda.preferred_blas_library(args.blas_backend)
    device = torch.device('cuda', local_rank)
    free, total = torch.cuda.mem_get_info(device)
    if args.vram_limit_gib is not None:
        limit = int(args.vram_limit_gib * GIB)
        if limit > total or limit > free:
            raise RuntimeError(f'Allocator limit exceeds available CUDA memory: free={free/GIB:.2f} GiB')
        torch.cuda.set_per_process_memory_fraction(limit / total, device)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if world_size > 1:
        dist.init_process_group('nccl', device_id=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_file = (args.output_dir / f'rank-{rank}.jsonl').open('x')

    def emit(record):
        record['rank'] = rank
        record['timestamp'] = time.time()
        line = json.dumps(record, sort_keys=True)
        record_file.write(line + '\n')
        record_file.flush()
        print(line, flush=True)

    checkpoint_identity = file_identity(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg = WarpStateConfig(**checkpoint['config'])
    cfg.validate()
    if args.backward_layout != 'baseline':
        from ops.backward_layout_execution import validate_backward_layout_config
        validate_backward_layout_config(cfg)
    if args.seq_len % cfg.chunk_size:
        raise ValueError('Sequence length must be divisible by the model chunk size')
    with torch.device('meta'):
        raw_model = WarpStateLM(cfg)
    raw_model.load_state_dict(checkpoint['model'], assign=True, strict=True)
    # CPU -> CUDA conversion allocates separate parameter storage before training.
    raw_model.to(device=device, dtype=torch.float32).requires_grad_(True)
    if any(parameter.device.type != 'cuda' for parameter in raw_model.parameters()):
        raise RuntimeError('Every optimized parameter must be detached from the CPU mapping')
    cfg.gradient_checkpointing = args.activation_checkpointing
    raw_model.loss_chunk_size = args.loss_chunk_size
    for module in raw_model.modules():
        if hasattr(module, 'use_fused'):
            module.use_fused = args.fused_rms_norm
    count = raw_model.num_parameters()
    if not 2_000_000_000 <= count <= 2_020_000_000:
        raise ValueError('This benchmark requires the complete approximately 2B model')
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1.5e-4, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1, fused=True)
    if args.optimizer_state == 'resume':
        state = copy_optimizer_state(checkpoint['optimizer'], device)
        optimizer.load_state_dict(state)
        del state
    start_tokens = checkpoint.get('tokens_seen', 0)
    start_step = checkpoint.get('step', 0)
    del checkpoint
    execution_experiments = {'loss_mode': args.loss_mode, 'bf16_residuals': args.bf16_residuals}
    extra_sources = []
    if matmul_plan is not None:
        extra_sources.extend(name for name, _ in matmul_plan.source_hashes)
    execution_experiments['matmul_execution'] = matmul_plan.metadata if matmul_plan else None
    execution_experiments['backward_layout_execution'] = None
    if args.backward_layout != 'baseline':
        from ops.backward_layout_execution import configure_backward_layout
        layout_metadata = configure_backward_layout(args.backward_layout)
        execution_experiments['backward_layout_execution'] = layout_metadata
        extra_sources.extend(layout_metadata['source_sha256'])
    from ops.gradient_execution import OverlapSettings, make_backward, scale_loss
    overlap_settings = (OverlapSettings(swap_with_collectives=args.overlap_swap_collectives,
                                        use_runtime_estimations=args.overlap_runtime_estimates,
                                        producer_schedule=args.overlap_producer_schedule,
                                        peak_memory_budget=args.overlap_memory_budget)
                        if args.gradient_mode == 'compiled-autograd-overlap' else None)
    backward = make_backward(args.gradient_mode, distributed=world_size > 1, overlap_settings=overlap_settings)
    extra_sources.append('ops/gradient_execution.py')
    if args.activation_mode != 'baseline':
        from ops.activation_execution import apply_activation_execution
        apply_activation_execution(raw_model, args.activation_mode)
        extra_sources.append('ops/activation_execution.py')
    if matmul_plan is not None:
        from ops.matmul_execution import apply_matmul_execution
        apply_matmul_execution(raw_model, matmul_plan)
    execution_experiments.update(gradient_mode=args.gradient_mode,
                                communication_overlap=overlap_settings.metadata if overlap_settings else None,
                                activation_mode=args.activation_mode,
                                cache_core_weight_casts=args.cache_core_weight_casts)
    if cudagraph_diagnostics is not None:
        extra_sources.append('ops/cudagraph_diagnostics.py')
    if args.compile_cores:
        # Compiling bound forwards preserves the original module/state-dict names.
        for core in raw_model.cores:
            core.forward = torch.compile(core.forward, fullgraph=True, dynamic=False, mode=args.compile_mode)
    if args.loss_mode != 'checkpointed':
        from ops.loss_experiment import apply_loss_execution
        apply_loss_execution(raw_model, args.loss_mode)
        extra_sources.append('ops/loss_experiment.py')
    if args.bf16_residuals:
        from ops.residual_experiment import apply_bf16_residuals
        apply_bf16_residuals(raw_model)
        extra_sources.append('ops/residual_experiment.py')
    weight_cast_cache = None
    if args.cache_core_weight_casts:
        from ops.weight_cast_execution import apply_weight_cast_cache
        weight_cast_cache = apply_weight_cast_cache(raw_model)
        extra_sources.append('ops/weight_cast_execution.py')
    model = raw_model
    if world_size > 1:
        from ops.distributed_execution import ddp_kwargs, ddp_runtime_settings
        ddp_settings = ddp_runtime_settings(bucket_cap_mb=args.ddp_bucket_cap_mb,
                                            batched_grad_copy=args.ddp_batched_grad_copy)
        model = DistributedDataParallel(raw_model, device_ids=[local_rank], output_device=local_rank,
                                        find_unused_parameters=False, **ddp_kwargs(ddp_settings))
        extra_sources.append('ops/distributed_execution.py')
        execution_experiments.update(ddp_settings)
    model.train()
    # Graph-owned .grad outputs cannot safely accumulate across replay calls.
    # Allocate persistent buffers outside capture and zero them in place.
    graph_grad_buffers = args.compile_cores and args.compile_mode == 'reduce-overhead'
    if graph_grad_buffers:
        for parameter in raw_model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
    train_data = np.memmap(args.data, dtype='<u2', mode='r')
    if len(train_data) <= batch_tokens:
        raise ValueError('Training data is shorter than a microbatch')
    metadata = {'event': 'configuration', 'args': {key: str(value) if isinstance(value, Path) else value
                                                 for key, value in vars(args).items()},
                'world_size': world_size, 'grad_accum': accumulation, 'parameters': count,
                'gpu': torch.cuda.get_device_name(device), 'torch': torch.__version__, 'cuda': torch.version.cuda,
                'checkpoint': checkpoint_identity, 'checkpoint_step': start_step, 'checkpoint_tokens': start_tokens,
                'data': file_identity(args.data), 'source_sha256': {
                    name: source_hash(PACKAGE / name) for name in
                    ['run_40b.py', 'warpstate/model.py', 'ops/benchmark_training.py', *extra_sources]},
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                'nccl_version': torch.cuda.nccl.version(),
                'backend_environment': {name: os.environ.get(name) for name in (
                    'TORCH_BLAS_PREFER_CUBLASLT', 'CUBLASLT_WORKSPACE_SIZE', 'CUBLAS_WORKSPACE_CONFIG',
                    'TORCHINDUCTOR_COMPILE_THREADS', 'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM',
                    'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS', 'TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE',
                    'NCCL_DEBUG', 'NCCL_DEBUG_SUBSYS', 'NCCL_P2P_DISABLE', 'NCCL_P2P_LEVEL',
                    'NCCL_SHM_DISABLE', 'NCCL_ALGO', 'NCCL_PROTO', 'NCCL_MIN_CTAS', 'NCCL_MAX_CTAS',
                    'NCCL_NTHREADS', 'NCCL_BUFFSIZE')},
                'preferred_blas_library': str(torch.backends.cuda.preferred_blas_library()),
                'allocator': os.environ.get('PYTORCH_ALLOC_CONF'), 'initial_device_free_gib': free/GIB,
                'parameter_dtype': 'float32', 'autocast_dtype': 'bfloat16', 'optimizer': 'AdamW(fused=True)',
                'execution_experiments': execution_experiments,
                'persistent_gradient_buffers': graph_grad_buffers,
                'update_timing_includes_gradient_zeroing': True,
                'checkpoint_writes': False}
    if args.diagnose_stalls:
        metadata['stage_wall_timing'] = ('Host wall seconds, including implicit synchronization waits; '
                                         'no additional CUDA synchronization. These are not GPU kernel timings.')
    emit(metadata)
    updates = []
    for update in range(args.warmup_updates + args.measured_updates):
        if args.backward_layout == 'paired':
            from ops.backward_layout_execution import backward_layout_stats, require_backward_layout_dispatch
            layout_calls_before = backward_layout_stats()['runtime_calls']
        if world_size > 1:
            dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        profile_this_update = args.profile and update == args.warmup_updates
        profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                          record_shapes=True, with_stack=False, profile_memory=False)
                    if profile_this_update else None)
        mark = torch.profiler.record_function if profile_this_update else lambda name: contextlib.nullcontext()
        stage_wall_seconds = {}
        if args.diagnose_stalls:
            mark = timed_stage_markers(stage_wall_seconds, mark)
        with (stalled_update_tracebacks(args.diagnose_stalls and update >= args.warmup_updates),
              profiler if profiler is not None else contextlib.nullcontext()):
            begin = time.perf_counter()
            with mark('warpstate::zero_gradients'):
                optimizer.zero_grad(set_to_none=not graph_grad_buffers)
            loss_sum = torch.zeros((), device=device)
            with weight_cast_cache.update() if weight_cast_cache else contextlib.nullcontext():
                for micro in range(accumulation):
                    cursor = start_tokens + update * args.global_tokens_per_update + (micro * world_size + rank) * batch_tokens
                    with mark('warpstate::batch_and_transfer'):
                        x, y = batches(train_data, cursor, args.micro_batch, args.seq_len, batch_tokens, device)
                    synchronize = micro == accumulation - 1 or world_size == 1
                    sync_context = contextlib.nullcontext() if synchronize else model.no_sync()
                    with sync_context:
                        with mark('warpstate::forward'):
                            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=args.autocast_cache):
                                _, loss = model(x, y, return_logits=False)
                        with mark('warpstate::backward'):
                            scaled = scale_loss(loss, 1.0 / accumulation, args.gradient_mode)
                            backward(scaled)
                    loss_sum += scaled.detach()
            with mark('warpstate::clip_gradients'):
                norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0, error_if_nonfinite=True)
            with mark('warpstate::adamw_step'):
                optimizer.step()
            if world_size > 1:
                with mark('warpstate::loss_allreduce'):
                    dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
            with mark('warpstate::cuda_sync'):
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - begin
            if world_size > 1:
                elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                elapsed = float(elapsed_tensor)
        if args.backward_layout == 'paired':
            require_backward_layout_dispatch(min_calls=layout_calls_before + cfg.logical_depth * accumulation)
            actual_calls = backward_layout_stats()['runtime_calls'] - layout_calls_before
            if update >= args.warmup_updates and actual_calls != cfg.logical_depth * accumulation:
                raise RuntimeError('Measured backward layout dispatch count differs from logical depth times accumulation')
        loss_value, norm_value = float(loss_sum), float(norm)
        if not math.isfinite(loss_value) or not math.isfinite(norm_value):
            raise RuntimeError('Non-finite loss or gradient norm')
        optimizer_bytes = sum(value.numel() * value.element_size()
                              for state in optimizer.state.values() for value in state.values()
                              if isinstance(value, torch.Tensor))
        if optimizer_bytes < count * 8:
            raise RuntimeError('The full FP32 Adam first and second moments were not materialized')
        record = {'event': 'update', 'update': update + 1,
                  'phase': 'warmup' if update < args.warmup_updates else 'measured',
                  'profiled': profile_this_update,
                  'global_tokens': args.global_tokens_per_update, 'seconds': elapsed,
                  'tokens_per_second': args.global_tokens_per_update / elapsed,
                  'loss': loss_value, 'grad_norm': norm_value, 'optimizer_state_gib': optimizer_bytes/GIB,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/GIB,
                  'peak_reserved_gib': torch.cuda.max_memory_reserved(device)/GIB,
                  'device_free_gib': torch.cuda.mem_get_info(device)[0]/GIB}
        if args.diagnose_stalls:
            record['stage_wall_seconds'] = stage_wall_seconds
        if args.backward_layout != 'baseline':
            from ops.backward_layout_execution import backward_layout_stats
            record['backward_layout'] = backward_layout_stats()
            if args.backward_layout == 'paired':
                record['backward_layout_calls'] = record['backward_layout']['runtime_calls'] - layout_calls_before
        if cudagraph_diagnostics is not None:
            record['cudagraph_diagnostics'] = cudagraph_diagnostics.snapshot()
        emit(record)
        updates.append(record)
        if profiler is not None:
            emit({'event': 'profile', 'update': update + 1,
                  **export_profile(profiler, args.output_dir, rank, update + 1, elapsed)})
    measured = [record for record in updates if record['phase'] == 'measured']
    summary = {'event': 'summary', 'configuration': metadata, 'updates': updates,
               'measured_tokens_per_second': sum(row['global_tokens'] for row in measured) / sum(row['seconds'] for row in measured),
               'median_tokens_per_second': statistics.median(row['tokens_per_second'] for row in measured),
               'first_update_peak_allocated_gib': updates[0]['peak_allocated_gib'],
               'first_update_peak_reserved_gib': updates[0]['peak_reserved_gib'],
               'steady_peak_allocated_gib': max(row['peak_allocated_gib'] for row in measured),
               'steady_peak_reserved_gib': max(row['peak_reserved_gib'] for row in measured),
               'checkpoint_path_identity_after': file_identity(args.checkpoint)}
    if world_size > 1:
        rank_summaries = [None] * world_size
        dist.all_gather_object(rank_summaries, summary)
        if rank == 0:
            summary['ranks'] = rank_summaries
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        emit({key: value for key, value in summary.items() if key not in ('configuration', 'updates', 'ranks')})
    record_file.close()


if __name__ == '__main__':
    args = arguments()
    if args.cudagraph_diagnostics:
        from ops.cudagraph_diagnostics import collect_cudagraph_diagnostics
        with collect_cudagraph_diagnostics() as diagnostics:
            benchmark(args, diagnostics)
    else:
        benchmark(args)
