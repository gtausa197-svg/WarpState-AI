#!/usr/bin/env python3
"""Run a tiny, independent two-rank BF16 SFT/resume gate before production SFT.

Fixture creation is CPU-only. The default executable then launches actual CUDA
training, so orchestration must wait until pretraining has exited. All generated
weights/data belong to this disposable preflight directory, never to an archive.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

import torch

from posttrain import atomic_json, digest, shared_pipeline_lock, source_identity
from ops.posttrain_data import encode_messages, renderer_sha256
from ops.launch_training import supervise


def make_fixture(directory, *, device='cpu', compile_core=False, base_work_dir=None):
    """Create/reuse a source-hashed small real model with a masked distributed tail."""
    import sentencepiece as spm
    from warpstate import WarpStateConfig, WarpStateLM
    directory = Path(directory).resolve()
    if not directory.is_relative_to(WORKSPACE) or directory == WORKSPACE:
        raise ValueError('Preflight fixtures must stay inside the workspace')
    if device not in ('cpu', 'cuda') or type(compile_core) is not bool:
        raise ValueError('Invalid fixture execution mode')
    tokenizer = PACKAGE / 'tokenizer.model'
    base = Path(base_work_dir).resolve() if base_work_dir is not None else None
    if base is not None and (not base.is_relative_to(WORKSPACE) or not base.is_dir()):
        raise ValueError('Invalid base work directory')
    if base is not None and (directory == base or directory.is_relative_to(base / 'run')):
        raise ValueError('Preflight must not overwrite the base run')
    args = argparse.Namespace(config=directory / 'plan.json', data_dir=directory / 'data',
                              weights=directory / 'base/model.pt', tokenizer=tokenizer,
                              work_dir=directory / 'run', max_steps=None, stop_file=None, device=device)
    identity = dict(device=device, compile_core=compile_core, base_work_dir=str(base) if base else None,
                    tokenizer_sha256=digest(tokenizer), renderer_sha256=renderer_sha256(),
                    generator_sha256=digest(__file__), runner_sources=source_identity())
    marker = directory / 'fixture.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if old['identity'] != identity:
            raise ValueError('Existing preflight fixture settings/source differ; use a new fixture directory')
        for filename, checksum in old['files'].items():
            if digest(directory / filename) != checksum:
                raise ValueError(f'Existing preflight input changed: {filename}')
        return args
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Refusing to create fixtures in a nonempty unrecognized directory')
    directory.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir()
    args.data_dir.mkdir()
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    cfg = WarpStateConfig(vocab_size=sp.vocab_size(), d_model=32, n_heads=4,
                          n_cores=1, logical_depth=1, ffn_hidden=64,
                          chunk_size=16, parallel_memory=True,
                          gradient_checkpointing=False, dropout=0.1 if device == 'cpu' else 0.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = WarpStateLM(cfg)
    torch.save(dict(format='warpstate-chat-v1', inference_only=True, synthetic_fixture=True,
                    config=vars(cfg), model=model.state_dict(), step=76294, tokens_seen=10_000_007_168,
                    tokenizer_sha256=identity['tokenizer_sha256']), args.weights)
    sequence_length = 128
    outputs = {}
    for filename, count in [('sft_train.jsonl', 5), ('sft_val.jsonl', 2)]:
        rows = []
        for index in range(count):
            messages = [{'role': 'system', 'content': 'Be concise.'},
                        {'role': 'user', 'content': 'Two plus two?'},
                        {'role': 'assistant', 'content': 'four' if index % 2 else 'The answer is four.'}]
            encoded = encode_messages(messages, sp, max_length=sequence_length)
            rows.append(dict(messages=messages, tokens=len(encoded['input_ids']),
                             supervised_tokens=sum(v != -100 for v in encoded['labels'][1:])))
        path = args.data_dir / filename
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        outputs[filename] = dict(sha256=digest(path), rows=count,
                                 tokens=sum(row['tokens'] for row in rows),
                                 supervised_tokens=sum(row['supervised_tokens'] for row in rows))
    atomic_json(args.data_dir / 'manifest.json', dict(format='warpstate-posttrain-data-v1',
        complete=True, outputs=outputs, tokenizer_sha256=identity['tokenizer_sha256'],
        renderer_sha256=identity['renderer_sha256'], max_length=sequence_length))
    plan = dict(format='warpstate-posttrain-plan-v1', stage='sft', base={'step': 76294},
        training=dict(sequence_length=sequence_length, micro_batch=1, grad_accum=2, lr=1e-3,
                      min_lr=1e-4, warmup_ratio=0., eval_every=2, save_every=2, eval_batches=2,
                      compile_core=compile_core, loss_mode='compiled-full' if compile_core else 'full', workers=1))
    if base is not None:
        plan['handoff'] = {'base_work_dir': str(base.relative_to(WORKSPACE))}
    atomic_json(args.config, plan)
    files = [args.weights, args.config, args.data_dir / 'manifest.json',
             args.data_dir / 'sft_train.jsonl', args.data_dir / 'sft_val.jsonl']
    atomic_json(marker, dict(format='warpstate-posttrain-fixture-v1', identity=identity,
                            files={str(path.relative_to(directory)): digest(path) for path in files}))
    return args


def training_command(args, processes, work_dir, stop_file, max_steps=None):
    if type(processes) is not int or processes != 2:
        raise ValueError('The preflight requires exactly two ranks to exercise its masked tail')
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
               '--nnodes', '1', '--nproc-per-node', str(processes), str(PACKAGE / 'posttrain.py'),
               '--config', str(args.config), '--data-dir', str(args.data_dir),
               '--weights', str(args.weights), '--tokenizer', str(args.tokenizer),
               '--work-dir', str(work_dir), '--stop-file', str(stop_file), '--device', args.device]
    if max_steps is not None:
        command += ['--max-steps', str(max_steps)]
    return command


def inherited_fds():
    """Preserve the explicit inheritable exclusion locks passed by the guardian."""
    descriptors = []
    for name in os.listdir('/proc/self/fd'):
        fd = int(name)
        try:
            if fd > 2 and os.get_inheritable(fd):
                descriptors.append(fd)
        except OSError:
            pass
    return tuple(descriptors)


def compare_tree(a, b, path='state', *, tolerant=False, report=None):
    if report is None:
        report = {'bitwise_equal': True, 'max_absolute_difference': 0., 'tensor_count': 0}
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f'Stop/resume tensor metadata mismatch at {path}')
        report['tensor_count'] += 1
        equal = torch.equal(a, b)
        report['bitwise_equal'] &= equal
        if a.numel():
            report['max_absolute_difference'] = max(report['max_absolute_difference'], float((a.double() - b.double()).abs().max()))
        if not equal and (not tolerant or not torch.allclose(a, b, rtol=1e-5, atol=1e-7)):
            raise ValueError(f'Stop/resume mismatch exceeds allowed tolerance at {path}')
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError(f'State keys differ at {path}')
        for key in a:
            compare_tree(a[key], b[key], f'{path}.{key}', tolerant=tolerant, report=report)
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise ValueError(f'State lengths differ at {path}')
        for index, (x, y) in enumerate(zip(a, b)):
            compare_tree(x, y, f'{path}.{index}', tolerant=tolerant, report=report)
    elif a != b:
        raise ValueError(f'State differs at {path}')
    return report


def verify_fixture_run(work, *, expected_step, completed):
    checkpoint = torch.load(work / 'latest.pt', map_location='cpu', weights_only=False, mmap=True)
    if checkpoint.get('format') != 'warpstate-posttrain-v1' or checkpoint['step'] != expected_step:
        raise ValueError('Preflight checkpoint format/step differs')
    if checkpoint['completed'] is not completed or checkpoint['signature']['world_size'] != 2:
        raise ValueError('Preflight completion/distributed state differs')
    if len(checkpoint['rng_by_rank']) != 2:
        raise ValueError('Preflight lacks both rank RNG states')
    for name, value in checkpoint['model'].items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'Invalid FP32 model weight: {name}')
    for value in checkpoint['optimizer']['state'].values():
        if float(value['step']) != expected_step:
            raise ValueError('Optimizer step was not restored')
        for name in ('exp_avg', 'exp_avg_sq'):
            if value[name].dtype != torch.float32 or not torch.isfinite(value[name]).all():
                raise ValueError('Invalid FP32 Adam state')
    metrics = [json.loads(line) for line in (work / 'metrics.jsonl').read_text().splitlines()]
    updates = [row for row in metrics if row['event'] == 'train']
    for row in updates:
        if not all(math.isfinite(row[key]) for key in ('loss', 'grad_norm', 'lr')):
            raise ValueError('Non-finite preflight training metric')
    if completed:
        completion = json.loads((work / 'completion.json').read_text())
        evaluation = json.loads((work / 'evaluation.json').read_text())
        if not completion['completed'] or not completion['paused_for_evaluation']:
            raise ValueError('Preflight did not pause for evaluation')
        if checkpoint['sample_cursor'] != 5 or updates[-1]['update_samples'] != 1:
            raise ValueError('Preflight did not exercise a nondivisible masked tail')
        if evaluation['step'] != expected_step or not math.isfinite(evaluation['val_loss']):
            raise ValueError('Preflight final evaluation is missing')
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--base-work-dir', type=Path, required=True)
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true', help='CPU fixture creation only; never launches training')
    args = parser.parse_args()
    if args.processes != 2:
        parser.error('--processes must be 2 for the masked-tail fixture')
    work = args.work_dir.resolve()
    base = args.base_work_dir.resolve()
    if not base.is_relative_to(WORKSPACE) or not base.is_dir():
        parser.error('--base-work-dir must be an existing workspace directory')
    stop_file = Path(os.environ.get('WARPSTATE_STOP_FILE', work / 'control/training-stop-request')).resolve()
    if not stop_file.is_relative_to(WORKSPACE):
        parser.error('Stop file must stay inside the workspace')
    # Keep the base excluded throughout the gaps between the three torchrun jobs.
    with shared_pipeline_lock(base / 'pipeline.lock'):
        fixture = make_fixture(work, device='cuda', compile_core=True, base_work_dir=base)
        if args.prepare_only:
            print(json.dumps({'prepared': True, 'config': str(fixture.config)}), flush=True)
            return 0
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        raw_hash = digest(fixture.weights)
        if stop_file.exists():
            return 0
        report_path = work / 'report.json'
        report_path.unlink(missing_ok=True)
        phases = [('split', 1), ('split', None), ('continuous', None)]
        for phase, limit in phases:
            if stop_file.exists():
                return 0
            target = work / phase
            # A previous successful phase is not rerun, preserving exact metrics.
            if (target / 'completion.json').exists():
                continue
            if limit == 1 and (target / 'latest.pt').exists():
                continue
            command = training_command(fixture, args.processes, target, stop_file, limit)
            result = supervise(command, stop_file, os.environ.copy(), pass_fds=inherited_fds())
            if stop_file.exists():
                return 0
            if result:
                raise RuntimeError(f'Preflight {phase} failed with status {result}')
            verify_fixture_run(target, expected_step=1 if limit else 2, completed=limit is None)
        resumed = verify_fixture_run(work / 'split', expected_step=2, completed=True)
        continuous = verify_fixture_run(work / 'continuous', expected_step=2, completed=True)
        comparison = {key: compare_tree(resumed[key], continuous[key], key, tolerant=True)
                      for key in ('model', 'optimizer')}
        for key in ('tokens_seen', 'assistant_tokens_seen', 'sample_cursor', 'step'):
            compare_tree(resumed[key], continuous[key], key)
        for rank in range(2):
            for key in ('torch', 'cuda', 'python'):
                compare_tree(resumed['rng_by_rank'][rank][key], continuous['rng_by_rank'][rank][key], f'rng.{rank}.{key}')
            a, b = resumed['rng_by_rank'][rank]['numpy'], continuous['rng_by_rank'][rank]['numpy']
            if a[0] != b[0] or not (a[1] == b[1]).all() or a[2:] != b[2:]:
                raise ValueError('NumPy RNG state differs after resume')
        if not math.isclose(resumed['val_loss'], continuous['val_loss'], rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError('Validation loss differs after resume')
        if digest(fixture.weights) != raw_hash:
            raise ValueError('Raw preflight fixture weights changed')
        if not any(not torch.equal(value, resumed['model'][name]) for name, value in
                   torch.load(fixture.weights, map_location='cpu', weights_only=False, mmap=True)['model'].items()):
            raise ValueError('Preflight did not update model parameters')
        atomic_json(report_path, dict(format='warpstate-posttrain-preflight-v1', accepted=True,
            processes=2, bf16_compute=True, fp32_parameters_and_adam=True,
            exact_resume_bookkeeping=True, numerical_rtol=1e-5, numerical_atol=1e-7,
            numerical_comparison=comparison, masked_tail=True, total_samples=5, optimizer_steps=2,
            fixture_sha256=raw_hash, source_sha256=resumed['signature']['sources'],
            generator_sha256=digest(__file__), final_validation_loss=resumed['val_loss']))
        print(f'Posttraining preflight accepted: {report_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
