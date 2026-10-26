#!/usr/bin/env python3
"""Deferred BF16 DPO gate for numerical continuation and exact bookkeeping.

Fixture creation is CPU-only. The executable must run after SFT has stopped:
it scores a frozen reference, then checks interrupted and continuous DPO on a
five-pair fixture with a masked final distributed update. No archive is edited.
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

from posttrain import atomic_json, digest, shared_pipeline_lock
from ops.launch_training import supervise
from ops.posttrain_data import encode_messages, renderer_sha256
from ops.posttrain_preflight import compare_tree, inherited_fds


def make_fixture(directory, *, device='cpu', compile_core=False, base_work_dir=None):
    """Create or verify a source-pinned small real model and preference data."""
    import sentencepiece as spm
    import dpo
    from warpstate import WarpStateConfig, WarpStateLM

    directory = Path(directory).resolve()
    if not directory.is_relative_to(WORKSPACE) or directory == WORKSPACE:
        raise ValueError('Preflight fixtures must stay inside the workspace')
    if device not in ('cpu', 'cuda') or type(compile_core) is not bool:
        raise ValueError('Invalid fixture execution mode')
    base = Path(base_work_dir).resolve() if base_work_dir is not None else None
    if base is not None and (not base.is_relative_to(WORKSPACE) or not base.is_dir()):
        raise ValueError('Invalid base work directory')
    if base is not None and (directory == base or directory.is_relative_to(base / 'run')):
        raise ValueError('Preflight must not overwrite the base run')
    tokenizer = PACKAGE / 'tokenizer.model'
    args = argparse.Namespace(config=directory / 'plan.json', data_dir=directory / 'data',
        weights=directory / 'base/model.pt', tokenizer=tokenizer, work_dir=directory / 'run',
        reference_cache_dir=directory / 'reference', reference_only=False,
        max_steps=None, stop_file=None, device=device)
    identity = dict(device=device, compile_core=compile_core,
        base_work_dir=str(base) if base else None, tokenizer_sha256=digest(tokenizer),
        renderer_sha256=renderer_sha256(), generator_sha256=digest(__file__),
        runner_sources=dpo.source_identity())
    marker = directory / 'fixture.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get('format') != 'warpstate-dpo-fixture-v1' or old.get('identity') != identity:
            raise ValueError('Existing preflight fixture settings/source differ; use a new fixture directory')
        for filename, checksum in old['files'].items():
            path = (directory / filename).resolve()
            if not path.is_relative_to(directory) or digest(path) != checksum:
                raise ValueError(f'Existing preflight input changed: {filename}')
        return args
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Refusing to create fixtures in a nonempty unrecognized directory')
    directory.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir()
    args.data_dir.mkdir()
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    cfg = WarpStateConfig(vocab_size=sp.vocab_size(), d_model=32, n_heads=4,
        n_cores=1, logical_depth=1, ffn_hidden=64, chunk_size=16,
        parallel_memory=True, gradient_checkpointing=False, dropout=0.0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = WarpStateLM(cfg)
    torch.save(dict(format='warpstate-chat-v1', inference_only=True, synthetic_fixture=True,
        stage='sft', config=vars(cfg), model=model.state_dict(), step=10312,
        tokens_seen=212651345, tokenizer_sha256=identity['tokenizer_sha256'],
        origin=dict(base_step=76294, base_tokens_seen=10_000_007_168)), args.weights)
    sequence_length = 128
    outputs = {}
    for filename, count in [('dpo_train.jsonl', 5), ('dpo_val.jsonl', 2)]:
        rows = []
        for index in range(count):
            prompt = [{'role': 'system', 'content': 'Be concise.'}]
            if index == 0:
                # Earlier assistant turns are conditioning, never DPO targets.
                prompt += [{'role': 'user', 'content': 'Say hello.'},
                           {'role': 'assistant', 'content': 'Hello.'}]
            prompt += [{'role': 'user', 'content': f'Question {index}: two plus two?'}]
            row = dict(prompt=prompt,
                chosen=[{'role': 'assistant', 'content': 'four' if index % 2 else 'The answer is four.'}],
                rejected=[{'role': 'assistant', 'content': 'five, or perhaps six.' if index % 2 else 'Three.'}],
                tokens={}, supervised_tokens={})
            for branch in ('chosen', 'rejected'):
                encoded = encode_messages(prompt + row[branch], sp, max_length=sequence_length)
                response = encode_messages(row[branch], sp, max_length=sequence_length)
                row['tokens'][branch] = len(encoded['input_ids'])
                row['supervised_tokens'][branch] = sum(v != -100 for v in response['labels'][1:])
            rows.append(row)
        path = args.data_dir / filename
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        outputs[filename] = dict(sha256=digest(path), rows=count,
            tokens=sum(sum(row['tokens'].values()) for row in rows),
            supervised_tokens=sum(sum(row['supervised_tokens'].values()) for row in rows))
    atomic_json(args.data_dir / 'manifest.json', dict(format='warpstate-posttrain-data-v1',
        complete=True, outputs=outputs, tokenizer_sha256=identity['tokenizer_sha256'],
        renderer_sha256=identity['renderer_sha256'], max_length=sequence_length))
    plan = dict(format='warpstate-dpo-plan-v1', stage='dpo', base={'step': 10312},
        after_completion='pause_for_evaluation',
        training=dict(sequence_length=sequence_length, micro_batch=1, grad_accum=2,
            lr=1e-3, min_lr=1e-4, warmup_ratio=0.0, epochs=1, eval_every=2,
            save_every=2, eval_batches=2, compile_core=compile_core, workers=1,
            beta=5.0, objective='length-normalized-dpo', head_chunk_size=32))
    if base is not None:
        plan['handoff'] = {'base_work_dir': str(base.relative_to(WORKSPACE))}
    atomic_json(args.config, plan)
    files = [args.weights, args.config, args.data_dir / 'manifest.json',
             args.data_dir / 'dpo_train.jsonl', args.data_dir / 'dpo_val.jsonl']
    atomic_json(marker, dict(format='warpstate-dpo-fixture-v1', identity=identity,
        files={str(path.relative_to(directory)): digest(path) for path in files}))
    return args


def training_command(args, processes, work_dir, stop_file, max_steps=None, *, reference_only=False):
    if type(processes) is not int or processes != 2:
        raise ValueError('The preflight requires exactly two ranks to exercise its masked tail')
    if reference_only and max_steps is not None:
        raise ValueError('Reference-only scoring cannot specify training steps')
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone',
        '--nnodes', '1', '--nproc-per-node', str(processes), str(PACKAGE / 'dpo.py'),
        '--config', str(args.config), '--data-dir', str(args.data_dir),
        '--weights', str(args.weights), '--tokenizer', str(args.tokenizer),
        '--reference-cache-dir', str(args.reference_cache_dir), '--work-dir', str(work_dir),
        '--stop-file', str(stop_file), '--device', args.device]
    if reference_only:
        command.append('--reference-only')
    if max_steps is not None:
        command += ['--max-steps', str(max_steps)]
    return command


def reference_artifacts(directory):
    """Validate all sealed reference arrays and return immutable content hashes."""
    import numpy as np

    directory = Path(directory).resolve()
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 'warpstate-dpo-reference-v1' or manifest.get('complete') is not True:
        raise ValueError('Frozen reference cache is incomplete')
    files = manifest.get('files', {})
    if set(files) != {'train.npy', 'val.npy'}:
        raise ValueError('Frozen reference cache lacks the complete train/validation arrays')
    result = {'manifest.json': digest(manifest_path)}
    for name, info in files.items():
        path = directory / name
        if not path.is_file() or path.is_symlink() or digest(path) != info.get('sha256'):
            raise ValueError(f'Frozen reference cache hash differs: {name}')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        expected_rows = 5 if name == 'train.npy' else 2
        if array.shape != (expected_rows, 2) or info.get('rows') != expected_rows:
            raise ValueError(f'Frozen reference cache shape differs: {name}')
        if array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError(f'Frozen reference cache contains invalid scores: {name}')
        if not (array < 0).all():
            raise ValueError(f'Frozen reference must contain summed log probabilities: {name}')
        result[name] = info['sha256']
    return result


def verify_fixture_run(work, *, expected_step, completed):
    work = Path(work)
    checkpoint = torch.load(work / 'latest.pt', map_location='cpu', weights_only=False, mmap=True)
    if checkpoint.get('format') != 'warpstate-dpo-v1' or checkpoint.get('step') != expected_step:
        raise ValueError('Preflight checkpoint format/step differs')
    if checkpoint.get('completed') is not completed or checkpoint['signature'].get('world_size') != 2:
        raise ValueError('Preflight completion/distributed state differs')
    if len(checkpoint['rng_by_rank']) != 2:
        raise ValueError('Preflight lacks both rank RNG states')
    for name, value in checkpoint['model'].items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f'Invalid FP32 model weight: {name}')
    optimizer = checkpoint['optimizer']['state']
    if not optimizer:
        raise ValueError('Preflight lacks Adam optimizer state')
    for value in optimizer.values():
        if float(value['step']) != expected_step:
            raise ValueError('Optimizer step was not restored')
        for name in ('exp_avg', 'exp_avg_sq'):
            if value[name].dtype != torch.float32 or not torch.isfinite(value[name]).all():
                raise ValueError('Invalid FP32 Adam state')
    metrics = [json.loads(line) for line in (work / 'metrics.jsonl').read_text().splitlines()]
    updates = [row for row in metrics if row.get('event') == 'train']
    if [row['step'] for row in updates] != list(range(1, expected_step + 1)):
        raise ValueError('Preflight skipped or repeated an optimizer update')
    for row in updates:
        if not all(math.isfinite(row[key]) for key in ('loss', 'grad_norm', 'lr')):
            raise ValueError('Non-finite preflight training metric')
    initial = [row for row in metrics if row.get('event') == 'validation' and row.get('step') == 0]
    if len(initial) != 1 or not math.isclose(initial[0]['val_loss'], math.log(2), rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError('Initial policy/reference DPO loss must equal log(2)')
    if completed:
        completion = json.loads((work / 'completion.json').read_text())
        evaluation = json.loads((work / 'evaluation.json').read_text())
        if not completion.get('completed') or not completion.get('paused_for_evaluation'):
            raise ValueError('Preflight did not pause for evaluation')
        if checkpoint['sample_cursor'] != 5 or updates[-1]['update_samples'] != 1:
            raise ValueError('Preflight did not exercise a nondivisible masked pair tail')
        if sum(row['update_samples'] for row in updates) != 5:
            raise ValueError('Preflight did not train every preference pair exactly once')
        if evaluation.get('step') != expected_step or evaluation.get('full_holdout') is not True:
            raise ValueError('Preflight final full evaluation is missing')
        if evaluation.get('evaluated_samples') != 2:
            raise ValueError('Preflight did not evaluate all held-out preference pairs')
        for key in ('val_loss', 'reward_accuracy', 'reward_margin', 'chosen_nll'):
            if not math.isfinite(evaluation[key]):
                raise ValueError(f'Invalid preflight evaluation metric: {key}')
        if not 0 <= evaluation['reward_accuracy'] <= 1:
            raise ValueError('Invalid preference reward accuracy')
    return checkpoint


def verify_resume(resumed, continuous):
    """Require close FP32 training state and exact counters and per-rank RNG."""
    comparison = {key: compare_tree(resumed[key], continuous[key], key, tolerant=True)
                  for key in ('model', 'optimizer')}
    for key in ('tokens_seen', 'response_tokens_seen', 'sample_cursor', 'step'):
        compare_tree(resumed[key], continuous[key], key)
    for rank in range(2):
        for key in ('torch', 'cuda', 'python'):
            compare_tree(resumed['rng_by_rank'][rank][key], continuous['rng_by_rank'][rank][key], f'rng.{rank}.{key}')
        a, b = resumed['rng_by_rank'][rank]['numpy'], continuous['rng_by_rank'][rank]['numpy']
        if a[0] != b[0] or not (a[1] == b[1]).all() or a[2:] != b[2:]:
            raise ValueError('NumPy RNG state differs after resume')
    if not math.isclose(resumed['val_loss'], continuous['val_loss'], rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError('Validation loss differs after resume')
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--base-work-dir', type=Path, required=True)
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true', help='Create CPU fixtures without launching GPU work')
    args = parser.parse_args()
    if args.processes != 2:
        parser.error('--processes must be 2 for the masked pair-tail fixture')
    work = args.work_dir.resolve()
    base = args.base_work_dir.resolve()
    if not base.is_relative_to(WORKSPACE) or not base.is_dir():
        parser.error('--base-work-dir must be an existing workspace directory')
    stop_file = Path(os.environ.get('WARPSTATE_STOP_FILE', work / 'control/training-stop-request')).resolve()
    if not stop_file.is_relative_to(WORKSPACE):
        parser.error('Stop file must stay inside the workspace')
    # The guardian releases its exclusive pipeline lock before spawning us.
    # This shared lock and inherited guardian FDs exclude base/SFT launches
    # throughout reference scoring and the gaps between torchrun processes.
    with shared_pipeline_lock(base / 'pipeline.lock'):
        fixture = make_fixture(work, device='cuda', compile_core=True, base_work_dir=base)
        if args.prepare_only:
            print(json.dumps({'prepared': True, 'config': str(fixture.config)}), flush=True)
            return 0
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        if stop_file.exists():
            return 0
        report_path = work / 'report.json'
        report_path.unlink(missing_ok=True)
        raw_hash = digest(fixture.weights)
        if not (fixture.reference_cache_dir / 'manifest.json').exists():
            result = supervise(training_command(fixture, args.processes, work / 'reference-scoring',
                stop_file, reference_only=True), stop_file, os.environ.copy(), pass_fds=inherited_fds())
            if stop_file.exists():
                return 0
            if result:
                raise RuntimeError(f'Preflight reference scoring failed with status {result}')
        reference_hashes = reference_artifacts(fixture.reference_cache_dir)
        phases = [('split', 1), ('split', None), ('continuous', None)]
        for phase, limit in phases:
            if stop_file.exists():
                return 0
            target = work / phase
            if (target / 'completion.json').exists():
                continue
            if limit == 1 and (target / 'latest.pt').exists():
                continue
            result = supervise(training_command(fixture, args.processes, target, stop_file, limit),
                stop_file, os.environ.copy(), pass_fds=inherited_fds())
            if stop_file.exists():
                return 0
            if result:
                raise RuntimeError(f'DPO preflight {phase} failed with status {result}')
            verify_fixture_run(target, expected_step=1 if limit else 2, completed=limit is None)
        resumed = verify_fixture_run(work / 'split', expected_step=2, completed=True)
        continuous = verify_fixture_run(work / 'continuous', expected_step=2, completed=True)
        comparison = verify_resume(resumed, continuous)
        if digest(fixture.weights) != raw_hash:
            raise ValueError('Raw SFT fixture weights changed')
        if reference_artifacts(fixture.reference_cache_dir) != reference_hashes:
            raise ValueError('Frozen reference cache changed during DPO')
        original = torch.load(fixture.weights, map_location='cpu', weights_only=False, mmap=True)
        if not any(not torch.equal(value, resumed['model'][name]) for name, value in original['model'].items()):
            raise ValueError('DPO preflight did not update policy parameters')
        evaluation = json.loads((work / 'split/evaluation.json').read_text())
        atomic_json(report_path, dict(format='warpstate-dpo-preflight-v1', accepted=True,
            processes=2, bf16_compute=True, fp32_parameters_and_adam=True,
            exact_resume_bookkeeping=True, numerical_rtol=1e-5, numerical_atol=1e-7,
            numerical_comparison=comparison, masked_pair_tail=True, total_pairs=5,
            optimizer_steps=2, fixture_sha256=raw_hash,
            source_sha256=resumed['signature']['sources'], generator_sha256=digest(__file__),
            config_sha256=digest(fixture.config), reference_artifacts=reference_hashes,
            reference_manifest_sha256=reference_hashes['manifest.json'],
            frozen_reference_unchanged=True, final_validation={key: evaluation[key]
                for key in ('val_loss', 'reward_accuracy', 'reward_margin', 'chosen_nll')}))
        print(f'DPO preflight accepted: {report_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
