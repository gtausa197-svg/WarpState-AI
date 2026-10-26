"""CPU-only checks of deferred DPO fixtures and acceptance decisions."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ops import dpo_preflight as preflight
from posttrain import atomic_json, digest


def test_fixture_is_source_pinned_cpu_only_and_contains_multiturn_pairs(tmp_path, monkeypatch):
    import sentencepiece as spm
    from ops.posttrain_data import encode_messages

    def no_cuda(*args, **kwargs):
        raise AssertionError('Fixture preparation must not initialize CUDA')

    monkeypatch.setattr(torch.cuda, '_lazy_init', no_cuda)
    fixture = preflight.make_fixture(tmp_path, device='cuda', compile_core=True)
    assert preflight.make_fixture(tmp_path, device='cuda', compile_core=True).weights == fixture.weights
    export = torch.load(fixture.weights, weights_only=False, mmap=True)
    assert export['format'] == 'warpstate-chat-v1' and export['stage'] == 'sft'
    assert export['config']['n_heads'] == 4 and export['config']['d_model'] == 32
    assert export['config']['dropout'] == 0.0
    rows = [json.loads(line) for line in (fixture.data_dir / 'dpo_train.jsonl').read_text().splitlines()]
    assert len(rows) == 5
    assert any(message['role'] == 'assistant' for message in rows[0]['prompt'])
    sp = spm.SentencePieceProcessor(model_file=str(fixture.tokenizer))
    for row in rows:
        for branch in ('chosen', 'rejected'):
            full = encode_messages(row['prompt'] + row[branch], sp, max_length=128)
            response = encode_messages(row[branch], sp, max_length=128)
            assert row['tokens'][branch] == len(full['input_ids'])
            assert row['supervised_tokens'][branch] == sum(label != -100 for label in response['labels'][1:])
    full = encode_messages(rows[0]['prompt'] + rows[0]['chosen'], sp)
    assert sum(label != -100 for label in full['labels'][1:]) > rows[0]['supervised_tokens']['chosen']
    assert any(row['supervised_tokens']['chosen'] != row['supervised_tokens']['rejected'] for row in rows)
    marker = json.loads((tmp_path / 'fixture.json').read_text())
    for name, expected in marker['files'].items():
        assert digest(tmp_path / name) == expected
    with pytest.raises(ValueError, match='settings/source differ'):
        preflight.make_fixture(tmp_path, device='cpu', compile_core=True)
    (fixture.data_dir / 'dpo_val.jsonl').write_text('changed\n')
    with pytest.raises(ValueError, match='input changed'):
        preflight.make_fixture(tmp_path, device='cuda', compile_core=True)


def test_fixture_refuses_unrecognized_contents_and_base_run(tmp_path):
    (tmp_path / 'unrecognized').write_text('preserve')
    with pytest.raises(ValueError, match='nonempty'):
        preflight.make_fixture(tmp_path)
    with pytest.raises(ValueError, match='overwrite the base run'):
        preflight.make_fixture(tmp_path / 'run/preflight', base_work_dir=tmp_path)
    with pytest.raises(ValueError, match='execution mode'):
        preflight.make_fixture(tmp_path / 'fixture', device='cuda', compile_core=1)


def test_distributed_commands_use_fixed_reference_and_independent_output(tmp_path):
    fixture = preflight.make_fixture(tmp_path)
    command = preflight.training_command(fixture, 2, tmp_path / 'split', tmp_path / 'stop', 1)
    assert command[-2:] == ['--max-steps', '1']
    assert command[command.index('--reference-cache-dir') + 1] == str(fixture.reference_cache_dir)
    assert command[command.index('--work-dir') + 1] == str(tmp_path / 'split')
    assert command[command.index('--nproc-per-node') + 1] == '2'
    reference = preflight.training_command(fixture, 2, tmp_path / 'scoring', tmp_path / 'stop', reference_only=True)
    assert reference[-1] == '--reference-only' and '--max-steps' not in reference
    with pytest.raises(ValueError, match='two ranks'):
        preflight.training_command(fixture, 1, tmp_path / 'run', tmp_path / 'stop')
    with pytest.raises(ValueError, match='cannot specify training steps'):
        preflight.training_command(fixture, 2, tmp_path / 'run', tmp_path / 'stop', 1, reference_only=True)


def make_reference(directory, *, dtype=np.float64, complete=True, shape=None, value=-2.0):
    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, count in [('train.npy', 5), ('val.npy', 2)]:
        np.save(directory / name, np.full(shape or (count, 2), value, dtype=dtype))
        files[name] = {'sha256': digest(directory / name), 'rows': count}
    atomic_json(directory / 'manifest.json', dict(format='warpstate-dpo-reference-v1',
        complete=complete, files=files))


def test_reference_artifacts_are_complete_finite_frozen_and_hash_checked(tmp_path):
    make_reference(tmp_path)
    hashes = preflight.reference_artifacts(tmp_path)
    assert set(hashes) == {'manifest.json', 'train.npy', 'val.npy'}
    assert preflight.reference_artifacts(tmp_path) == hashes
    with (tmp_path / 'train.npy').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        preflight.reference_artifacts(tmp_path)


@pytest.mark.parametrize(('kwargs', 'message'), [
    ({'complete': False}, 'incomplete'),
    ({'dtype': np.float32}, 'invalid scores'),
    ({'shape': (5, 3)}, 'shape differs'),
    ({'value': math.nan}, 'invalid scores'),
    ({'value': 0.0}, 'summed log probabilities'),
])
def test_reference_artifacts_reject_invalid_arrays(tmp_path, kwargs, message):
    make_reference(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=message):
        preflight.reference_artifacts(tmp_path)


def sample_checkpoint():
    rng = dict(torch=torch.tensor([1, 2], dtype=torch.uint8), cuda=None,
        python=(3, (1, 2), None), numpy=('MT19937', np.array([1, 2], dtype=np.uint32), 2, 0, 0.0))
    return dict(format='warpstate-dpo-v1', step=2, completed=True, sample_cursor=5,
        tokens_seen=123, response_tokens_seen=45, val_loss=0.65,
        signature={'world_size': 2}, model={'weight': torch.tensor([1.0], dtype=torch.float32)},
        optimizer={'state': {0: {'step': torch.tensor(2.), 'exp_avg': torch.tensor([0.1]),
                                'exp_avg_sq': torch.tensor([0.01])}}, 'param_groups': [{'lr': 0.001}]},
        rng_by_rank=[copy.deepcopy(rng), copy.deepcopy(rng)])


def make_finished_run(directory):
    directory.mkdir(exist_ok=True)
    state = sample_checkpoint()
    torch.save(state, directory / 'latest.pt')
    rows = [dict(event='validation', step=0, val_loss=math.log(2)),
        dict(event='train', step=1, update_samples=4, loss=math.log(2), grad_norm=0.1, lr=0.001),
        dict(event='train', step=2, update_samples=1, loss=0.65, grad_norm=0.1, lr=0.0001)]
    (directory / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    atomic_json(directory / 'completion.json', dict(completed=True, paused_for_evaluation=True))
    atomic_json(directory / 'evaluation.json', dict(step=2, full_holdout=True, evaluated_samples=2,
        val_loss=0.65, reward_accuracy=0.5, reward_margin=0.01, chosen_nll=10.0))
    return rows


def test_finished_run_requires_initial_reference_equality_and_masked_tail(tmp_path):
    rows = make_finished_run(tmp_path)
    assert preflight.verify_fixture_run(tmp_path, expected_step=2, completed=True)['sample_cursor'] == 5
    rows[-1]['update_samples'] = 2
    (tmp_path / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='masked pair tail'):
        preflight.verify_fixture_run(tmp_path, expected_step=2, completed=True)
    rows[-1]['update_samples'] = 1
    rows[0]['val_loss'] = 0.68
    (tmp_path / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match=r'equal log\(2\)'):
        preflight.verify_fixture_run(tmp_path, expected_step=2, completed=True)


def test_resume_compares_counters_rng_exactly_and_model_optimizer_with_tolerance():
    resumed = sample_checkpoint()
    continuous = copy.deepcopy(resumed)
    continuous['model']['weight'] += 1e-6
    comparison = preflight.verify_resume(resumed, continuous)
    assert not comparison['model']['bitwise_equal']
    assert comparison['optimizer']['bitwise_equal']
    continuous['tokens_seen'] += 1
    with pytest.raises(ValueError, match='tokens_seen'):
        preflight.verify_resume(resumed, continuous)
    continuous = copy.deepcopy(resumed)
    continuous['rng_by_rank'][1]['torch'][0] += 1
    with pytest.raises(ValueError, match='rng.1.torch'):
        preflight.verify_resume(resumed, continuous)
    continuous = copy.deepcopy(resumed)
    continuous['optimizer']['state'][0]['exp_avg'] += 0.01
    with pytest.raises(ValueError, match='optimizer'):
        preflight.verify_resume(resumed, continuous)


def test_prepare_only_never_supervises_a_training_process(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('Preparation-only invocation cannot launch subprocesses')

    monkeypatch.setattr(preflight, 'supervise', forbidden)
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr('sys.argv', ['dpo_preflight.py', '--work-dir', str(tmp_path / 'fixture'),
        '--base-work-dir', str(tmp_path), '--prepare-only'])
    assert preflight.main() == 0
    assert json.loads(capsys.readouterr().out)['prepared'] is True
    assert not (tmp_path / 'fixture/reference').exists()
