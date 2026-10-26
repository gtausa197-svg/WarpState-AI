"""CPU-only proof that DPO starts only after immutable full-state SFT archival."""
import copy
import json
from pathlib import Path

import pytest
import torch

from ops import dpo_handoff as handoff


@pytest.fixture
def fixture(tmp_path):
    base = tmp_path / 'base'
    sft = base / 'posttrain/sft'
    run = sft / 'run'
    run.mkdir(parents=True)
    (sft / 'control').mkdir()
    (base / 'control').mkdir()
    tokenizer = tmp_path / 'tokenizer.model'
    tokenizer.write_bytes(b'original tokenizer fixture')
    tokenizer_hash = handoff.sha256(tokenizer)
    sft_plan = dict(format='warpstate-posttrain-plan-v1', stage='sft',
        after_completion='pause_for_evaluation', base=dict(step=76294,
            tokens_seen=10000007168, tokenizer_sha256=tokenizer_hash))
    sft_plan_path = tmp_path / 'sft-plan.json'
    sft_plan_path.write_text(json.dumps(sft_plan))
    data = base / 'posttrain/data'
    data.mkdir()
    outputs = {}
    for name in ('dpo_train.jsonl', 'dpo_val.jsonl'):
        (data / name).write_text('{}\n')
        outputs[name] = dict(rows=1, tokens=10, supervised_tokens=4, sha256=handoff.sha256(data / name))
    manifest = dict(format='warpstate-posttrain-data-v1', complete=True,
                    tokenizer_sha256=tokenizer_hash, max_length=2048, outputs=outputs)
    (data / 'manifest.json').write_text(json.dumps(manifest))
    data_hash = handoff.sha256(data / 'manifest.json')
    expected = dict(step=10312, train_rows=329973, validation_rows=1603,
                    tokens_seen=212321372, assistant_tokens_seen=143695039,
                    base_step=76294, base_tokens_seen=10000007168,
                    raw_archive_sha256='raw-base-hash', tokenizer_sha256=tokenizer_hash)
    work = base / 'posttrain/dpo'
    archive = base / 'archives/sft-final'
    plan = dict(format='warpstate-dpo-plan-v1', stage='dpo', after_completion='pause_for_evaluation',
                sft=expected, handoff=dict(base_work_dir=str(base), sft_work_dir=str(sft),
                    sft_plan=str(sft_plan_path), sft_program='warpstate-posttrain',
                    supervisor_socket=str(base / 'control/supervisor.sock'), archive_dir=str(archive),
                    work_dir=str(work), data_dir=str(data), data_manifest_sha256=data_hash,
                    tokenizer=str(tokenizer), processes=2, poll_seconds=1),
                training=dict(sequence_length=2048))
    config = tmp_path / 'dpo-plan.json'
    config.write_text(json.dumps(plan))
    checkpoint = dict(format='warpstate-posttrain-v1', completed=True, step=10312,
        total_steps=10312, sample_cursor=329973, tokens_seen=212321372,
        assistant_tokens_seen=143695039, val_loss=1.2, val_step=10312, config={'d_model': 2},
        model={'weight': torch.arange(6, dtype=torch.float32).reshape(2, 3)},
        optimizer={'state': {0: {'step': torch.tensor(10312.), 'exp_avg': torch.arange(6.)}}},
        rng_by_rank=[{'torch': torch.tensor([1, 2], dtype=torch.uint8)},
                     {'torch': torch.tensor([3, 4], dtype=torch.uint8)}],
        signature=dict(plan=sft_plan, base_sha256='raw-base-hash', tokenizer_sha256=tokenizer_hash,
                       data_sha256=data_hash),
        origin=dict(step=76294, tokens_seen=10000007168, weights_sha256='raw-base-hash'))
    completion = dict(format='warpstate-posttrain-completion-v1', stage='sft', completed=True,
                      paused_for_evaluation=True, step=10312, total_steps=10312, base_step=76294)
    evaluation = dict(stage='sft', step=10312, full_holdout=True, evaluated_samples=1603,
                      available_validation_samples=1603, val_loss=1.2, base_sha256='raw-base-hash')
    torch.save(checkpoint, run / 'latest.pt')
    for name, obj in [('completion.json', completion), ('evaluation.json', evaluation),
                      ('settings.json', {'signature': checkpoint['signature']})]:
        (run / name).write_text(json.dumps(obj))
    (run / 'metrics.jsonl').write_text(json.dumps(evaluation) + '\n')
    (sft / 'control/handoff.json').write_text(json.dumps({'phase': 'complete'}))
    (sft / 'control/execution.json').write_text(json.dumps({'plan': sft_plan}))
    return dict(base=base, sft=sft, run=run, work=work, archive=archive, tokenizer=tokenizer,
                plan=plan, sft_plan=sft_plan, checkpoint=checkpoint, completion=completion,
                evaluation=evaluation, config=config, data=data)


@pytest.mark.parametrize('field,value', [('completed', False), ('step', 10311),
    ('step', 10313), ('sample_cursor', 329972), ('tokens_seen', 212321371),
    ('assistant_tokens_seen', 143695038), ('val_step', 10300), ('val_loss', float('nan'))])
def test_only_complete_exact_sft_state_is_accepted(fixture, field, value):
    checkpoint = copy.deepcopy(fixture['checkpoint'])
    checkpoint[field] = value
    with pytest.raises(ValueError):
        handoff.validate_sft(checkpoint, fixture['evaluation'], fixture['completion'],
                             fixture['plan'], fixture['sft_plan'])


def test_full_sft_archive_retains_optimizer_rng_exactly_and_model_copy_is_independent(fixture):
    source = fixture['run'] / 'latest.pt'
    original = source.read_bytes()
    metadata, entries = handoff.archive_sft(fixture['sft'], fixture['tokenizer'], fixture['archive'],
                                           fixture['plan'], fixture['sft_plan'], 'test-revision')
    assert (fixture['archive'] / 'checkpoint.pt').read_bytes() == original == source.read_bytes()
    full = torch.load(fixture['archive'] / 'checkpoint.pt', weights_only=False, mmap=True)
    model = torch.load(fixture['archive'] / 'model.pt', weights_only=False, mmap=True)
    assert full['optimizer']['state'][0]['step'].item() == 10312
    assert len(full['rng_by_rank']) == 2 and 'optimizer' not in model
    assert model['stage'] == 'sft' and model['step'] == 10312 and model['origin']['step'] == 76294
    assert torch.equal(model['model']['weight'], full['model']['weight'])
    copied = handoff.initialize_copy(fixture['archive'], fixture['work'] / 'initialization', metadata)
    assert handoff.sha256(copied) == entries['model.pt']
    assert copied.stat().st_ino != (fixture['archive'] / 'model.pt').stat().st_ino
    assert all(path.stat().st_mode & 0o222 == 0 for path in fixture['archive'].iterdir())
    inodes = {path.name: path.stat().st_ino for path in fixture['archive'].iterdir()}
    assert handoff.archive_sft(fixture['sft'], fixture['tokenizer'], fixture['archive'],
                              fixture['plan'], fixture['sft_plan'], 'different-revision') == (metadata, entries)
    assert {path.name: path.stat().st_ino for path in fixture['archive'].iterdir()} == inodes
    assert handoff.initialize_copy(fixture['archive'], copied.parent, metadata) == copied


def test_existing_incomplete_archive_is_preserved(fixture):
    fixture['archive'].mkdir(parents=True)
    note = fixture['archive'] / 'keep.txt'
    note.write_text('user file')
    with pytest.raises(ValueError, match='checksum'):
        handoff.archive_sft(fixture['sft'], fixture['tokenizer'], fixture['archive'],
                            fixture['plan'], fixture['sft_plan'], 'test')
    assert list(fixture['archive'].iterdir()) == [note]
    assert note.read_text() == 'user file'


def test_subset_evaluation_does_not_qualify_for_archival(fixture):
    fixture['evaluation']['full_holdout'] = False
    with pytest.raises(ValueError, match='full-holdout'):
        handoff.validate_sft(fixture['checkpoint'], fixture['evaluation'], fixture['completion'],
                             fixture['plan'], fixture['sft_plan'])


def test_preference_data_tamper_blocks_dpo(fixture):
    assert handoff.data_readiness(fixture['data'], fixture['plan'], verify=True)
    (fixture['data'] / 'dpo_train.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        handoff.data_readiness(fixture['data'], fixture['plan'], verify=True)


def test_cannot_target_sft_or_historical_archives(fixture):
    for path in (fixture['sft'], fixture['run'], fixture['archive']):
        plan = copy.deepcopy(fixture['plan'])
        plan['handoff']['work_dir'] = str(path)
        fixture['config'].write_text(json.dumps(plan))
        with pytest.raises(ValueError):
            handoff.load_plan(fixture['config'])


def test_archive_completed_sft_before_starting_any_gpu_job(fixture, monkeypatch):
    monkeypatch.setattr(handoff, 'STOPPED', False)
    monkeypatch.setattr(handoff, 'source_identity', lambda *a: {'code': 'pinned'})
    monkeypatch.setattr(handoff, 'verify_preflight', lambda *a: {'accepted': True})
    monkeypatch.setattr(handoff, 'verify_dpo_completion', lambda *a: {'completed': True})
    statuses = iter([{'statename': 'RUNNING', 'exitstatus': 0}])
    monkeypatch.setattr(handoff, 'process_status', lambda *a: next(statuses, {'statename': 'EXITED', 'exitstatus': 0}))
    monkeypatch.setattr(handoff.time, 'sleep', lambda *a: None)
    commands = []
    def fake_supervise(command, stop_file, environment, *, pass_fds):
        commands.append(command)
        assert len(pass_fds) == 3
        # Complete full-state archive must exist before even the tiny GPU preflight.
        assert (fixture['archive'] / 'checkpoint.pt').read_bytes() == (fixture['run'] / 'latest.pt').read_bytes()
        for path in (fixture['sft'] / 'control/handoff.lock', fixture['base'] / 'control/training-launcher.lock'):
            with pytest.raises(BlockingIOError):
                with handoff.exclusive_lock(path):
                    pass
        with handoff.exclusive_lock(fixture['base'] / 'pipeline.lock'):
            pass  # Released for all-rank shared locks immediately before launch.
        if 'dpo_preflight.py' in ' '.join(command):
            return 0
        work = Path(command[command.index('--work-dir') + 1])
        weights = Path(command[command.index('--weights') + 1])
        cache = Path(command[command.index('--reference-cache-dir') + 1])
        assert not weights.is_relative_to(work) and not cache.is_relative_to(work)
        assert Path(command[command.index('--stop-file') + 1]) == stop_file
        work.mkdir()
        model_hash = handoff.sha256(weights)
        checkpoint = dict(format='warpstate-dpo-v1', completed=True, step=2, total_steps=2,
                          val_step=2, signature=dict(plan=fixture['plan'], base_sha256=model_hash))
        torch.save(checkpoint, work / 'latest.pt')
        (work / 'evaluation.json').write_text(json.dumps(dict(stage='dpo', step=2, full_holdout=True)))
        (work / 'completion.json').write_text(json.dumps(dict(stage='dpo', step=2,
                                                              completed=True, paused_for_evaluation=True)))
        return 0
    monkeypatch.setattr(handoff, 'supervise', fake_supervise)
    assert handoff.run_handoff(fixture['config']) == 0
    assert len(commands) == 2
    assert json.loads((fixture['work'] / 'control/handoff.json').read_text())['phase'] == 'complete'
    assert handoff.run_handoff(fixture['config']) == 0
    assert len(commands) == 2  # Completed jobs never retrain automatically.


def test_preflight_stop_never_starts_full_dpo(fixture, monkeypatch):
    monkeypatch.setattr(handoff, 'STOPPED', False)
    monkeypatch.setattr(handoff, 'source_identity', lambda *a: {'code': 'pinned'})
    monkeypatch.setattr(handoff, 'process_status', lambda *a: {'statename': 'EXITED', 'exitstatus': 0})
    calls = []
    def stopped(command, stop, env, **kwargs):
        calls.append(command)
        stop.touch()
        return 0
    monkeypatch.setattr(handoff, 'supervise', stopped)
    assert handoff.run_handoff(fixture['config']) == 0
    assert len(calls) == 1 and 'dpo_preflight.py' in ' '.join(calls[0])
    assert json.loads((fixture['work'] / 'control/handoff.json').read_text())['phase'] == 'paused'


@pytest.mark.parametrize('corruption', [None, 'incomplete_epoch', 'subset_validation', 'cache_tamper'])
def test_dpo_completion_checks_epoch_full_evaluation_and_fixed_reference(fixture, corruption):
    import dpo
    import numpy as np
    run = fixture['work'] / 'run'
    cache = fixture['work'] / 'reference-cache'
    run.mkdir(parents=True)
    cache.mkdir()
    settings = dpo.DEFAULTS | fixture['plan']['training']
    model_hash = 'immutable-sft-model'
    data_hash = fixture['plan']['handoff']['data_manifest_sha256']
    identity = dpo.reference_identity(model_hash, data_hash, fixture['plan']['sft']['tokenizer_sha256'],
                                      settings, torch.device('cuda'), 2)
    files = {}
    for name in ('train.npy', 'val.npy'):
        dpo.atomic_array(cache / name, np.array([[-3., -4.]], dtype=np.float64))
        files[name] = dict(rows=1, sha256=handoff.sha256(cache / name))
    handoff.atomic_json(cache / 'manifest.json', dict(format=dpo.CACHE_FORMAT, complete=True,
        identity=identity, identity_sha256=dpo.identity_digest(identity), files=files))
    reference_hash = handoff.sha256(cache / 'manifest.json')
    checkpoint = dict(format='warpstate-dpo-v1', completed=True, step=1, total_steps=1,
        sample_cursor=1, tokens_seen=8, response_tokens_seen=4, val_step=1, val_loss=.69,
        signature=dict(plan=fixture['plan'], settings=settings, base_sha256=model_hash,
                       data_sha256=data_hash, reference_sha256=reference_hash))
    completion = dict(format='warpstate-dpo-completion-v1', stage='dpo', completed=True,
        paused_for_evaluation=True, step=1, total_steps=1, sample_cursor=1,
        tokens_seen=8, response_tokens_seen=4, val_loss=.69, reference_sha256=reference_hash)
    evaluation = dict(stage='dpo', step=1, full_holdout=True, evaluated_samples=1,
        available_validation_samples=1, val_loss=.69, reward_margin=.01, reward_accuracy=1.,
        chosen_nll=1., base_sha256=model_hash, data_sha256=data_hash, reference_sha256=reference_hash)
    if corruption == 'incomplete_epoch':
        checkpoint['sample_cursor'] = 0
    elif corruption == 'subset_validation':
        evaluation['evaluated_samples'] = 0
    elif corruption == 'cache_tamper':
        dpo.atomic_array(cache / 'train.npy', np.array([[-6., -7.]], dtype=np.float64))
    torch.save(checkpoint, run / 'latest.pt')
    handoff.atomic_json(run / 'completion.json', completion)
    handoff.atomic_json(run / 'evaluation.json', evaluation)
    if corruption:
        with pytest.raises(ValueError):
            handoff.verify_dpo_completion(run, fixture['plan'], model_hash)
    else:
        assert handoff.verify_dpo_completion(run, fixture['plan'], model_hash) == completion
