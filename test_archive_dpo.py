"""CPU-only checks for final DPO archive integrity and immutable publication."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import dpo
from ops import archive_dpo as archive


@pytest.fixture
def fixture(tmp_path):
    base = tmp_path / 'base'
    work = base / 'posttrain/dpo'
    run = work / 'run'
    reference_dir = work / 'reference-cache'
    data_dir = base / 'posttrain/data'
    for path in (run, reference_dir, data_dir, work / 'control', work / 'preflight'):
        path.mkdir(parents=True)
    tokenizer = tmp_path / 'tokenizer.model'
    tokenizer.write_bytes(b'original tokenizer fixture')
    tokenizer_hash = archive.sha256(tokenizer)
    sources = dpo.source_identity()
    renderer_hash = sources['ops/posttrain_data.py']
    data = dict(format='warpstate-posttrain-data-v1', complete=True, max_length=2048,
        tokenizer_sha256=tokenizer_hash, renderer_sha256=renderer_hash,
        outputs={'dpo_train.jsonl': dict(rows=2, tokens=20, supervised_tokens=8),
                 'dpo_val.jsonl': dict(rows=1, tokens=10, supervised_tokens=4)})
    archive.atomic_json(data_dir / 'manifest.json', data)
    data_hash = archive.sha256(data_dir / 'manifest.json')
    plan = dict(format='warpstate-dpo-plan-v1', stage='dpo', after_completion='pause_for_evaluation',
        training={}, sft=dict(step=10312, tokens_seen=212321372, base_step=76294,
            base_tokens_seen=10000007168, tokenizer_sha256=tokenizer_hash),
        handoff=dict(processes=2, data_dir=str(data_dir), data_manifest_sha256=data_hash))
    reference_identity = dpo.reference_identity('sft-model-hash', data_hash, tokenizer_hash,
        dpo.DEFAULTS, torch.device('cuda'), 2)
    arrays = {}
    for split, count in (('train', 2), ('val', 1)):
        path = reference_dir / f'{split}.npy'
        np.save(path, np.asarray([[-2., -3.]] * count, dtype=np.float64), allow_pickle=False)
        arrays[path.name] = dict(sha256=archive.sha256(path), rows=count)
    reference = dict(format='warpstate-dpo-reference-v1', complete=True, identity=reference_identity,
        identity_sha256=dpo.identity_digest(reference_identity), files=arrays)
    archive.atomic_json(reference_dir / 'manifest.json', reference)
    reference_hash = archive.sha256(reference_dir / 'manifest.json')
    signature = dict(plan=plan, settings=dpo.DEFAULTS, base_sha256='sft-model-hash',
        data_sha256=data_hash, tokenizer_sha256=tokenizer_hash, reference_sha256=reference_hash,
        world_size=2, sources=sources, device_type='cuda', torch_version=torch.__version__,
        mask='warpstate-final-assistant-response-v1')
    evaluation = dict(stage='dpo', step=1, full_holdout=True, evaluated_samples=1,
        available_validation_samples=1, val_loss=.5, reward_margin=.4, reward_accuracy=1., chosen_nll=2.,
        objective='length-normalized-dpo', base_sha256='sft-model-hash', data_sha256=data_hash,
        tokenizer_sha256=tokenizer_hash, reference_sha256=reference_hash)
    checkpoint = dict(format='warpstate-dpo-v1', completed=True, step=1, total_steps=1,
        sample_cursor=2, tokens_seen=16, response_tokens_seen=8, val_step=1, val_loss=.5,
        config={'d_model': 2}, model={'weight': torch.arange(6, dtype=torch.float32).reshape(2, 3)},
        optimizer={'state': {0: dict(step=torch.tensor(1.), exp_avg=torch.arange(6.), exp_avg_sq=torch.arange(6.))}},
        rng_by_rank=[dict(torch=torch.tensor([1, 2], dtype=torch.uint8), cuda=torch.tensor([3], dtype=torch.uint8))] * 2,
        signature=signature, validation={key: evaluation[key] for key in
            ('val_loss', 'reward_margin', 'reward_accuracy', 'chosen_nll', 'evaluated_samples')},
        origin=dict(stage='sft', step=10312, tokens_seen=212321372, weights_sha256='sft-model-hash',
                    origin=dict(step=76294, tokens_seen=10000007168)))
    completion = dict(format='warpstate-dpo-completion-v1', stage='dpo', completed=True,
        paused_for_evaluation=True, reference_sha256=reference_hash,
        **{key: checkpoint[key] for key in ('step', 'total_steps', 'sample_cursor', 'tokens_seen', 'response_tokens_seen', 'val_loss')})
    settings_report = dict(signature=signature, total_steps=1, train_rows=2, val_rows=1)
    torch.save(checkpoint, run / 'latest.pt')
    for name, value in (('evaluation.json', evaluation), ('completion.json', completion),
                        ('settings.json', settings_report)):
        archive.atomic_json(run / name, value)
    (run / 'metrics.jsonl').write_text(json.dumps(evaluation) + '\n')
    archive.atomic_json(work / 'control/execution.json', {'plan': plan})
    archive.atomic_json(work / 'control/handoff.json', {'phase': 'complete'})
    archive.atomic_json(work / 'preflight/report.json', {'accepted': True})
    return dict(work=work, run=run, reference_dir=reference_dir, tokenizer=tokenizer,
        destination=base / 'archives/dpo-final', plan=plan, checkpoint=checkpoint,
        evaluation=evaluation, completion=completion, settings=settings_report,
        data=data, reference=reference)


def create(fixture):
    return archive.archive_dpo(fixture['work'], fixture['tokenizer'], fixture['destination'],
                               fixture['plan'], 'test-revision')


def test_full_dpo_archive_exact_model_optimizer_rng_and_idempotent_reuse(fixture):
    source = fixture['run'] / 'latest.pt'
    before = source.read_bytes()
    metadata, entries = create(fixture)
    folder = fixture['destination']
    assert (folder / 'checkpoint.pt').read_bytes() == before == source.read_bytes()
    assert (folder / 'checkpoint.pt').stat().st_ino != source.stat().st_ino
    full = torch.load(folder / 'checkpoint.pt', weights_only=False, mmap=True)
    model = torch.load(folder / 'model.pt', weights_only=False, mmap=True)
    assert model['stage'] == 'dpo' and 'optimizer' not in model
    assert torch.equal(full['model']['weight'], model['model']['weight'])
    assert full['optimizer']['state'][0]['step'].item() == 1
    assert len(full['rng_by_rank']) == 2
    assert model['origin']['step'] == 10312 and model['origin']['origin']['step'] == 76294
    assert entries['checkpoint.pt'] == archive.sha256(source)
    assert all(path.stat().st_mode & 0o222 == 0 for path in folder.iterdir())
    original_inodes = {path.name: path.stat().st_ino for path in folder.iterdir()}
    assert create(fixture) == (metadata, entries)
    assert {path.name: path.stat().st_ino for path in folder.iterdir()} == original_inodes
    assert archive.verify_dpo_archive(folder) == (metadata, entries)


@pytest.mark.parametrize('field,value', [('completed', False), ('step', 0), ('sample_cursor', 1),
    ('tokens_seen', 15), ('response_tokens_seen', 7), ('val_step', 0), ('val_loss', float('nan'))])
def test_incomplete_or_wrong_final_state_is_rejected(fixture, field, value):
    fixture['checkpoint'][field] = value
    torch.save(fixture['checkpoint'], fixture['run'] / 'latest.pt')
    with pytest.raises(ValueError):
        create(fixture)
    assert not fixture['destination'].exists()


def test_changed_reference_or_subset_evaluation_is_rejected(fixture):
    fixture['evaluation']['full_holdout'] = False
    with pytest.raises(ValueError, match='full-holdout'):
        archive.validate_state(fixture['checkpoint'], fixture['evaluation'], fixture['completion'],
            fixture['settings'], fixture['data'], fixture['reference'], fixture['plan'])
    fixture['evaluation']['full_holdout'] = True
    fixture['reference']['identity']['weights_sha256'] = 'changed-reference'
    with pytest.raises(ValueError, match='reference'):
        archive.validate_state(fixture['checkpoint'], fixture['evaluation'], fixture['completion'],
            fixture['settings'], fixture['data'], fixture['reference'], fixture['plan'])


def test_existing_unrecognized_archive_is_never_changed(fixture):
    fixture['destination'].mkdir(parents=True)
    marker = fixture['destination'] / 'keep.txt'
    marker.write_text('user data')
    with pytest.raises(ValueError, match='checksum'):
        create(fixture)
    assert list(fixture['destination'].iterdir()) == [marker]
    assert marker.read_text() == 'user data'


def test_corrupt_sealed_reference_copy_is_detected(fixture):
    create(fixture)
    array = fixture['destination'] / 'reference-train.npy'
    array.chmod(0o644)
    array.write_bytes(array.read_bytes() + b'changed')
    array.chmod(0o444)
    with pytest.raises(ValueError, match='checksum'):
        archive.verify_dpo_archive(fixture['destination'])


def test_source_replacement_during_archive_blocks_publication(fixture, monkeypatch):
    real_copy = archive.copy_regular
    def changing_copy(source, destination):
        real_copy(source, destination)
        if destination.name == 'tokenizer.model':
            checkpoint = copy.deepcopy(fixture['checkpoint'])
            replacement = fixture['run'] / 'replacement.pt'
            torch.save(checkpoint, replacement)
            replacement.replace(fixture['run'] / 'latest.pt')
    monkeypatch.setattr(archive, 'copy_regular', changing_copy)
    with pytest.raises(ValueError, match='changed during archival'):
        create(fixture)
    assert not fixture['destination'].exists()


def test_running_guardian_blocks_before_archival(fixture, monkeypatch):
    paths = dict(work_dir=fixture['work'], base_work_dir=fixture['work'].parents[1],
        sft_work_dir=fixture['work'].parent / 'sft', archive_dir=fixture['destination'].parent / 'sft-final',
        supervisor_socket=fixture['work'] / 'control/socket', tokenizer=fixture['tokenizer'])
    monkeypatch.setattr(archive, 'load_plan', lambda p: (fixture['plan'], paths))
    monkeypatch.setattr(archive, 'process_status', lambda *a: {'statename': 'RUNNING', 'exitstatus': 0})
    monkeypatch.setattr(archive, 'archive_dpo', lambda *a: pytest.fail('must not archive an active job'))
    with pytest.raises(ValueError, match='exited successfully'):
        archive.run(fixture['tokenizer'], fixture['destination'])
