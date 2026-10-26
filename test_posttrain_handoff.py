"""CPU tests for the unattended base -> immutable archive -> separate SFT gate."""
import copy
import json
from pathlib import Path
import stat

import pytest
import torch

from ops import posttrain_handoff as handoff


@pytest.fixture
def fixture(tmp_path):
    base_work = tmp_path / 'base'
    (base_work / 'run').mkdir(parents=True)
    (base_work / 'control').mkdir()
    tokenizer = tmp_path / 'tokenizer.model'
    tokenizer.write_bytes(b'original required tokenizer fixture')
    base = dict(step=76294, tokens_seen=10000007168, tokenizer_sha256=handoff.sha256(tokenizer))
    checkpoint = dict(format='warpstate-server-v1', config=dict(d_model=2),
                      step=base['step'], tokens_seen=base['tokens_seen'],
                      val_step=base['step'], val_tokens=base['tokens_seen'], val_loss=2.3,
                      signature=dict(data=dict(signature=dict(tokenizer_sha256=base['tokenizer_sha256']))),
                      model={'weights': torch.arange(6, dtype=torch.float32).reshape(2, 3)},
                      optimizer={'unmodified': torch.tensor([12.0])})
    source = base_work / 'run/latest.pt'
    torch.save(checkpoint, source)
    plan = dict(format='warpstate-posttrain-plan-v1', stage='sft', after_completion='pause_for_evaluation',
                base=base, handoff=dict(base_work_dir=str(base_work), base_program='warpstate-train',
                    supervisor_socket=str(base_work / 'control/supervisor.sock'),
                    archive_dir=str(tmp_path / 'archives/step-00076294'),
                    work_dir=str(tmp_path / 'posttrain'), data_dir=str(tmp_path / 'data'),
                    tokenizer=str(tokenizer), processes=2, poll_seconds=1),
                training=dict(sequence_length=2048))
    config = tmp_path / 'plan.json'
    config.write_text(json.dumps(plan))
    return dict(root=tmp_path, source=source, tokenizer=tokenizer, base=base, checkpoint=checkpoint,
                archive=tmp_path / 'archives/step-00076294', plan=plan, config=config)


@pytest.mark.parametrize('state,exitcode,expected', [
    ('RUNNING', 0, False), ('STOPPED', 0, False), ('STOPPING', 0, False),
    ('EXITED', 1, False), ('FATAL', 0, False), ('EXITED', 0, True)])
def test_only_clean_supervisor_exit_opens_gate(state, exitcode, expected):
    assert handoff.clean_exit(dict(statename=state, exitstatus=exitcode)) is expected


@pytest.mark.parametrize('field,value', [('step', 76293), ('step', 76295),
    ('tokens_seen', 10000000000), ('val_step', 76200), ('val_tokens', 0),
    ('val_loss', float('nan')), ('format', 'warpstate-chat-v1')])
def test_wrong_or_stale_boundary_rejected(fixture, field, value):
    checkpoint = copy.deepcopy(fixture['checkpoint'])
    checkpoint[field] = value
    with pytest.raises(ValueError):
        handoff.validate_boundary(checkpoint, fixture['base'], fixture['base']['tokenizer_sha256'])


def test_archive_and_copy_preserve_exact_raw_tensors_and_bytes(fixture):
    source_hash = handoff.sha256(fixture['source'])
    metadata = handoff.archive_checkpoint(fixture['source'], fixture['tokenizer'], fixture['archive'],
                                         fixture['base'], 'test-revision')
    original = torch.load(fixture['source'], weights_only=False, mmap=True)
    exported = torch.load(fixture['archive'] / 'model.pt', weights_only=False, mmap=True)
    assert torch.equal(original['model']['weights'], exported['model']['weights'])
    assert 'optimizer' not in exported and 'rng_by_rank' not in exported
    assert handoff.sha256(fixture['source']) == source_hash
    hashes = {p.name: handoff.sha256(p) for p in fixture['archive'].iterdir()}
    assert all(not stat.S_IMODE(p.stat().st_mode) & 0o222 for p in fixture['archive'].iterdir())
    model = handoff.independent_initialization(fixture['archive'], fixture['root'] / 'init',
                                              fixture['base'], metadata['snapshot_sha256'])
    assert handoff.sha256(model) == metadata['snapshot_sha256']
    assert model.stat().st_ino != (fixture['archive'] / 'model.pt').stat().st_ino
    # Repeated handoff reuses the original archive/copy without changing bytes.
    inode = model.stat().st_ino
    assert handoff.archive_checkpoint(fixture['source'], fixture['tokenizer'], fixture['archive'],
                                      fixture['base'], 'different-revision') == metadata
    handoff.independent_initialization(fixture['archive'], model.parent, fixture['base'],
                                      metadata['snapshot_sha256'])
    assert model.stat().st_ino == inode
    assert {p.name: handoff.sha256(p) for p in fixture['archive'].iterdir()} == hashes


def test_incomplete_existing_archive_is_not_replaced(fixture):
    fixture['archive'].mkdir(parents=True)
    marker = fixture['archive'] / 'user-file'
    marker.write_text('keep')
    with pytest.raises(ValueError, match='checksum'):
        handoff.archive_checkpoint(fixture['source'], fixture['tokenizer'], fixture['archive'],
                                   fixture['base'], 'test')
    assert marker.read_text() == 'keep'
    assert list(fixture['archive'].iterdir()) == [marker]


def test_publishing_does_not_replace_even_an_empty_directory(tmp_path):
    staging, target = tmp_path / 'staging', tmp_path / 'target'
    staging.mkdir()
    target.mkdir()
    (staging / 'payload').write_text('complete')
    inode = target.stat().st_ino
    with pytest.raises(FileExistsError):
        handoff.publish_directory(staging, target)
    assert target.stat().st_ino == inode and (staging / 'payload').exists()


def prepare_data(fixture):
    directory = Path(fixture['plan']['handoff']['data_dir'])
    directory.mkdir()
    outputs = {}
    for name in ('sft_train.jsonl', 'sft_val.jsonl'):
        (directory / name).write_text('{}\n')
        outputs[name] = dict(rows=1, sha256=handoff.sha256(directory / name))
    manifest = dict(format='warpstate-posttrain-data-v1', complete=True,
                    tokenizer_sha256=fixture['base']['tokenizer_sha256'], max_length=2048, outputs=outputs)
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    return directory


def test_data_hashes_must_match_before_launch(fixture):
    directory = prepare_data(fixture)
    assert handoff.data_readiness(directory, fixture['plan'], verify=True)[0]
    (directory / 'sft_train.jsonl').write_text('changed\n')
    with pytest.raises(ValueError, match='checksum'):
        handoff.data_readiness(directory, fixture['plan'], verify=True)


def test_corrupted_raw_tensor_cannot_reuse_existing_archive(fixture):
    handoff.archive_checkpoint(fixture['source'], fixture['tokenizer'], fixture['archive'], fixture['base'], 'test')
    fixture['checkpoint']['model']['weights'][0, 0] = 99
    torch.save(fixture['checkpoint'], fixture['source'])
    with pytest.raises(ValueError, match='tensor'):
        handoff.archive_checkpoint(fixture['source'], fixture['tokenizer'], fixture['archive'], fixture['base'], 'test')


def test_complete_handoff_waits_archives_and_passes_separate_output_paths(fixture, monkeypatch):
    prepare_data(fixture)
    monkeypatch.setattr(handoff, 'STOPPED', False)
    monkeypatch.setattr(handoff, 'source_identity', lambda config: {'fixture': 'immutable'})
    monkeypatch.setattr(handoff, 'verify_preflight', lambda *args: {'accepted': True})
    statuses = iter([dict(statename='RUNNING', exitstatus=0), dict(statename='EXITED', exitstatus=0)])
    monkeypatch.setattr(handoff, 'process_status', lambda *args: next(statuses, dict(statename='EXITED', exitstatus=0)))
    monkeypatch.setattr(handoff.time, 'sleep', lambda seconds: None)
    commands = []
    base_work = Path(fixture['plan']['handoff']['base_work_dir'])
    def fake_train(command, stop, env, *, pass_fds):
        commands.append(command)
        assert len(pass_fds) == 2
        with pytest.raises(BlockingIOError):
            with handoff.exclusive_lock(base_work / 'control/training-launcher.lock'):
                pass
        # Released precisely before worker SH acquisition.
        with handoff.exclusive_lock(base_work / 'pipeline.lock'):
            pass
        if 'posttrain_preflight.py' in ' '.join(command):
            return 0
        work = Path(command[command.index('--work-dir') + 1])
        weights = Path(command[command.index('--weights') + 1])
        assert not weights.is_relative_to(work)
        assert Path(command[command.index('--stop-file') + 1]) == stop
        assert work.name == 'run'
        work.mkdir(parents=True)
        model_hash = handoff.sha256(weights)
        completion = dict(format='warpstate-posttrain-completion-v1', stage='sft',
                          completed=True, paused_for_evaluation=True, step=1,
                          total_steps=1, base_step=fixture['base']['step'])
        checkpoint = dict(format='warpstate-posttrain-v1', completed=True, step=1, total_steps=1,
                          val_step=1, val_loss=2., signature=dict(plan=fixture['plan'], base_sha256=model_hash),
                          origin=dict(step=fixture['base']['step']))
        torch.save(checkpoint, work / 'latest.pt')
        (work / 'completion.json').write_text(json.dumps(completion))
        (work / 'evaluation.json').write_text(json.dumps(dict(stage='sft', step=1,
                                                             base_sha256=model_hash, val_loss=2.)))
        return 0
    monkeypatch.setattr(handoff, 'supervise', fake_train)
    assert handoff.run_handoff(fixture['config']) == 0
    state = json.loads((fixture['root'] / 'posttrain/control/handoff.json').read_text())
    assert state['phase'] == 'complete'
    count = len(commands)
    assert handoff.run_handoff(fixture['config']) == 0
    assert len(commands) == count  # Never retrain after terminal completion.


def test_overshoot_never_archives_or_trains(fixture, monkeypatch):
    prepare_data(fixture)
    fixture['checkpoint']['step'] = 76295
    torch.save(fixture['checkpoint'], fixture['source'])
    monkeypatch.setattr(handoff, 'STOPPED', False)
    monkeypatch.setattr(handoff, 'source_identity', lambda config: {'fixture': 'immutable'})
    monkeypatch.setattr(handoff, 'process_status', lambda *args: dict(statename='EXITED', exitstatus=0))
    monkeypatch.setattr(handoff, 'supervise', lambda *a, **kw: pytest.fail('Must not start SFT'))
    with pytest.raises(ValueError, match='step must equal'):
        handoff.run_handoff(fixture['config'])
    assert not fixture['archive'].exists()


def test_sft_completion_requires_matching_durable_checkpoint(fixture, tmp_path):
    (tmp_path / 'completion.json').write_text(json.dumps(dict(completed=True)))
    torch.save(dict(format='warpstate-posttrain-v1', completed=False, step=1, total_steps=2,
                    val_step=1, val_loss=2.), tmp_path / 'latest.pt')
    (tmp_path / 'evaluation.json').write_text('{}')
    with pytest.raises(ValueError, match='fully evaluated'):
        handoff.verify_completion(tmp_path, fixture['plan'], 'hash')
