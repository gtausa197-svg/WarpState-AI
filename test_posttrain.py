"""CPU correctness tests; no GPU allocation or production state is involved."""

import copy
import json
from pathlib import Path

import pytest
import torch

import posttrain
from ops.posttrain_data import encode_messages, renderer_sha256
from ops.posttrain_preflight import make_fixture
from warpstate import WarpStateConfig, WarpStateLM


class CharacterTokenizer:
    def eos_id(self):
        return 2

    def bos_id(self):
        return 1

    def vocab_size(self):
        return 256

    def encode(self, text, out_type=int):
        return [ord(char) for char in text]


def messages(answer='four'):
    return [{'role': 'system', 'content': 'Be concise.'},
            {'role': 'user', 'content': 'Two plus two?'},
            {'role': 'assistant', 'content': answer}]


def test_assistant_content_and_real_eos_only():
    rows = messages() + [{'role': 'tool', 'content': 'untrusted result'},
                         {'role': 'assistant', 'content': 'done'}]
    encoded = encode_messages(rows, CharacterTokenizer())
    targets = [value for value in encoded['labels'] if value != -100]
    assert targets == [ord(c) for c in 'four'] + [2] + [ord(c) for c in 'done'] + [2]
    assert encoded['input_ids'][0] == 1
    assert len(encoded['input_ids']) == len(encoded['labels'])
    with pytest.raises(ValueError, match='exceeds'):
        encode_messages(rows, CharacterTokenizer(), max_length=len(encoded['input_ids']) - 1)
    assert encode_messages(rows, CharacterTokenizer(), max_length=len(encoded['input_ids'])) == encoded


@pytest.mark.parametrize('bad', [[], [{'role': 'user', 'content': 'no answer'}],
                               [{'role': 'unknown', 'content': 'text'}],
                               [{'role': 'assistant', 'content': ''}],
                               [{'role': 'assistant', 'content': 'call', 'tool_calls': [{'name': 'x'}]}]])
def test_invalid_conversations_rejected(bad):
    with pytest.raises(ValueError):
        encode_messages(bad, CharacterTokenizer())


def test_collation_shifts_and_masks_independent_padding():
    a = {'input_ids': [1, 4, 5, 2], 'labels': [-100, -100, 5, 2]}
    x, y = posttrain.collate([a, None], 8, 0)
    assert x.tolist() == [[1, 4, 5, 0, 0, 0, 0, 0], [0] * 8]
    assert y.tolist() == [[-100, 5, 2, -100, -100, -100, -100, -100], [-100] * 8]


def test_sample_order_resume_and_tail_do_not_duplicate():
    cursor, groups = 0, []
    for _ in range(6):
        values, cursor = posttrain.update_indices(5, 17, cursor, 2, 2)
        groups.append(values)
    assert sorted(sum(groups[:3], [])) == list(range(5))
    assert sorted(sum(groups[3:], [])) == list(range(5))
    assert [len(v) for v in groups] == [2, 2, 1, 2, 2, 1]
    assert posttrain.update_indices(5, 17, 4, 2, 2)[0] == groups[2]
    assert posttrain.update_indices(5, 17, cursor, 2, 2) == ([], 10)


def test_ddp_token_weighting_matches_global_objective_with_empty_rank():
    # Simulate DDP's gradient average with unequal rank/microbatch target counts.
    parameter = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    partitions = [[torch.tensor([1., 2., 4.]), torch.tensor([7.])],
                  [torch.tensor([]), torch.tensor([3., 8.])]]
    gradients = []
    for rank in partitions:
        parameter.grad = None
        for target in rank:
            target = target.double()
            mean = ((parameter - target) ** 2).sum() / max(1, len(target))
            posttrain.token_weighted_loss(mean, len(target), 6, 2).backward()
        gradients.append(parameter.grad.clone())
    parameter.grad = None
    target = torch.tensor([1., 2., 4., 7., 3., 8.], dtype=torch.float64)
    ((parameter - target) ** 2).mean().backward()
    torch.testing.assert_close(sum(gradients) / 2, parameter.grad, rtol=1e-14, atol=1e-14)


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    else:
        assert a == b


def test_real_bf16_training_exact_resume_and_frozen_base(tmp_path):
    args = make_fixture(tmp_path)
    original_hash = posttrain.digest(args.weights)
    original = torch.load(args.weights, weights_only=False, mmap=True)
    try:
        posttrain.train(args)
        uninterrupted = torch.load(args.work_dir / 'latest.pt', weights_only=False, mmap=True)
        args.work_dir = tmp_path / 'resumed'
        args.max_steps = 1
        posttrain.train(args)
        partial = torch.load(args.work_dir / 'latest.pt', weights_only=False, mmap=True)
        assert partial['step'] == 1 and partial['completed'] is False
        assert not (args.work_dir / 'completion.json').exists()
        args.max_steps = None
        posttrain.train(args)
        resumed = torch.load(args.work_dir / 'latest.pt', weights_only=False, mmap=True)
        assert resumed['step'] == 3 and resumed['sample_cursor'] == 5
        assert resumed['completed'] is True
        for key in ('model', 'optimizer', 'sample_cursor', 'tokens_seen', 'assistant_tokens_seen', 'val_loss'):
            assert_tree_equal(uninterrupted[key], resumed[key])
        assert any(not torch.equal(original['model'][k], v) for k, v in resumed['model'].items())
        assert all(v.dtype == torch.float32 for v in resumed['model'].values())
        assert resumed['origin']['step'] == 76294
        report = json.loads((args.work_dir / 'evaluation.json').read_text())
        assert report['step'] == 3 and report['evaluated_samples'] == 2
        assert report['objective'] == 'assistant-token-mean'
        completion = json.loads((args.work_dir / 'completion.json').read_text())
        assert completion['completed'] is True and completion['paused_for_evaluation'] is True
        assert completion['step'] == resumed['step']
        assert completion['val_loss'] == resumed['val_loss']
        assert torch.equal(uninterrupted['rng_by_rank'][0]['torch'], resumed['rng_by_rank'][0]['torch'])
        assert posttrain.digest(args.weights) == original_hash
        bad = copy.deepcopy(resumed['signature'])
        bad['world_size'] = 2
        with pytest.raises(ValueError, match='signature'):
            posttrain.validate_resume(resumed, bad, 2)
    finally:
        torch.use_deterministic_algorithms(False)


def test_ready_manifest_and_per_record_counts_are_enforced(tmp_path):
    import sentencepiece as spm
    args = make_fixture(tmp_path)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    path = args.data_dir / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['complete'] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='not complete'):
        posttrain.load_data(args.data_dir, sp, posttrain.digest(args.tokenizer), 128)
    manifest['complete'] = True
    manifest['renderer_sha256'] = 'wrong'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='renderer'):
        posttrain.load_data(args.data_dir, sp, posttrain.digest(args.tokenizer), 128)


def test_inputs_cannot_be_training_outputs(tmp_path):
    args = make_fixture(tmp_path)
    for output in (args.data_dir, args.weights.parent / 'run', tmp_path):
        with pytest.raises(ValueError, match='separate|inside'):
            posttrain.validate_paths(output, args.data_dir, args.weights, args.tokenizer, args.config)


def test_wrong_base_token_boundary_is_rejected(tmp_path):
    args = make_fixture(tmp_path)
    plan = json.loads(args.config.read_text())
    plan['base']['tokens_seen'] = 999
    args.config.write_text(json.dumps(plan))
    try:
        with pytest.raises(ValueError, match='tokens_seen differs'):
            posttrain.train(args)
    finally:
        torch.use_deterministic_algorithms(False)


@pytest.mark.parametrize('bad', [{'stage': 'dpo'}, {'training': {'min_lr': -1}},
                               {'training': {'warmup_ratio': 1}}, {'training': {'lr': 0}},
                               {'training': {'bogus': True}}, {'training': {'micro_batch': True}}])
def test_plan_rejects_unsupported_settings(tmp_path, bad):
    plan = dict(format='warpstate-posttrain-plan-v1', stage='sft') | bad
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        posttrain.load_plan(path)


def test_schedule_is_fixed_by_plan_not_pause_boundary():
    cfg = posttrain.DEFAULTS | dict(warmup_ratio=.1)
    assert posttrain.learning_rate(0, 100, cfg) == pytest.approx(cfg['lr'] / 10)
    assert posttrain.learning_rate(9, 100, cfg) == pytest.approx(cfg['lr'])
    assert posttrain.learning_rate(99, 100, cfg) == 0


def test_preflight_fixture_identity_and_command_are_cpu_only(tmp_path):
    from ops.posttrain_preflight import training_command, compare_tree
    fixture = make_fixture(tmp_path, device='cuda', compile_core=True)
    repeated = make_fixture(tmp_path, device='cuda', compile_core=True)
    assert repeated.weights == fixture.weights
    cfg = torch.load(fixture.weights, weights_only=False, mmap=True)['config']
    assert cfg['d_model'] == 32 and cfg['n_heads'] == 4
    command = training_command(fixture, 2, tmp_path / 'split', tmp_path / 'stop', 1)
    assert command[-2:] == ['--max-steps', '1']
    assert '--device' in command and command[command.index('--device') + 1] == 'cuda'
    with pytest.raises(ValueError, match='two ranks'):
        training_command(fixture, 1, tmp_path / 'split', tmp_path / 'stop')
    with pytest.raises(ValueError, match='settings/source differ'):
        make_fixture(tmp_path, device='cpu', compile_core=True)
    a = torch.tensor([1.], dtype=torch.float32)
    b = a + 1e-6
    report = compare_tree(a, b, tolerant=True)
    assert report['bitwise_equal'] is False and report['max_absolute_difference'] > 0
    with pytest.raises(ValueError, match='tolerance'):
        compare_tree(a, a + 1e-3, tolerant=True)
    with pytest.raises(ValueError, match='tolerance'):
        compare_tree(a, b)


def test_evaluation_full_holdout_exceeds_intermediate_subset():
    class ConstantLoss(torch.nn.Module):
        def forward(self, x, y, return_logits=False):
            return None, torch.tensor(2.)
    row = dict(input_ids=[1, 3, 4, 2], labels=[-100, -100, 4, 2])
    settings = posttrain.DEFAULTS | dict(eval_batches=1, micro_batch=1, sequence_length=8)
    model = ConstantLoss()
    assert posttrain.evaluate(model, [row] * 3, settings, torch.device('cpu'), 0) == (2., 2)
    assert posttrain.evaluate(model, [row] * 3, settings, torch.device('cpu'), 0, full=True) == (2., 6)
    assert model.training
