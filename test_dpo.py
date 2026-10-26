"""CPU DPO arithmetic, frozen-reference and exact continuation checks."""

import copy
import json
import math
import shutil

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import dpo
from ops.dpo_data import collate_pairs, encode_pair, load_data
from ops.dpo_preflight import make_fixture
from ops.posttrain_data import encode_messages
from test_posttrain import CharacterTokenizer, assert_tree_equal


def pair():
    return dict(prompt=[{'role': 'user', 'content': 'hello'},
                        {'role': 'assistant', 'content': 'old answer'},
                        {'role': 'user', 'content': 'two plus two?'}],
                chosen=[{'role': 'assistant', 'content': 'four'}],
                rejected=[{'role': 'assistant', 'content': 'five, perhaps six'}])


def test_multiturn_masks_only_final_response_and_eos():
    tokenizer = CharacterTokenizer()
    row = pair()
    encoded = encode_pair(row, tokenizer, 512)
    for branch in ('chosen', 'rejected'):
        assert encoded[branch]['input_ids'] == encode_messages(row['prompt'] + row[branch], tokenizer)['input_ids']
        assert [value for value in encoded[branch]['labels'] if value != -100] == [ord(c) for c in row[branch][0]['content']] + [2]
    x, y, active = collate_pairs([encoded, None], 512, 0)
    assert active.tolist() == [True, False]
    assert not y[2:].ne(-100).any()
    assert x.shape == y.shape == (4, 512)


@pytest.mark.parametrize('mutation', [lambda r: r.update(prompt=[]),
    lambda r: r['prompt'].append({'role': 'assistant', 'content': 'invalid end'}),
    lambda r: r.update(rejected=r['chosen']),
    lambda r: r['chosen'].append({'role': 'assistant', 'content': 'extra'})])
def test_invalid_preference_pairs_rejected(mutation):
    row = pair()
    mutation(row)
    with pytest.raises(ValueError):
        encode_pair(row, CharacterTokenizer(), 512)


def test_length_normalized_and_summed_objectives_match_explicit_formula():
    policy = torch.tensor([[-10., -19.], [-4., -9.]], dtype=torch.float64, requires_grad=True)
    reference = torch.tensor([[-12., -20.], [-4., -8.]], dtype=torch.float64, requires_grad=True)
    counts = torch.tensor([[2, 10], [1, 3]])
    loss, margin, accuracy = dpo.preference_metrics(policy, reference, counts, 5., 'length-normalized-dpo')
    expected = 5 * ((policy[:, 0] - reference.detach()[:, 0]) / counts[:, 0] -
                    (policy[:, 1] - reference.detach()[:, 1]) / counts[:, 1])
    torch.testing.assert_close(loss, -F.logsigmoid(expected))
    torch.testing.assert_close(margin, expected)
    assert accuracy.tolist() == [1., 1.]
    loss.sum().backward()
    assert reference.grad is None
    assert (policy.grad[:, 0] < 0).all() and (policy.grad[:, 1] > 0).all()
    summed, _, _ = dpo.preference_metrics(policy.detach(), reference, counts, .1, 'summed-dpo')
    expected = .1 * ((policy[:, 0] - reference[:, 0]) - (policy[:, 1] - reference[:, 1]))
    torch.testing.assert_close(summed, -F.logsigmoid(expected))


def test_global_pair_mean_handles_empty_rank_and_unequal_lengths():
    parameter = torch.tensor(.2, dtype=torch.float64, requires_grad=True)
    counts = torch.tensor([[2, 10], [4, 2], [1, 6]])
    base = torch.tensor([[-10., -19.], [-5., -3.], [-2., -8.]], dtype=torch.float64)
    direction = torch.tensor([[1., 2.], [3., 1.], [2., 5.]], dtype=torch.float64)
    gradients = []
    for ids in ([0, 1, 2], []):
        parameter.grad = None
        if ids:
            loss, _, _ = dpo.preference_metrics(base[ids] + parameter * direction[ids], base[ids], counts[ids], 5., 'length-normalized-dpo')
            (loss.sum() * 2 / 3).backward()
        else:
            (parameter * 0.).backward()
        gradients.append(parameter.grad.clone())
    parameter.grad = None
    loss, _, _ = dpo.preference_metrics(base + parameter * direction, base, counts, 5., 'length-normalized-dpo')
    loss.mean().backward()
    torch.testing.assert_close(sum(gradients) / 2, parameter.grad)


def test_chunked_sequence_logps_and_gradients_match_full_logits():
    from warpstate import WarpStateConfig, WarpStateLM
    cfg = WarpStateConfig(vocab_size=32, d_model=16, n_heads=4, n_cores=1,
        logical_depth=1, ffn_hidden=32, chunk_size=4, parallel_memory=True)
    torch.manual_seed(19)
    original = WarpStateLM(cfg)
    chunked = copy.deepcopy(original)
    dpo.apply_sequence_logps(chunked, 3)
    x = torch.randint(0, 32, (4, 8))
    y = torch.randint(0, 32, (4, 8))
    y[:, :2] = -100
    y[3] = -100
    logits = original(x)
    expected = -F.cross_entropy(logits.reshape(-1, 32), y.reshape(-1), reduction='none').reshape(4, 8).sum(1)
    _, actual = chunked(x, y, return_logits=False)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
    expected.sum().backward()
    actual.sum().backward()
    for a, b in zip(original.parameters(), chunked.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=2e-6)
    assert actual[3] == 0


def test_exact_bf16_resume_rebuilt_reference_and_immutable_sft(tmp_path):
    args = make_fixture(tmp_path / 'fixture')
    original_hash = dpo.digest(args.weights)
    original = torch.load(args.weights, weights_only=False, mmap=True)
    try:
        dpo.train(args)
        uninterrupted = torch.load(args.work_dir / 'latest.pt', weights_only=False, mmap=True)
        original_cache = dpo.digest(args.reference_cache_dir / 'manifest.json')
        args.work_dir = tmp_path / 'split'
        args.max_steps = 1
        dpo.train(args)
        partial = torch.load(args.work_dir / 'latest.pt', weights_only=False, mmap=True)
        assert partial['step'] == 1 and not partial['completed']
        # A deterministic external-data cache can be rebuilt while resuming a
        # changed policy; the source remains the frozen SFT, never latest.pt.
        shutil.rmtree(args.reference_cache_dir)
        args.max_steps = None
        dpo.train(args)
        resumed = torch.load(args.work_dir / 'latest.pt', weights_only=False, mmap=True)
        assert dpo.digest(args.reference_cache_dir / 'manifest.json') == original_cache
        assert resumed['step'] == 3 and resumed['sample_cursor'] == 5 and resumed['completed']
        for key in ('model', 'optimizer', 'sample_cursor', 'tokens_seen', 'response_tokens_seen', 'validation'):
            assert_tree_equal(uninterrupted[key], resumed[key])
        assert torch.equal(uninterrupted['rng_by_rank'][0]['torch'], resumed['rng_by_rank'][0]['torch'])
        assert all(v.dtype == torch.float32 for v in resumed['model'].values())
        assert any(not torch.equal(original['model'][key], value) for key, value in resumed['model'].items())
        assert dpo.digest(args.weights) == original_hash
        metrics = [json.loads(line) for line in (tmp_path / 'fixture/run/metrics.jsonl').read_text().splitlines()]
        assert metrics[0]['val_loss'] == pytest.approx(math.log(2), abs=1e-7)
        report = json.loads((args.work_dir / 'evaluation.json').read_text())
        assert report['full_holdout'] and report['evaluated_samples'] == 2
        completion = json.loads((args.work_dir / 'completion.json').read_text())
        assert completion['paused_for_evaluation'] and completion['completed']
    finally:
        torch.use_deterministic_algorithms(False)


def test_reference_identity_tampering_and_incomplete_data_fail_closed(tmp_path):
    import sentencepiece as spm
    args = make_fixture(tmp_path / 'fixture')
    args.reference_only = True
    try:
        dpo.train(args)
        manifest_path = args.reference_cache_dir / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        identity = manifest['identity']
        with pytest.raises(ValueError, match='identity'):
            dpo.validate_cache(args.reference_cache_dir, identity | {'weights_sha256': 'changed'}, (5, 2))
        array = args.reference_cache_dir / 'train.npy'
        original = array.read_bytes()
        array.write_bytes(original + b'changed')
        with pytest.raises(ValueError, match='changed'):
            dpo.validate_cache(args.reference_cache_dir, identity, (5, 2))
        array.write_bytes(original)
        manifest['complete'] = False
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match='completion'):
            dpo.validate_cache(args.reference_cache_dir, identity, (5, 2))
        data_manifest_path = args.data_dir / 'manifest.json'
        data_manifest = json.loads(data_manifest_path.read_text())
        data_manifest['renderer_sha256'] = 'changed'
        data_manifest_path.write_text(json.dumps(data_manifest))
        with pytest.raises(ValueError, match='renderer'):
            load_data(args.data_dir, spm.SentencePieceProcessor(model_file=str(args.tokenizer)), dpo.digest(args.tokenizer), 128)
    finally:
        torch.use_deterministic_algorithms(False)


def test_schedule_is_linear_and_pause_independent():
    settings = dpo.DEFAULTS | dict(lr=1., min_lr=0., warmup_ratio=.1)
    assert dpo.learning_rate(0, 100, settings) == .1
    assert dpo.learning_rate(9, 100, settings) == 1.
    assert dpo.learning_rate(10, 100, settings) == 1.
    assert dpo.learning_rate(54, 100, settings) == pytest.approx(45 / 89)
    assert dpo.learning_rate(99, 100, settings) == 0.


@pytest.mark.parametrize('settings', [dict(beta=0), dict(objective='dpo'), dict(micro_batch=True),
    dict(lr=0), dict(warmup_ratio=1), dict(unknown=True), dict(head_chunk_size=0)])
def test_invalid_plan_rejected(tmp_path, settings):
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(dict(format='warpstate-dpo-plan-v1', stage='dpo', training=settings)))
    with pytest.raises(ValueError):
        dpo.load_plan(path)
