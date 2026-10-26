"""Tiny CPU-only chat checks; all generated fixtures stay inside this workspace."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch

import chat
from warpstate import WarpStateConfig, WarpStateLM

ROOT = Path(__file__).resolve().parent
TOKENIZER = ROOT / 'tokenizer.model'


@pytest.fixture
def fixture_dir():
    root = ROOT.parent / 'tmp/chat-test-fixtures'
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        yield Path(directory)


def tiny(vocab_size=32, parallel_memory=False):
    torch.manual_seed(17)
    cfg = WarpStateConfig(vocab_size=vocab_size, d_model=16, n_heads=2,
                          n_cores=1, logical_depth=3, ffn_hidden=24, chunk_size=4,
                          gradient_checkpointing=False, parallel_memory=parallel_memory)
    return WarpStateLM(cfg).eval()


def source_checkpoint(directory):
    model = tiny(32768)
    state = dict(format='warpstate-server-v1', config=vars(model.cfg),
                 model=model.state_dict(), step=7, tokens_seen=112,
                 optimizer={'state': {'must_not_survive': torch.ones(3)}},
                 signature={'data': {'signature': {
                     'tokenizer_sha256': hashlib.sha256(TOKENIZER.read_bytes()).hexdigest()}}})
    source = directory / 'latest.pt'
    torch.save(state, source)
    return source, state


@pytest.mark.parametrize('parallel_memory', [False, True])
def test_step_matches_full_forward_across_three_chunk_boundaries(parallel_memory):
    model = tiny(parallel_memory=parallel_memory)
    tokens = torch.randint(0, 32, (2, 12))
    caches = model.init_generation_cache(2, torch.device('cpu'), torch.float32)
    with torch.inference_mode():
        expected = model(tokens)
        actual = torch.cat([model.step(tokens[:, i:i+1], caches) for i in range(12)], dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert all(cache.chunk_len == 0 for cache in caches)


def test_shared_core_has_independent_logical_depth_caches():
    model = tiny()
    caches = model.init_generation_cache(1, torch.device('cpu'), torch.float32)
    assert len(caches) == model.cfg.logical_depth > model.cfg.n_cores
    for field in ('fast_state', 'slow_state', 'k_cache', 'v_cache', 'write_accum'):
        assert len({getattr(cache, field).data_ptr() for cache in caches}) == len(caches)
    for _ in range(3):
        model.step(torch.tensor([[4]]), caches)
    assert all(cache.chunk_len == 3 and not cache.fast_state.count_nonzero() for cache in caches)
    model.step(torch.tensor([[4]]), caches)
    assert all(cache.chunk_len == 0 and not cache.k_cache.count_nonzero() for cache in caches)


def test_sampling_greedy_masks_and_seeded_replay():
    logits = torch.tensor([0.2, 0.8, -10.0, 0.1])
    before = logits.clone()
    assert chat.sample_token(logits, 0.0, 0, 1.0, torch.Generator().manual_seed(3)) == 1
    def draw(seed):
        generator = torch.Generator().manual_seed(seed)
        return [chat.sample_token(logits, 0.7, 3, 0.9, generator) for _ in range(12)]
    assert draw(42) == draw(42)
    assert 2 not in draw(42)
    torch.testing.assert_close(logits, before)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_sampling_rejects_nonfinite_unmasked_scores(bad):
    with pytest.raises((ValueError, RuntimeError)):
        chat.sample_token(torch.tensor([bad, 0.0]), 1.0, 0, 1.0, torch.Generator())


def test_capture_is_model_only_and_survives_atomic_source_replacement(fixture_dir, monkeypatch):
    source, state = source_checkpoint(fixture_dir)
    replacement = fixture_dir / 'replacement.pt'
    changed = dict(state, step=8, tokens_seen=128)
    torch.save(changed, replacement)
    original_load = torch.load
    replaced = False
    def replace_during_load(path, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            os.replace(replacement, source)
            replaced = True
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(torch, 'load', replace_during_load)
    snapshot = chat.capture_latest(source, TOKENIZER, fixture_dir / 'snapshots')
    monkeypatch.setattr(torch, 'load', original_load)
    payload = torch.load(snapshot, map_location='cpu', weights_only=False)
    assert 'optimizer' not in payload
    assert torch.load(source, map_location='cpu', weights_only=False)['step'] == 8
    again = torch.load(snapshot, map_location='cpu', weights_only=False)
    assert payload['step'] == again['step'] == 7
    for key in state['model']:
        torch.testing.assert_close(again['model'][key], state['model'][key], rtol=0, atol=0)
    model, sp, metadata = chat.load_model(snapshot, TOKENIZER)
    assert not model.training and sp.vocab_size() == model.cfg.vocab_size
    assert next(model.parameters()).device.type == 'cpu'
    assert all(buffer.device.type == 'cpu' for buffer in model.buffers())
    tokens = torch.tensor([[3, 4, 5, 6]])
    caches = model.init_generation_cache(1, torch.device('cpu'), torch.float32)
    stepped = torch.cat([model.step(tokens[:, i:i+1], caches) for i in range(4)], dim=1)
    torch.testing.assert_close(stepped, model(tokens), rtol=2e-5, atol=2e-6)


def test_cli_snapshot_accepts_training_checkpoint_without_sidecar(fixture_dir):
    source, _ = source_checkpoint(fixture_dir)
    source_hash = chat.sha256(source)
    snapshots = fixture_dir / 'snapshots'
    run = subprocess.run([
        sys.executable, '-B', str(ROOT / 'chat.py'), '--snapshot', str(source),
        '--snapshot-root', str(snapshots), '--threads', '1', '--prompt', 'Hello',
        '--max-new-tokens', '2', '--max-context', '32',
        '--repetition-penalty', '1.2', '--no-repeat-ngram-size', '4',
    ], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert 'CPU FP32 | step 7 | 112 training tokens' in run.stdout
    assert 'Repetition protection: penalty=1.2, no-repeat-ngram-size=4' in run.stdout
    assert 'Model: ' in run.stdout and 'Traceback' not in run.stderr
    exported = list(snapshots.glob('*/model.pt'))
    assert len(exported) == 1 and exported[0] != source
    proof = json.loads(exported[0].with_name('snapshot.json').read_text())
    assert proof['source']['path'] == str(source.resolve())
    assert proof['snapshot_sha256'] == chat.sha256(exported[0])
    assert chat.sha256(source) == source_hash
    assert not source.with_name('snapshot.json').exists()
    assert 'optimizer' not in torch.load(exported[0], map_location='cpu', weights_only=False)


def test_resolve_reuses_sealed_snapshot_without_export(fixture_dir, monkeypatch):
    source, _ = source_checkpoint(fixture_dir)
    snapshot = chat.capture_latest(source, TOKENIZER, fixture_dir / 'snapshots')
    def unexpected_capture(*args):
        pytest.fail('A sealed snapshot must be reused, not re-exported')
    monkeypatch.setattr(chat, 'capture_latest', unexpected_capture)
    assert chat.resolve_snapshot(snapshot, TOKENIZER, fixture_dir / 'unused') == snapshot
    assert not (fixture_dir / 'unused').exists()


def test_snapshot_missing_metadata_is_not_resealed(fixture_dir):
    source, _ = source_checkpoint(fixture_dir)
    snapshot = chat.capture_latest(source, TOKENIZER, fixture_dir / 'snapshots')
    snapshot.with_name('snapshot.json').unlink()
    with pytest.raises(ValueError, match='snapshot.json'):
        chat.resolve_snapshot(snapshot, TOKENIZER, fixture_dir / 'unused')
    with pytest.raises(ValueError, match='Missing snapshot metadata'):
        chat.load_model(snapshot, TOKENIZER)
    assert not (fixture_dir / 'unused').exists()


@pytest.mark.parametrize('metadata', ['{', '[]', '{}', '{"snapshot_sha256": "wrong"}'])
def test_snapshot_bad_metadata_is_not_bypassed(fixture_dir, metadata):
    source, _ = source_checkpoint(fixture_dir)
    snapshot = chat.capture_latest(source, TOKENIZER, fixture_dir / 'snapshots')
    snapshot.with_name('snapshot.json').write_text(metadata)
    resolved = chat.resolve_snapshot(snapshot, TOKENIZER, fixture_dir / 'unused')
    with pytest.raises(ValueError):
        chat.load_model(resolved, TOKENIZER)
    assert not (fixture_dir / 'unused').exists()


def test_missing_snapshot_path_is_a_clear_cli_error(fixture_dir):
    missing = fixture_dir / 'missing.pt'
    run = subprocess.run([sys.executable, '-B', str(ROOT / 'chat.py'),
                          '--snapshot', str(missing)],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 2
    assert 'Checkpoint file does not exist' in run.stderr and str(missing) in run.stderr
    assert 'Traceback' not in run.stderr


def test_tokenizer_hash_mismatch_is_rejected(fixture_dir):
    source, state = source_checkpoint(fixture_dir)
    state['signature']['data']['signature']['tokenizer_sha256'] = '0' * 64
    torch.save(state, source)
    with pytest.raises((ValueError, RuntimeError), match='[Tt]okenizer'):
        snapshot = chat.capture_latest(source, TOKENIZER, fixture_dir / 'snapshots')
        chat.load_model(snapshot, TOKENIZER)


def test_model_load_is_strict(fixture_dir):
    source, state = source_checkpoint(fixture_dir)
    del state['model']['depth_bias']
    torch.save(state, source)
    with pytest.raises((ValueError, RuntimeError)):
        snapshot = chat.capture_latest(source, TOKENIZER, fixture_dir / 'snapshots')
        chat.load_model(snapshot, TOKENIZER)


def test_generation_stops_at_eos_without_yielding_it(fixture_dir, monkeypatch):
    source, _ = source_checkpoint(fixture_dir)
    snapshot = chat.capture_latest(source, TOKENIZER, fixture_dir / 'snapshots')
    model, sp, _ = chat.load_model(snapshot, TOKENIZER)
    logits = torch.full((1, 1, sp.vocab_size()), -100.0)
    logits[0, 0, sp.eos_id()] = 100.0
    monkeypatch.setattr(model, 'step', lambda tokens, caches: logits)
    output = list(chat.generate(model, sp, 'Hello', max_new_tokens=5,
                               temperature=0, top_k=0, top_p=1, seed=1, max_context=32))
    assert output == []


@pytest.mark.parametrize('scores', [[3.0, 2.9], [-3.0, -3.4]])
@pytest.mark.parametrize('temperature', [0.0, 0.8])
def test_repetition_penalty_lowers_both_signs_before_top_k(scores, temperature):
    logits = torch.tensor(scores)
    before = logits.clone()
    result = chat.sample_token(logits, temperature, 1, 1.0, torch.Generator(),
                              previous_tokens=[0, 0], repetition_penalty=1.2)
    assert result == 1
    torch.testing.assert_close(logits, before)


def test_repetition_penalty_applies_once_per_distinct_token():
    assert chat.sample_token(torch.tensor([3.0, 2.4]), 0, 0, 1, torch.Generator(),
                             previous_tokens=[0, 0, 0], repetition_penalty=1.2) == 0


@pytest.mark.parametrize('temperature', [0.0, 0.8])
def test_ngram_mask_applies_before_greedy_or_top_k_and_handles_exhaustion(temperature):
    logits = torch.tensor([100.0, 1.0])
    before = logits.clone()
    assert chat.sample_token(logits, temperature, 1, 0.9, torch.Generator(),
                             banned_tokens={0}) == 1
    assert chat.sample_token(logits, temperature, 1, 0.9, torch.Generator(),
                             banned_tokens={0, 1}) is None
    torch.testing.assert_close(logits, before)


@pytest.mark.parametrize('tokens,size,expected', [
    ([1, 2, 3, 4, 1, 2, 3], 4, {4}),
    ([1, 2, 3, 4, 1, 2], 4, set()),
    ([1, 1, 1], 2, {1}),
    ([1, 2, 3], 1, {1, 2, 3}),
    ([1, 1], 0, set()),
    ([], 4, set()),
    ([1, 2], 4, set()),
    ([1, 2, 3, 1, 2, 4, 1, 2], 3, {3, 4}),
])
def test_ngram_blocking_includes_overlapping_and_multiple_matches(tokens, size, expected):
    assert chat.banned_ngram_tokens(tokens, size) == expected


class LoopTokenizer:
    def encode(self, prompt, out_type):
        # Repeated phrases in the prompt must still be usable in an answer.
        return [3, 4, 5, 6, 3, 4, 5, 6]

    def eos_id(self):
        return 2


class LoopModel:
    def init_generation_cache(self, *args):
        return []

    def step(self, token, caches):
        next_id = {3: 4, 4: 5, 5: 6, 6: 3, 7: 2}[int(token.item())]
        logits = torch.full((1, 1, 8), -100.0)
        logits[0, 0, 7] = 1.0
        logits[0, 0, next_id] = 10.0
        return logits


@pytest.mark.parametrize('temperature', [0.0, 0.8])
def test_generation_breaks_a_phrase_loop_preserves_prompt_and_resets_each_reply(temperature):
    model, sp = LoopModel(), LoopTokenizer()
    settings = dict(max_new_tokens=12, max_context=32, temperature=temperature, top_k=1)
    unguarded = list(chat.generate(model, sp, 'prompt', repetition_penalty=1,
                                  no_repeat_ngram_size=0, **settings))
    assert unguarded == [3, 4, 5, 6] * 3
    protected = list(chat.generate(model, sp, 'prompt', **settings))
    assert protected == [3, 4, 5, 6, 3, 4, 5, 7]
    assert list(chat.generate(model, sp, 'prompt', **settings)) == protected
    grams = [tuple(protected[i:i+4]) for i in range(len(protected) - 3)]
    assert len(grams) == len(set(grams))


def test_sampling_with_repetition_protection_is_seeded_without_changing_global_rng():
    logits = torch.tensor([3.0, 2.9, 2.8, 2.7])
    state = torch.random.get_rng_state().clone()
    def draw():
        generator = torch.Generator().manual_seed(42)
        return [chat.sample_token(logits, 0.8, 3, 0.9, generator,
                                  previous_tokens=[0], repetition_penalty=1.2,
                                  banned_tokens={3}) for _ in range(20)]
    assert draw() == draw()
    assert 3 not in draw()
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize('settings', [
    {'repetition_penalty': 0.9}, {'repetition_penalty': float('nan')},
    {'repetition_penalty': float('inf')}, {'no_repeat_ngram_size': -1},
    {'no_repeat_ngram_size': 1.5},
])
def test_generation_rejects_invalid_repetition_settings_before_model_use(settings):
    with pytest.raises(ValueError, match='repetition-penalty|no-repeat-ngram-size'):
        list(chat.generate(None, None, 'prompt', **settings))
