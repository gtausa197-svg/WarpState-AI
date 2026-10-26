"""CPU tests for posttraining prompt serialization and cached inference math."""

import copy
import json

import pytest
import torch

from ops import posttrain_chat as chat
from ops.posttrain_data import encode_messages
from ops.residual_experiment import apply_bf16_residuals
from warpstate import WarpStateConfig, WarpStateLM


class SegmentTokenizer:
    def vocab_size(self):
        return 256

    def eos_id(self):
        return 2

    def bos_id(self):
        return 1

    def encode(self, text, out_type=int):
        # Segment marker makes concat-then-tokenize observably different.
        return [3] + [ord(character) + 10 for character in text]

    def decode(self, tokens):
        return ','.join(str(value) for value in tokens)


def conversation():
    return [{'role': 'system', 'content': 'Be concise.'},
            {'role': 'user', 'content': 'Say hello.'},
            {'role': 'assistant', 'content': 'Hello.'},
            {'role': 'user', 'content': 'Two plus two?'}]


@pytest.mark.parametrize('original_tokenizer', [False, True])
def test_prompt_is_exact_training_prefix_with_real_eos_only_for_history(original_tokenizer):
    if original_tokenizer:
        import sentencepiece as spm
        tokenizer = spm.SentencePieceProcessor(model_file=str(chat.PACKAGE / 'tokenizer.model'))
    else:
        tokenizer = SegmentTokenizer()
    messages = conversation()
    prompt = chat.encode_prompt(messages, tokenizer)
    encoded = encode_messages(messages + [{'role': 'assistant', 'content': 'Four.'}], tokenizer)
    answer_tokens = tokenizer.encode('Four.', out_type=int)
    assert encoded['input_ids'][:len(prompt)] == prompt
    assert encoded['input_ids'][len(prompt):len(prompt) + len(answer_tokens)] == answer_tokens
    assert encoded['labels'][len(prompt)] != -100
    assert prompt[0] == tokenizer.bos_id()
    assert prompt.count(tokenizer.eos_id()) == 1
    assert prompt[-len(tokenizer.encode('<|assistant|>\n', out_type=int)):] == tokenizer.encode('<|assistant|>\n', out_type=int)


@pytest.mark.parametrize('messages', [[], ['bad'], [{'role': 'assistant', 'content': 'bad'}],
    [{'role': 'user', 'content': ''}], [{'role': 'user', 'content': 'call', 'tool_calls': [1]}]])
def test_prompt_rejects_invalid_conversations(messages):
    with pytest.raises(ValueError):
        chat.encode_prompt(messages, SegmentTokenizer())


def tiny_model(parallel=True):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        cfg = WarpStateConfig(vocab_size=64, d_model=32, n_heads=4, n_cores=2,
            logical_depth=3, ffn_hidden=64, chunk_size=4, dropout=0.0,
            parallel_memory=parallel, gradient_checkpointing=False)
        model = WarpStateLM(cfg).eval().requires_grad_(False)
        model.embed.weight.normal_(std=0.2)
        model.depth_scale.normal_(std=0.05)
        model.depth_bias.normal_(std=0.01)
        for core in model.cores:
            core.in_proj.weight.normal_(std=0.15)
            core.out_proj.weight.normal_(std=0.12)
    return model


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cached_logits_match_full_forward_across_seven_chunks(parallel, dtype):
    torch.set_num_threads(1)
    model = tiny_model(parallel)
    original = {name: parameter.clone() for name, parameter in model.named_parameters()}
    full = copy.deepcopy(model)
    apply_bf16_residuals(full)
    decoder = chat.CachedDecoder(model, dtype=dtype)
    tokens = torch.arange(28).reshape(1, -1) % model.cfg.vocab_size
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.bfloat16,
            enabled=dtype == torch.bfloat16, cache_enabled=False):
        expected = full(tokens)
    caches = decoder.new_caches()
    actual = torch.cat([decoder.step(tokens[:, index:index + 1], caches)
                        for index in range(tokens.shape[1])], dim=1)
    # Shape-dependent GEMM/SDPA rounding is expected in BF16. FP32 verifies the
    # recurrence itself tightly, including shared cores with separate caches.
    tolerance = dict(rtol=0.025, atol=0.025) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(actual, expected, **tolerance)
    assert all(cache.position == 0 and cache.fast.dtype == dtype for cache in caches)
    assert all(len(cache.writes) == (7 if parallel else 0) for cache in caches)
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32 and torch.equal(parameter, original[name])


class FakeDecoder:
    device = torch.device('cpu')

    def __init__(self, prefix_length, sequence):
        self.prefix_length, self.sequence, self.calls = prefix_length, sequence, 0

    def new_caches(self):
        return []

    def step(self, token, caches):
        self.calls += 1
        index = max(0, self.calls - self.prefix_length)
        value = self.sequence[min(index, len(self.sequence) - 1)]
        logits = torch.full((1, 1, 256), -100.)
        logits[0, 0, value] = 100.
        return logits


def test_generation_stops_on_eos_and_excludes_eos_from_response():
    tokenizer = SegmentTokenizer()
    messages = [{'role': 'user', 'content': 'Question'}]
    prefix = chat.encode_prompt(messages, tokenizer)
    decoder = FakeDecoder(len(prefix), [7, 8, tokenizer.eos_id()])
    emitted = []
    result = chat.generate(decoder, tokenizer, messages, max_new_tokens=10,
        on_token=lambda tokens: emitted.append((tokens, decoder.calls)))
    assert result['token_ids'] == [7, 8] and result['response'] == '7,8'
    assert result['generated_tokens'] == 2 and result['sampled_tokens'] == 3
    assert result['stop_reason'] == 'eos'
    assert decoder.calls == len(prefix) + 2
    assert emitted == [((7,), len(prefix)), ((7, 8), len(prefix) + 1)]
    assert result['elapsed_seconds'] >= result['first_token_seconds'] >= 0


def test_generation_preserves_prompt_and_respects_context_limit():
    tokenizer = SegmentTokenizer()
    messages = [{'role': 'user', 'content': 'Question'}]
    prefix = chat.encode_prompt(messages, tokenizer)
    decoder = FakeDecoder(len(prefix), [7])
    result = chat.generate(decoder, tokenizer, messages, max_new_tokens=10, max_context=len(prefix) + 2)
    assert result['prompt_tokens'] == len(prefix) and result['generated_tokens'] == 2
    assert result['stop_reason'] == 'context_limit'
    with pytest.raises(ValueError, match='context limit'):
        chat.generate(decoder, tokenizer, messages, max_context=len(prefix))


def test_suite_preserves_ids_and_ignores_rubric_fields(tmp_path):
    path = tmp_path / 'prompts.json'
    rows = [{'id': 'math', 'prompt': 'Two plus two?', 'expected_exact': '4', 'rubric': 'Correct arithmetic'}]
    path.write_text(json.dumps(rows))
    assert chat.read_prompts(path) == rows
    path.write_text(json.dumps({'prompts': rows}))
    assert chat.read_prompts(path) == rows
    path.write_text(json.dumps(rows * 2))
    with pytest.raises(ValueError, match='distinct'):
        chat.read_prompts(path)


def test_ctrl_c_cancels_typing_and_generation_preserving_completed_history(monkeypatch, capsys):
    inputs = iter(['first', KeyboardInterrupt(), 'cancel reply', 'next', '/quit'])
    calls = []

    def read(prompt):
        value = next(inputs)
        if isinstance(value, BaseException):
            raise value
        return value

    def reply(messages, *, seed):
        calls.append(copy.deepcopy(messages))
        if messages[-1]['content'] == 'cancel reply':
            raise KeyboardInterrupt()
        return {'response': 'completed'}

    monkeypatch.setattr('builtins.input', read)
    assert chat.interactive_chat(reply, lambda prompt: [{'role': 'user', 'content': prompt}], seed=42) == 0
    assert calls[-1] == [
        {'role': 'user', 'content': 'first'},
        {'role': 'assistant', 'content': 'completed'},
        {'role': 'user', 'content': 'next'}]
    assert capsys.readouterr().out.count('Cancelled.') == 2


def test_ctrl_c_unwinds_cached_parameter_views_and_allows_next_generation(monkeypatch):
    model = tiny_model()
    original = dict(model.named_parameters())
    decoder = chat.CachedDecoder(model)
    step = chat._core_step

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(chat, '_core_step', interrupt)
    with pytest.raises(KeyboardInterrupt):
        decoder.step(torch.tensor([[4]]), decoder.new_caches())
    assert all(parameter is original[name] for name, parameter in model.named_parameters())
    monkeypatch.setattr(chat, '_core_step', step)
    assert torch.isfinite(decoder.step(torch.tensor([[4]]), decoder.new_caches())).all()
