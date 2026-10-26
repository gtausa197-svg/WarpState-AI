#!/usr/bin/env python3
"""Chat with a sealed SFT/DPO export using its original training serialization.

FP32 master parameters remain unchanged. BF16 compute views and bounded chunk
caches reproduce parallel-memory inference without changing training sources.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE = PACKAGE.parent
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

import torch
import torch.nn.functional as F
from torch.func import functional_call

from chat import load_model, sample_token
from ops.posttrain_data import RENDER_FORMAT, ROLES, renderer_sha256
from posttrain import atomic_json, digest
from warpstate.model import rotate_half

DEFAULT_SNAPSHOT = PACKAGE / 'work/chat/dpo-final/model.pt'


def encode_prompt(messages, tokenizer):
    """Match segmented SFT tokens, ending immediately after an assistant header."""
    if not isinstance(messages, list) or not messages:
        raise ValueError('Prompt requires nonempty messages')
    if not isinstance(messages[-1], dict) or messages[-1].get('role') not in ('user', 'tool'):
        raise ValueError('The conditioning conversation must end in a user/tool message')
    vocab, eos, bos = tokenizer.vocab_size(), tokenizer.eos_id(), tokenizer.bos_id()
    if not 0 <= eos < vocab:
        raise ValueError('Tokenizer requires a real EOS token')
    ids = [bos] if 0 <= bos < vocab else []

    def append(text):
        encoded = tokenizer.encode(text, out_type=int)
        if any(type(token) is not int or not 0 <= token < vocab for token in encoded):
            raise ValueError('Tokenizer returned invalid token IDs')
        ids.extend(encoded)

    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ROLES:
            raise ValueError('Unsupported prompt role')
        if not isinstance(message.get('content'), str) or not message['content'].strip():
            raise ValueError('Messages require nonempty text content')
        if message.get('tool_calls'):
            raise ValueError('Tool calls must be serialized into message content')
        append(f"<|{message['role']}|>\n")
        append(message['content'])
        if message['role'] == 'assistant':
            ids.append(eos)
        append('\n')
    append('<|assistant|>\n')
    return ids


@dataclass
class DecodeCache:
    keys: torch.Tensor
    values: torch.Tensor
    memory_values: torch.Tensor
    fast: torch.Tensor
    slow: torch.Tensor
    writes: list[torch.Tensor] = field(default_factory=list)
    position: int = 0


def init_caches(model, device, dtype):
    caches = []
    for depth in range(model.cfg.logical_depth):
        core = model.cores[depth % model.cfg.n_cores]
        state_shape = (1, core.h, core.hd, core.hd)
        kv_shape = (1, core.h, core.chunk, core.hd)
        caches.append(DecodeCache(
            keys=torch.zeros(kv_shape, device=device, dtype=dtype),
            values=torch.zeros(kv_shape, device=device, dtype=dtype),
            memory_values=torch.zeros(kv_shape, device=device, dtype=dtype),
            fast=torch.zeros(state_shape, device=device, dtype=dtype),
            slow=torch.zeros(state_shape, device=device, dtype=dtype)))
    return caches


def _next_states(core, cache):
    # Form the chunk write in one BF16 GEMM, as forward() does. Accumulating
    # separately rounded token outer products would alter the recurrent memory.
    write = (torch.tanh(cache.keys).transpose(-2, -1) @ torch.tanh(cache.memory_values)) / core.chunk
    if core.cfg.parallel_memory:
        cache.writes.append(write)
        values = torch.stack(cache.writes, dim=2).flatten(-2)  # B,H,N,Dh²
        lag = torch.arange(len(cache.writes) - 1, -1, -1, device=write.device)

        def state(logit):
            decay = logit.float().sigmoid()[:, None, None]
            weights = ((1 - decay) * decay.pow(lag[None, None])).to(values.dtype)
            return (weights @ values).reshape_as(cache.fast)

        cache.fast = state(core.fast_decay_logit)
        cache.slow = state(core.slow_decay_logit)
    else:
        fast_decay = core.fast_decay_logit.sigmoid().to(write.dtype).view(1, core.h, 1, 1)
        slow_decay = core.slow_decay_logit.sigmoid().to(write.dtype).view(1, core.h, 1, 1)
        cache.fast = fast_decay * cache.fast + (1 - fast_decay) * write
        cache.slow = slow_decay * cache.slow + (1 - slow_decay) * write
    cache.position = 0


def _core_step(core, x, cache):
    q, k, v, g, u = (value.permute(0, 2, 1, 3) for value in core._split(core.norm1(x)))
    pos = cache.position
    cos = core.rope_cos[pos:pos + 1].to(q.dtype).view(1, 1, 1, core.hd)
    sin = core.rope_sin[pos:pos + 1].to(q.dtype).view(1, 1, 1, core.hd)
    q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
    cache.keys[:, :, pos:pos + 1].copy_(k)
    cache.values[:, :, pos:pos + 1].copy_(v)
    cache.memory_values[:, :, pos:pos + 1].copy_(u)
    local = F.scaled_dot_product_attention(q, cache.keys[:, :, :pos + 1],
        cache.values[:, :, :pos + 1], is_causal=False, dropout_p=0.0)
    mix = core.memory_mix_logit.sigmoid().to(q.dtype).view(1, core.h, 1, 1)
    if core.cfg.parallel_memory:
        memory = (q @ (mix * cache.fast + (1 - mix) * cache.slow)) / math.sqrt(core.hd)
    else:
        memory = mix * ((q @ cache.fast) / math.sqrt(core.hd))
        memory = memory + (1 - mix) * ((q @ cache.slow) / math.sqrt(core.hd))
    gate = torch.sigmoid(g)
    y = (gate * local + (1 - gate) * memory).permute(0, 2, 1, 3).reshape(1, 1, core.cfg.d_model)
    x = x + core.out_proj(y)
    x = x + core.ffn(core.norm2(x))
    cache.position += 1
    if cache.position == core.chunk:
        _next_states(core, cache)
    return x


class _Step(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.wrapped = model

    def forward(self, token, caches):
        model = self.wrapped
        x = model.embed(token)
        for depth in range(model.cfg.logical_depth):
            x = x * (1 + model.depth_scale[depth]) + model.depth_bias[depth]
            x = _core_step(model.cores[depth % model.cfg.n_cores], x, caches[depth])
        return F.linear(model.final_norm(x), model.embed.weight)


class CachedDecoder:
    """Inference-only compute views, retaining the model's original FP32 masters."""
    def __init__(self, model, *, dtype=torch.bfloat16):
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('Only BF16 compute or FP32 correctness fixtures are supported')
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError('Decode requires an eval-only frozen model')
        parameters = dict(model.named_parameters())
        if any(parameter.dtype != torch.float32 for parameter in parameters.values()):
            raise ValueError('Model masters must retain original FP32 values')
        self.device = next(model.parameters()).device
        if any(parameter.device != self.device for parameter in parameters.values()):
            raise ValueError('The model must occupy one device')
        self.dtype, self.model, self.wrapper = dtype, model, _Step(model)
        # Decay/mix sigmoids must use original FP32 logits before rounding.
        self.views = {'wrapped.' + name: parameter.to(dtype)
            for name, parameter in parameters.items()
            if not name.endswith(('fast_decay_logit', 'slow_decay_logit', 'memory_mix_logit'))}

    @torch.inference_mode()
    def step(self, token, caches):
        if token.shape != (1, 1) or token.device != self.device:
            raise ValueError('Decode expects one token on the model device')
        with torch.autocast(self.device.type, dtype=torch.bfloat16,
                            enabled=self.dtype == torch.bfloat16, cache_enabled=False):
            return functional_call(self.wrapper, self.views, (token, caches), strict=False)

    def new_caches(self):
        return init_caches(self.model, self.device, self.dtype)


@torch.inference_mode()
def generate(decoder, tokenizer, messages, *, max_new_tokens=256, max_context=2048,
             temperature=0.0, seed=42, on_token=None):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    if type(max_context) is not int or max_context < 2:
        raise ValueError('max_context must exceed one token')
    if not math.isfinite(temperature) or temperature < 0 or type(seed) is not int or seed < 0:
        raise ValueError('Temperature and seed must be nonnegative')
    ids = encode_prompt(messages, tokenizer)
    if len(ids) >= max_context:
        raise ValueError(f'Prompt has {len(ids)} tokens; context limit is {max_context}')
    limit = min(max_new_tokens, max_context - len(ids))
    caches = decoder.new_caches()
    token = torch.empty((1, 1), dtype=torch.long, device=decoder.device)
    generator = torch.Generator(device=decoder.device).manual_seed(seed)
    if decoder.device.type == 'cuda':
        torch.cuda.synchronize(decoder.device)
    begin = time.perf_counter()
    for value in ids:
        token.fill_(value)
        logits = decoder.step(token, caches)[0, -1]
    generated, first = [], None
    stop_reason = 'context_limit' if limit < max_new_tokens else 'length'
    sampled_tokens = 0
    for index in range(limit):
        value = sample_token(logits, temperature, 0, 1.0, generator)
        sampled_tokens += 1
        if first is None:
            first = time.perf_counter() - begin
        if value == tokenizer.eos_id():
            stop_reason = 'eos'
            break
        if value is None:
            raise RuntimeError('No legal continuation token')
        generated.append(value)
        if on_token is not None:
            on_token(tuple(generated))
        if index + 1 < limit:
            token.fill_(value)
            logits = decoder.step(token, caches)[0, -1]
    if decoder.device.type == 'cuda':
        torch.cuda.synchronize(decoder.device)
    elapsed = time.perf_counter() - begin
    return dict(response=tokenizer.decode(generated), token_ids=generated,
        prompt_tokens=len(ids), generated_tokens=len(generated), sampled_tokens=sampled_tokens,
        elapsed_seconds=elapsed, first_token_seconds=first,
        tokens_per_second=len(generated) / elapsed, stop_reason=stop_reason,
        temperature=temperature, seed=seed)


def read_prompts(path):
    value = json.loads(Path(path).read_text())
    rows = value.get('prompts') if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError('Prompt suite must be a nonempty list or an object with prompts')
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError(f'Invalid prompt suite row {index}')
        identity = row.get('id', str(index))
        if not isinstance(identity, (str, int)) or identity in seen:
            raise ValueError('Prompt suite IDs must be distinct strings or integers')
        seen.add(identity)
    return rows


def inside_workspace(value, *, must_exist=True):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(WORKSPACE) or path == WORKSPACE:
        raise ValueError('Every input and output must stay inside /workspace/WarpState')
    if must_exist and not path.is_file():
        raise ValueError(f'Missing input file: {path}')
    return path


def interactive_chat(reply, messages, *, seed):
    history = []
    print('Enter a message; Ctrl-C cancels input/reply, /reset clears history, /quit exits.', flush=True)
    while True:
        try:
            prompt = input('You: ').strip()
            if prompt in ('/quit', '/exit'):
                break
            if prompt == '/reset':
                history = []
                continue
            if not prompt:
                continue
            conversation = history + [{'role': 'user', 'content': prompt}] if history else messages(prompt)
            try:
                result = reply(conversation, seed=seed)
            except ValueError as error:
                print(f'{error}; use /reset to start a new conversation.', flush=True)
                continue
            if result['response'].strip():
                history = conversation + [{'role': 'assistant', 'content': result['response']}]
        except KeyboardInterrupt:
            # Discard this input/unfinished reply, retaining completed turns.
            # Generation's context managers unwind and restore parameter views.
            print('\nCancelled.', flush=True)
            continue
        except EOFError:
            print()
            break
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument('--tokenizer', type=Path, help='Defaults to tokenizer.model beside the snapshot')
    parser.add_argument('--device', default='cuda:0', help='cuda:0, another CUDA device, or cpu')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prompt')
    mode.add_argument('--prompts-json', type=Path)
    parser.add_argument('--output-json', type=Path, help='Atomic output report for a prompt or batch suite')
    parser.add_argument('--system', default='You are a helpful assistant.')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--max-context', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    try:
        snapshot = inside_workspace(args.snapshot)
        tokenizer = inside_workspace(args.tokenizer or snapshot.with_name('tokenizer.model'))
        rows = read_prompts(inside_workspace(args.prompts_json)) if args.prompts_json else None
        if args.prompt is not None:
            if not args.prompt.strip():
                raise ValueError('Prompt cannot be empty')
            rows = [dict(id='prompt', prompt=args.prompt)]
        output = inside_workspace(args.output_json, must_exist=False) if args.output_json else None
        if output is not None and (rows is None or output.exists()
                or not output.is_relative_to(PACKAGE / 'work')
                or output.is_relative_to(snapshot.parent)
                or output.is_relative_to(PACKAGE / 'work/archives')
                or output == tokenizer
                or (args.prompts_json and output == args.prompts_json.resolve())):
            raise ValueError('Select a new batch report under work/, separate from all model and input files')
        device = torch.device(args.device)
        if device.type not in ('cuda', 'cpu'):
            raise ValueError('Device must be CUDA or CPU')
        if args.threads < 1 or not 0 < args.max_new_tokens < args.max_context <= 2048:
            raise ValueError('Require threads >= 1 and 0 < max_new_tokens < max_context <= 2048')
        if not math.isfinite(args.temperature) or args.temperature < 0 or args.seed < 0:
            raise ValueError('Temperature and seed must be nonnegative')
        if not isinstance(args.system, str):
            raise ValueError('System instruction must be text')
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            if not torch.cuda.is_bf16_supported():
                raise ValueError('Selected GPU must support BF16')
        torch.set_num_threads(args.threads)
        print(f'Loading sealed snapshot {snapshot}', flush=True)
        model, sp, metadata = load_model(snapshot, tokenizer)
        model.to(device=device, dtype=torch.float32)
        decoder = CachedDecoder(model)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f'{metadata.get("stage", "posttrain")} step {metadata["step"]:,} | '
          f'{model.num_parameters() / 1e9:.3f}B | {device} BF16 compute, FP32 master weights', flush=True)

    def messages(prompt, system):
        prefix = [{'role': 'system', 'content': system}] if system.strip() else []
        return prefix + [{'role': 'user', 'content': prompt}]

    def reply(conversation, *, seed):
        result = generate(decoder, sp, conversation, max_new_tokens=args.max_new_tokens,
            max_context=args.max_context, temperature=args.temperature, seed=seed)
        print(result['response'], flush=True)
        print(f'[{result["generated_tokens"]} tokens; {result["elapsed_seconds"]:.2f}s; '
              f'stop={result["stop_reason"]}]', flush=True)
        return result

    if rows is not None:
        report = dict(format='warpstate-posttrain-generation-v1', snapshot=str(snapshot),
            snapshot_sha256=json.loads(snapshot.with_name('snapshot.json').read_text())['snapshot_sha256'],
            metadata=metadata, tokenizer_sha256=digest(tokenizer), renderer_format=RENDER_FORMAT,
            renderer_sha256=renderer_sha256(), inference_sha256=digest(__file__),
            device=str(device), compute_dtype='bfloat16', parameter_dtype='float32',
            max_context=args.max_context, max_new_tokens=args.max_new_tokens,
            complete=False, results=[])
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(rows):
            identity = row.get('id', str(index))
            system = row.get('system', args.system)
            if not isinstance(system, str):
                raise ValueError('Prompt system override must be text')
            print(f'\n[{identity}] {row["prompt"]}', flush=True)
            result = reply(messages(row['prompt'], system), seed=args.seed + index)
            report['results'].append(dict(id=identity, prompt=row['prompt'], system=system, **result))
            if output:
                atomic_json(output, report)
        report['complete'] = True
        if output:
            atomic_json(output, report)
            print(f'Results saved: {output}', flush=True)
        return 0

    return interactive_chat(reply, lambda prompt: messages(prompt, args.system), seed=args.seed)


if __name__ == '__main__':
    raise SystemExit(main())
