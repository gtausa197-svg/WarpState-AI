#!/usr/bin/env python3
"""CPU-only chat/completion probe of an independent WarpState weight snapshot.

Only load trusted local torch checkpoints. This does not stop training, touch
CUDA, load Adam into a model, or write to the training checkpoint directory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time

import sentencepiece as spm
import torch

from warpstate import WarpStateConfig, WarpStateLM

ROOT = Path(__file__).resolve().parent
FORMAT = 'warpstate-chat-v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def capture_latest(source, tokenizer, snapshot_root):
    """Pin one atomically published inode, export only its original FP32 model.

    latest.pt is published by rename, never updated in place. An open descriptor
    remains bound to that complete version even if the trainer publishes again.
    No hardlink to the production checkpoint is created.
    """
    source, tokenizer, snapshot_root = map(Path, (source, tokenizer, snapshot_root))
    if source.suffix != '.pt' or source.is_symlink():
        raise ValueError('Select a completed regular .pt checkpoint, not a temporary file')
    tokenizer_sha = sha256(tokenizer)
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('Checkpoint must be a regular file')
        checkpoint = torch.load(f'/proc/self/fd/{fd}', map_location='cpu',
                                weights_only=False, mmap=True)
        if checkpoint.get('format') == FORMAT:
            raise ValueError(
                f'Chat snapshot requires its original snapshot.json beside {source}. '
                'Restore that metadata or export a fresh snapshot from a training checkpoint.')
        if checkpoint.get('format') != 'warpstate-server-v1':
            raise ValueError('Expected a full warpstate-server-v1 training checkpoint')
        recorded_tokenizer = checkpoint['signature']['data']['signature']['tokenizer_sha256']
        if recorded_tokenizer != tokenizer_sha:
            raise ValueError('Tokenizer SHA256 differs from training checkpoint')
        snapshot_root.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix=f'step-{checkpoint["step"]}-', dir=snapshot_root))
        destination = folder / 'model.pt'
        temporary = folder / 'model.pt.tmp'
        metadata = dict(format=FORMAT, inference_only=True, config=checkpoint['config'],
                        step=checkpoint['step'], tokens_seen=checkpoint['tokens_seen'],
                        val_loss=checkpoint.get('val_loss'), tokenizer_sha256=tokenizer_sha,
                        captured_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        source=dict(path=str(source.resolve()), device=before.st_dev,
                                    inode=before.st_ino, size=before.st_size,
                                    mtime_ns=before.st_mtime_ns))
        print(f'Capturing completed step {metadata["step"]:,} / '
              f'{metadata["tokens_seen"]:,} tokens; model weights only...', flush=True)
        with temporary.open('xb') as stream:
            torch.save(dict(metadata, model=checkpoint['model']), stream)
            stream.flush()
            os.fsync(stream.fileno())
        after = os.fstat(fd)
        # Atomic unlink/rename may change ctime/nlink without changing contents.
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError('Pinned checkpoint contents changed; snapshot not published')
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
        metadata['snapshot_sha256'] = sha256(destination)
        metadata['snapshot_bytes'] = destination.stat().st_size
        with (folder / 'snapshot.json').open('x') as stream:
            json.dump(metadata, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination
    finally:
        os.close(fd)


def resolve_snapshot(source, tokenizer, snapshot_root):
    """Reuse a sealed export, or safely capture a full training checkpoint.

    Missing metadata is not permission to load unverified inference weights:
    capture_latest accepts only the embedded training-checkpoint format and
    pins its inode across atomic replacements by the trainer.
    """
    source = Path(source)
    if not source.is_file():
        raise ValueError(f'Checkpoint file does not exist or is not a regular file: {source}')
    if source.with_name('snapshot.json').exists():
        return source
    return capture_latest(source, tokenizer, snapshot_root)


def load_model(snapshot, tokenizer):
    snapshot, tokenizer = Path(snapshot), Path(tokenizer)
    proof_path = snapshot.with_name('snapshot.json')
    try:
        proof = json.loads(proof_path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(
            f'Missing snapshot metadata: {proof_path}. '
            'Use --checkpoint to export training weights, or restore the original metadata.') from exc
    if not isinstance(proof, dict) or not isinstance(proof.get('snapshot_sha256'), str):
        raise ValueError(f'Invalid snapshot metadata: {proof_path}')
    if proof['snapshot_sha256'] != sha256(snapshot):
        raise ValueError('Inference snapshot SHA256 mismatch')
    checkpoint = torch.load(snapshot, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint.get('format') != FORMAT or not checkpoint.get('inference_only'):
        raise ValueError('Use the independent inference snapshot, not live latest.pt')
    if checkpoint['tokenizer_sha256'] != sha256(tokenizer):
        raise ValueError('Tokenizer SHA256 mismatch')
    config = WarpStateConfig(**checkpoint['config'])
    tokenizer_model = spm.SentencePieceProcessor(model_file=str(tokenizer))
    if tokenizer_model.vocab_size() != config.vocab_size:
        raise ValueError('Tokenizer vocabulary differs from model')
    with torch.device('meta'):
        model = WarpStateLM(config)
    model.load_state_dict(checkpoint['model'], strict=True, assign=True)
    model.eval().requires_grad_(False)
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('This CPU probe requires the original FP32 model parameters')
    metadata = {key: value for key, value in checkpoint.items() if key != 'model'}
    del checkpoint
    gc.collect()
    return model, tokenizer_model, metadata


def banned_ngram_tokens(tokens, size):
    """Block completions of n-grams already present in this reply, not its prompt."""
    if size == 0 or len(tokens) + 1 < size:
        return set()
    prefix = tuple(tokens[-(size - 1):]) if size > 1 else ()
    return {tokens[start + size - 1] for start in range(len(tokens) - size + 1)
            if tuple(tokens[start:start + size - 1]) == prefix}


def sample_token(logits, temperature, top_k, top_p, generator, *,
                 previous_tokens=(), repetition_penalty=1.0, banned_tokens=()):
    logits = logits.float().clone()
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite generation logits')
    if not math.isfinite(temperature) or temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError('Invalid sampling settings')
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
        raise ValueError('Require a finite repetition-penalty >= 1')
    if previous_tokens and repetition_penalty != 1:
        seen = list(set(previous_tokens))
        scores = logits[seen]
        # Multiplying negative scores and dividing positive ones lowers both.
        logits[seen] = torch.where(scores < 0, scores * repetition_penalty,
                                   scores / repetition_penalty)
    if banned_tokens:
        logits[list(banned_tokens)] = -torch.inf
    if not torch.isfinite(logits).any():
        return None  # No legal continuation; never sample a masked token.
    if temperature == 0:
        return int(logits.argmax())
    values = logits / temperature
    if top_k:
        threshold = values.topk(min(top_k, values.numel())).values[-1]
        values = values.masked_fill(values < threshold, -torch.inf)
    ordered, indices = values.sort(descending=True)
    probability = ordered.softmax(dim=-1)
    # Keep the token crossing the cumulative threshold, including at least one.
    ordered = ordered.masked_fill(probability.cumsum(-1) - probability > top_p, -torch.inf)
    picked = torch.multinomial(ordered.softmax(dim=-1), 1, generator=generator)
    return int(indices[picked].item())


@torch.inference_mode()
def generate(model, sp, prompt, *, max_new_tokens=256, temperature=0.8,
             top_k=40, top_p=0.9, seed=1337, max_context=2048,
             repetition_penalty=1.2, no_repeat_ngram_size=4):
    if max_new_tokens < 1 or max_context <= max_new_tokens:
        raise ValueError('Require 0 < max-new-tokens < max-context')
    if not math.isfinite(repetition_penalty) or repetition_penalty < 1:
        raise ValueError('Require a finite repetition-penalty >= 1')
    if not isinstance(no_repeat_ngram_size, int) or no_repeat_ngram_size < 0:
        raise ValueError('Require an integer no-repeat-ngram-size >= 0')
    ids = sp.encode(prompt, out_type=int)
    if not ids:
        raise ValueError('Prompt contains no tokens')
    ids = ids[-(max_context - max_new_tokens):]
    caches = model.init_generation_cache(1, torch.device('cpu'), torch.float32)
    token = torch.empty((1, 1), dtype=torch.long)
    for value in ids:
        token.fill_(value)
        logits = model.step(token, caches)[0, -1]
    generator = torch.Generator(device='cpu').manual_seed(seed)
    generated = []
    for index in range(max_new_tokens):
        value = sample_token(logits, temperature, top_k, top_p, generator,
                             previous_tokens=generated, repetition_penalty=repetition_penalty,
                             banned_tokens=banned_ngram_tokens(generated, no_repeat_ngram_size))
        if value is None or value == sp.eos_id():
            break
        generated.append(value)
        yield value
        if index + 1 < max_new_tokens:
            token.fill_(value)
            logits = model.step(token, caches)[0, -1]


def reply(model, sp, prompt, args):
    ids, emitted = [], ''
    begin, first = time.perf_counter(), None
    for value in generate(model, sp, prompt, max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                          seed=args.seed, max_context=args.max_context,
                          repetition_penalty=args.repetition_penalty,
                          no_repeat_ngram_size=args.no_repeat_ngram_size):
        if first is None:
            first = time.perf_counter()
        ids.append(value)
        text = sp.decode(ids)
        # Delay incomplete UTF-8 byte-fallback characters until they are complete.
        safe = text.split('\ufffd', 1)[0]
        if safe.startswith(emitted):
            print(safe[len(emitted):], end='', flush=True)
            emitted = safe
    text = sp.decode(ids)
    if text.startswith(emitted):
        print(text[len(emitted):], end='', flush=True)
    elapsed = time.perf_counter() - begin
    ttft = '-' if first is None else f'{first - begin:.2f}s'
    print(f'\n[{len(ids)} tokens; {elapsed:.1f}s total; TTFT {ttft}; CPU FP32]', flush=True)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'work/run/latest.pt')
    parser.add_argument('--snapshot', type=Path,
                        help='Reuse a sealed model-only snapshot, or export a training checkpoint')
    parser.add_argument('--snapshot-root', type=Path, default=ROOT / 'work/chat')
    parser.add_argument('--tokenizer', type=Path, default=ROOT / 'tokenizer.model')
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('--prompt', help='One-shot probe instead of interactive input')
    parser.add_argument('--raw', action='store_true', help='Plain text completion instead of Q/A labels')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--max-context', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.8, help='0 = greedy')
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                        help='Lower scores of tokens already generated in this reply; 1 disables')
    parser.add_argument('--no-repeat-ngram-size', type=int, default=4,
                        help='Block repeated token sequences within a reply; 0 disables')
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    if args.threads < 1 or not 0 < args.max_new_tokens < args.max_context:
        parser.error('Require threads >= 1 and 0 < max-new-tokens < max-context')
    if not math.isfinite(args.temperature) or args.temperature < 0 or args.top_k < 0 or not 0 < args.top_p <= 1:
        parser.error('Invalid temperature / top-k / top-p')
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty < 1 or args.no_repeat_ngram_size < 0:
        parser.error('Require finite repetition-penalty >= 1 and no-repeat-ngram-size >= 0')
    if args.prompt is not None and not args.prompt.strip():
        parser.error('A one-shot prompt must not be empty')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    try:
        if args.snapshot is not None:
            snapshot = resolve_snapshot(args.snapshot, args.tokenizer, args.snapshot_root)
        else:
            snapshot = capture_latest(args.checkpoint, args.tokenizer, args.snapshot_root)
        print(f'Snapshot: {snapshot}', flush=True)
        if args.capture_only:
            return
        model, sp, metadata = load_model(snapshot, args.tokenizer)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'WarpState {model.num_parameters() / 1e9:.3f}B | CPU FP32 | '
          f'step {metadata["step"]:,} | {metadata["tokens_seen"]:,} training tokens', flush=True)
    print('Base pretrained model, not instruction-tuned. Q/A labels are only a probe.\n'
          'Commands: /new, /raw, /chat, /info, /help, /exit. Ctrl-C cancels a reply.\n'
          'Each turn replays up to the latest context tokens; long prompts take time on CPU.', flush=True)
    print(f'Repetition protection: penalty={args.repetition_penalty:g}, '
          f'no-repeat-ngram-size={args.no_repeat_ngram_size} (per reply).', flush=True)
    history = ''
    while True:
        try:
            user = args.prompt if args.prompt is not None else input('\nYou: ').strip()
            if user is None or (args.prompt is None and user in ('/exit', '/quit')):
                break
            if not user:
                continue
            if args.prompt is None and user in ('/new', '/raw', '/chat'):
                history = ''
                if user != '/new':
                    args.raw = user == '/raw'
                print('Context cleared. Mode: ' + ('raw completion' if args.raw else 'Q/A probe'))
                continue
            if args.prompt is None and user == '/info':
                print(json.dumps(metadata, indent=2))
                continue
            if args.prompt is None and user == '/help':
                print('/new clears context; /raw plain completion; /chat Q/A; /info snapshot; /exit quit')
                continue
            prompt = history + (user if args.raw else f'Question: {user}\nAnswer:')
            print('Model: ', end='', flush=True)
            answer = reply(model, sp, prompt, args)
            history = prompt + ('' if args.raw else ' ') + answer + '\n\n'
            if args.prompt is not None:
                break
        except EOFError:
            break
        except KeyboardInterrupt:
            print('\nReply cancelled; training was not affected.', flush=True)
            if args.prompt is not None:
                break


if __name__ == '__main__':
    main()
