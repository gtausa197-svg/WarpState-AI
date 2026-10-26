"""Frozen, vocabulary-preserving SFT serialization shared by preparation/training.

Segments are encoded separately: the textual role header, content, and separator.
This is intentional (SentencePiece boundary normalization is not compositional).
Inference must use the same segments. Assistant content and the real EOS token
are targets; role headers, system/user/tool content and separators are masked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


RENDER_FORMAT = 'warpstate-segmented-roles-v1'
IGNORE_INDEX = -100
ROLES = frozenset({'system', 'user', 'assistant', 'tool'})


def renderer_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def encode_messages(messages, tokenizer, max_length=None):
    """Return unshifted token IDs/labels, never truncating a conversation.

Training shifts with ``x = input_ids[:-1], y = labels[1:]``. A valid optional
SentencePiece BOS begins the conversation; only assistant ends receive EOS.
``max_length`` bounds the entire serialized record including BOS/EOS.
"""
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    if max_length is not None and (type(max_length) is not int or max_length < 2):
        raise ValueError('max_length must be an integer >= 2')
    eos = tokenizer.eos_id()
    bos = tokenizer.bos_id()
    vocab = tokenizer.vocab_size()
    if not 0 <= eos < vocab:
        raise ValueError('Tokenizer must define a real EOS token')
    ids, labels = [], []
    if 0 <= bos < vocab:
        ids.append(bos)
        labels.append(IGNORE_INDEX)

    def append(text, supervised=False):
        tokens = tokenizer.encode(text, out_type=int)
        if any(type(token) is not int or not 0 <= token < vocab for token in tokens):
            raise ValueError('Tokenizer returned invalid token IDs')
        ids.extend(tokens)
        labels.extend(tokens if supervised else [IGNORE_INDEX] * len(tokens))

    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ROLES:
            raise ValueError('Unsupported conversation role')
        content = message.get('content')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Each message requires nonempty text content')
        if message.get('tool_calls'):
            raise ValueError('Structured tool calls must be serialized into content by preparation')
        assistant = message['role'] == 'assistant'
        append(f"<|{message['role']}|>\n")
        append(content, assistant)
        if assistant:
            ids.append(eos)
            labels.append(eos)
        append('\n')
    if max_length is not None and len(ids) > max_length:
        raise ValueError(f'Conversation has {len(ids)} tokens, exceeds max_length={max_length}')
    if len(ids) < 2 or not any(label != IGNORE_INDEX for label in labels[1:]):
        raise ValueError('Conversation has no supervised next-token targets')
    return {'input_ids': ids, 'labels': labels}
