"""Preference branches using the frozen SFT renderer, with response-only masks."""

from __future__ import annotations

import json
from pathlib import Path

from ops.posttrain_data import encode_messages, renderer_sha256

MASK_FORMAT = 'warpstate-final-assistant-response-v1'


def encode_pair(row, tokenizer, max_length):
    prompt = row.get('prompt')
    if not isinstance(prompt, list) or not prompt or prompt[-1].get('role') not in ('user', 'tool'):
        raise ValueError('DPO requires a nonempty shared prompt ending user/tool')
    result = {}
    for name in ('chosen', 'rejected'):
        response = row.get(name)
        if not isinstance(response, list) or len(response) != 1 or response[0].get('role') != 'assistant':
            raise ValueError('DPO requires exactly one final assistant response per branch')
        encoded = encode_messages(prompt + response, tokenizer, max_length=max_length)
        tail = encode_messages(response, tokenizer)
        bos = int(0 <= tokenizer.bos_id() < tokenizer.vocab_size())
        prefix_length = len(encoded['input_ids']) - len(tail['input_ids']) + bos
        if encoded['input_ids'][prefix_length:] != tail['input_ids'][bos:]:
            raise ValueError('Response segmentation differs from the frozen SFT renderer')
        encoded['labels'][:prefix_length] = [-100] * prefix_length
        if not any(value != -100 for value in encoded['labels'][1:]):
            raise ValueError('Preference branch has no response targets')
        result[name] = encoded
    if result['chosen']['input_ids'] == result['rejected']['input_ids']:
        raise ValueError('Chosen and rejected responses must differ')
    return result


def load_data(data_dir, tokenizer, tokenizer_hash, sequence_length):
    from posttrain import digest
    directory = Path(data_dir)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('format') != 'warpstate-posttrain-data-v1' or manifest.get('complete') is not True:
        raise ValueError('Posttraining data is not complete')
    if manifest.get('tokenizer_sha256') != tokenizer_hash:
        raise ValueError('Prepared data tokenizer hash differs')
    if manifest.get('renderer_sha256') != renderer_sha256():
        raise ValueError('Prepared data renderer hash differs')
    if manifest.get('max_length') != sequence_length:
        raise ValueError('Prepared data max_length differs from sequence_length')
    result = []
    for filename in ('dpo_train.jsonl', 'dpo_val.jsonl'):
        path = directory / filename
        info = manifest.get('outputs', {}).get(filename)
        if not info or not path.is_file() or digest(path) != info.get('sha256'):
            raise ValueError(f'Missing or changed prepared file: {filename}')
        records = []
        token_count = target_count = 0
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                row = json.loads(line)
                encoded = encode_pair(row, tokenizer, sequence_length)
                for branch in ('chosen', 'rejected'):
                    tokens = len(encoded[branch]['input_ids'])
                    targets = sum(value != -100 for value in encoded[branch]['labels'][1:])
                    if (row.get('tokens', {}).get(branch) != tokens or
                            row.get('supervised_tokens', {}).get(branch) != targets):
                        raise ValueError(f'{filename}:{number}: token metadata differs')
                    token_count += tokens
                    target_count += targets
                records.append(encoded)
                if number % 10000 == 0:
                    print(json.dumps(dict(event='data_validation', file=filename, rows=number)), flush=True)
        if not records or len(records) != info.get('rows'):
            raise ValueError(f'{filename}: empty data or row count mismatch')
        if token_count != info.get('tokens') or target_count != info.get('supervised_tokens'):
            raise ValueError(f'{filename}: manifest token totals differ')
        result.append(records)
    return *result, manifest, digest(manifest_path)


def collate_pairs(records, sequence_length, pad_id):
    """Adjacent chosen/rejected sequences; empty tail pairs have zero weight."""
    import torch
    from posttrain import collate
    branches = [record[branch] if record is not None else None
                for record in records for branch in ('chosen', 'rejected')]
    x, y = collate(branches, sequence_length, pad_id)
    return x, y, torch.tensor([record is not None for record in records], dtype=torch.bool)
