#!/usr/bin/env python3
"""Evaluate a sealed posttraining snapshot on the complete pinned SFT holdout."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from chat import load_model, sha256
from ops.loss_experiment import apply_loss_execution
from ops.posttrain_data import encode_messages, renderer_sha256
from ops.residual_experiment import apply_bf16_residuals
from posttrain import evaluate


def load_holdout(directory, tokenizer, tokenizer_hash):
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('format') != 'warpstate-posttrain-data-v1'
            or manifest.get('complete') is not True
            or manifest.get('tokenizer_sha256') != tokenizer_hash
            or manifest.get('renderer_sha256') != renderer_sha256()):
        raise ValueError('Completed data, tokenizer and renderer identities must match')
    info = manifest['outputs']['sft_val.jsonl']
    source = directory / 'sft_val.jsonl'
    if sha256(source) != info['sha256']:
        raise ValueError('SFT holdout checksum mismatch')
    records, tokens, targets = [], 0, 0
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('tools'):
                raise ValueError('Tools must be serialized into conversation messages')
            record = encode_messages(row['messages'], tokenizer, max_length=manifest['max_length'])
            count = sum(value != -100 for value in record['labels'][1:])
            if row['tokens'] != len(record['input_ids']) or row['supervised_tokens'] != count:
                raise ValueError('SFT holdout token counts differ from exact encoding')
            tokens += len(record['input_ids'])
            targets += count
            records.append(record)
    if (len(records), tokens, targets) != (info['rows'], info['tokens'], info['supervised_tokens']):
        raise ValueError('SFT holdout totals differ from manifest')
    return records, manifest, sha256(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'work/posttrain/data')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    snapshot, data, output = (path.resolve() for path in (args.snapshot, args.data_dir, args.output))
    if any(not path.is_relative_to(ROOT.parent) for path in (snapshot, data, output)):
        parser.error('All paths must stay inside /workspace/WarpState')
    if output.exists() or output.is_relative_to(snapshot.parent) or output.is_relative_to(data):
        parser.error('Select a new report outside the snapshot and dataset directories')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        parser.error('This full-size evaluation requires a CUDA GPU')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        parser.error('BF16 support required')
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    tokenizer = snapshot.with_name('tokenizer.model')
    model, sp, metadata = load_model(snapshot, tokenizer)
    records, manifest, data_hash = load_holdout(data, sp, metadata['tokenizer_sha256'])
    print(f'Validated {len(records):,} SFT holdout conversations; loading {device}', flush=True)
    model.to(device)
    apply_loss_execution(model, 'full')
    apply_bf16_residuals(model)
    settings = dict(sequence_length=manifest['max_length'], micro_batch=4, eval_batches=32)
    pad_id = sp.pad_id() if sp.pad_id() >= 0 else sp.eos_id()
    torch.cuda.reset_peak_memory_stats(device)
    evaluation_started = time.perf_counter()
    loss, targets = evaluate(model, records, settings, device, pad_id, full=True)
    torch.cuda.synchronize(device)
    report = dict(format='warpstate-sft-snapshot-evaluation-v1',
        timestamp_utc=datetime.now(timezone.utc).isoformat(), snapshot=str(snapshot),
        snapshot_sha256=sha256(snapshot), snapshot_metadata=metadata,
        data_sha256=data_hash, holdout_sha256=manifest['outputs']['sft_val.jsonl']['sha256'],
        renderer_sha256=renderer_sha256(), full_holdout=True, evaluated_samples=len(records),
        assistant_target_tokens=targets, val_loss=loss, val_perplexity=math.exp(loss),
        objective='assistant-token-mean', settings=settings,
        execution=dict(device=str(device), device_name=torch.cuda.get_device_name(device),
            torch_version=torch.__version__, compute_dtype='bfloat16', parameter_dtype='float32',
            bf16_residuals=True, compiled=False, deterministic_algorithms=True),
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
            (Path(__file__), ROOT / 'chat.py', ROOT / 'posttrain.py',
             ROOT / 'ops/loss_experiment.py', ROOT / 'ops/residual_experiment.py',
             ROOT / 'warpstate/model.py')},
        evaluation_seconds=time.perf_counter() - evaluation_started,
        total_seconds=time.perf_counter() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('evaluated_samples', 'assistant_target_tokens',
        'val_loss', 'val_perplexity', 'evaluation_seconds')}) + f'\nReport: {output}', flush=True)


if __name__ == '__main__':
    main()
