#!/usr/bin/env python3
"""Verify the final DPO archive and create an independent sealed chat copy."""

import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from ops.archive_dpo import verify_dpo_archive
from ops.dpo_handoff import initialize_copy


def main():
    archive = PACKAGE / 'work/archives/step-00076294-dpo-final'
    destination = PACKAGE / 'work/chat/dpo-final'
    metadata, _ = verify_dpo_archive(archive)
    model = initialize_copy(archive, destination, metadata)
    print(json.dumps(dict(snapshot=str(model), archive=str(archive),
        model_sha256=metadata['snapshot_sha256'], independent_copy=True), indent=2))


if __name__ == '__main__':
    main()
