# WarpState — assembled source bundle

This bundle combines the **three source archives you provided** with the newly designed README and SVG assets. Existing source files were preserved byte-for-byte. No Python implementation was rewritten. Python bytecode caches (`__pycache__`, `.pyc`) were omitted.

## Included

- `warpstate/model.py`, `warpstate/config.py`, `warpstate/__init__.py` — model implementation
- `run_40b.py` — pretraining
- `posttrain.py` — supervised fine-tuning
- `dpo.py` — preference optimization
- `chat.py` — chat inference/probe
- `ops/*.py` and `ops/*.sh` — execution and training operations
- `test_*.py` — tests
- `tokenizer.model` — supplied tokenizer
- `README.md` and `assets/` — updated GitHub presentation

## Not supplied in the uploaded archives

- Trained model checkpoints / weights
- Training datasets
- `configs/` directory, including `configs/warpstate_2b.json` referenced by supplied `PACKAGE_MANIFEST.json` and training documents
- Explicit license file

**Note:** This is a *complete assembly of the available source*, not a turnkey pretrained model. Before publishing, review all source and documentation, choose a license, restore valid configs, and independently validate environment-specific commands. Scripts with old server-specific paths may require adaptation. Do not upload private secrets or checkpoints.

Original archives:
- `drive-download-20261026T200043Z-1-001.zip`
- `warpstate-20261026T200055Z-1-001.zip`
- `ops-20261026T200120Z-1-001.zip`
