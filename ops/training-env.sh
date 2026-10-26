#!/usr/bin/env bash
# Source this file before installing dependencies, running tests, or training.
export TMPDIR=/workspace/WarpState/tmp
export XDG_CACHE_HOME=/workspace/WarpState/.cache
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export HF_HOME=/workspace/WarpState/rtx6000_40b/work/hf-cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export PYTHONDONTWRITEBYTECODE=1
export UV_PYTHON_DOWNLOADS=never
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=backend:native
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
