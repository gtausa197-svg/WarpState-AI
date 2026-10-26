"""Bounded BF16 matrix kernels; autograd and policy selection belong to callers.

Triton variants accumulate dot products in FP32 and return BF16, without split-K
atomics or reduced-precision operands. ``aten`` preserves the process's existing
BLAS settings. Persistent scheduling assigns multiple output tiles to each
resident program; ``persistent_tma`` also uses TensorDescriptor loads. These
kernels do not use architecture-specific warp-specialized execution.
"""

from collections.abc import Mapping
import json

import torch
from torch.library import wrap_triton
import triton
import triton.language as tl


ALGORITHMS = ("aten", "tiled", "persistent", "persistent_tma")
TILE_FIELDS = ("block_m", "block_n", "block_k", "num_warps", "num_stages")
DEFAULT_TILE = (128, 128, 32, 4, 3)
TUNING_TILES = (
    (64, 128, 64, 4, 3),
    DEFAULT_TILE,
    (128, 256, 32, 8, 3),
    (256, 128, 32, 8, 3),
    (128, 128, 64, 4, 2),
)


def normalize_tile(tile=None):
    """Normalize a five-integer tile tuple or named JSON object."""
    if tile is None:
        tile = DEFAULT_TILE
    if isinstance(tile, Mapping):
        if set(tile) != set(TILE_FIELDS):
            raise ValueError("Tile must contain exactly " + ", ".join(TILE_FIELDS))
        tile = tuple(tile[key] for key in TILE_FIELDS)
    if not isinstance(tile, (tuple, list)) or len(tile) != 5:
        raise ValueError("Tile must be (block_m, block_n, block_k, num_warps, num_stages)")
    if any(type(value) is not int for value in tile):
        raise ValueError("Tile values must be integers")
    bm, bn, bk, warps, stages = tile
    if any(value < 16 or value > 256 or value & (value - 1) for value in (bm, bn, bk)):
        raise ValueError("Matrix tile dimensions must be powers of two between 16 and 256")
    if warps not in (4, 8) or stages not in (1, 2, 3, 4):
        raise ValueError("Supported launch settings are 4/8 warps and 1–4 stages")
    if (bm + bn) * bk * 2 * stages > 96 * 1024:
        raise ValueError("BF16 operand pipeline exceeds the conservative 96 KiB shared-memory budget")
    return (bm, bn, bk, warps, stages)


def tile_dict(tile):
    return dict(zip(TILE_FIELDS, normalize_tile(tile), strict=True))


def matrix_signature(a, b):
    """Serializable dispatch signature; use outside captured execution."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected compatible two-dimensional matrix operands")
    if a.dtype != b.dtype:
        raise ValueError("Matrix operands must have matching dtypes")
    return {"m": a.shape[0], "n": b.shape[1], "k": a.shape[1],
            "a_stride": list(a.stride()), "b_stride": list(b.stride()),
            "dtype": str(a.dtype).removeprefix("torch.")}


def policy_key(a, b):
    return json.dumps(matrix_signature(a, b), sort_keys=True, separators=(",", ":"))


@triton.jit
def _tile_position(tile_id, tiles_m, tiles_n, GROUP_M: tl.constexpr):
    group_size = GROUP_M * tiles_n
    first_m = (tile_id // group_size) * GROUP_M
    actual_m = tl.minimum(tiles_m - first_m, GROUP_M)
    within = tile_id % group_size
    return first_m + within % actual_m, within // actual_m


@triton.jit
def _matmul_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   A0: tl.constexpr, A1: tl.constexpr, B0: tl.constexpr, B1: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   GROUP_M: tl.constexpr, PROGRAMS: tl.constexpr,
                   PERSISTENT: tl.constexpr):
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    first = tl.program_id(0)
    # A tiled launch has one iteration. Persistent programs advance through
    # multiple output tiles, keeping the launch bounded by the SM count.
    stop = tiles_m * tiles_n if PERSISTENT else first + 1
    step = PROGRAMS if PERSISTENT else 1
    for tile_id in range(first, stop, step):
        tile_m, tile_n = _tile_position(tile_id, tiles_m, tiles_n, GROUP_M)
        rows = tile_m * BM + tl.arange(0, BM)
        cols = tile_n * BN + tl.arange(0, BN)
        reduction = tl.arange(0, BK)
        accumulator = tl.full((BM, BN), 0, tl.float32)
        for block in range(tl.cdiv(K, BK)):
            kk = block * BK + reduction
            av = tl.load(A + rows[:, None] * A0 + kk[None, :] * A1,
                         (rows[:, None] < M) & (kk[None, :] < K), 0)
            bv = tl.load(B + kk[:, None] * B0 + cols[None, :] * B1,
                         (kk[:, None] < K) & (cols[None, :] < N), 0)
            accumulator = tl.dot(av, bv, accumulator, input_precision="ieee")
        tl.store(C + rows[:, None] * N + cols[None, :], accumulator.to(tl.bfloat16),
                 (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def _persistent_tma_kernel(A_DESC, B_DESC, C,
                           M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           PROGRAMS: tl.constexpr, A_ROW: tl.constexpr, B_ROW: tl.constexpr):
    tiles_m = tl.cdiv(M, BM)
    tiles_n = tl.cdiv(N, BN)
    for tile_id in tl.range(tl.program_id(0), tiles_m * tiles_n, PROGRAMS):
        tile_m, tile_n = _tile_position(tile_id, tiles_m, tiles_n, 8)
        row = (tile_m * BM).to(tl.int32)
        col = (tile_n * BN).to(tl.int32)
        accumulator = tl.full((BM, BN), 0, tl.float32)
        for block in tl.range(tl.cdiv(K, BK)):
            kk = (block * BK).to(tl.int32)
            av = A_DESC.load([row, kk] if A_ROW else [kk, row])
            bv = B_DESC.load([kk, col] if B_ROW else [col, kk])
            accumulator = tl.dot(av if A_ROW else av.T, bv if B_ROW else bv.T,
                                 accumulator, input_precision="ieee")
        rows = row + tl.arange(0, BM)
        cols = col + tl.arange(0, BN)
        tl.store(C + rows[:, None] * N + cols[None, :], accumulator.to(tl.bfloat16),
                 (rows[:, None] < M) & (cols[None, :] < N))


@torch.library.custom_op("warpstate::persistent_tma_mm", mutates_args=())
def _tma_mm(a: torch.Tensor, b: torch.Tensor, bm: int, bn: int, bk: int,
            warps: int, stages: int) -> torch.Tensor:
    """Opaque launch boundary for the host descriptor API in PyTorch 2.14.

    That release's triton_op/AOT path does not preserve host descriptor metadata.
    This custom-op boundary keeps the TMA implementation compilable without
    substituting a different algorithm or making inputs appear mutable.
    """
    from triton.tools.tensor_descriptor import TensorDescriptor

    m, k = a.shape
    n = b.shape[1]
    a_row, b_row = a.stride(1) == 1, b.stride(1) == 1
    a_desc = TensorDescriptor.from_tensor(a if a_row else a.T,
                                         [bm, bk] if a_row else [bk, bm])
    b_desc = TensorDescriptor.from_tensor(b if b_row else b.T,
                                         [bk, bn] if b_row else [bn, bk])
    output = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    programs = min(triton.cdiv(m, bm) * triton.cdiv(n, bn),
                   torch.cuda.get_device_properties(a.device).multi_processor_count)
    _persistent_tma_kernel[(programs,)](
        a_desc, b_desc, output, m, n, k, bm, bn, bk, programs, a_row, b_row,
        num_warps=warps, num_stages=stages)
    return output


@_tma_mm.register_fake
def _tma_mm_fake(a, b, bm, bn, bk, warps, stages):
    return torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16)


def launch_mm(a, b, algorithm="aten", tile=None):
    """Return CUDA BF16 ``a @ b``; caller supplies any autograd registration.

    Supports positive 2D strides, including ordinary transposed weights and
    transposed output gradients. No inputs or parameters are modified. The
    conservative tile footprint check is not a guarantee of compiler resource
    usage; compilation failures remain explicit and are recorded by the tuner.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError("Unknown matrix algorithm: " + str(algorithm))
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected compatible two-dimensional matrix operands")
    if a.device != b.device or not a.is_cuda:
        raise ValueError("Matrix kernels require operands on the same CUDA device")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("Matrix kernels require BF16 inputs")
    if min(a.shape[0], a.shape[1], b.shape[1]) < 1 or min(*a.stride(), *b.stride()) < 1:
        raise ValueError("Matrix kernels require positive dimensions and strides")
    if algorithm == "aten":
        return torch.mm(a, b)
    bm, bn, bk, warps, stages = normalize_tile(tile)
    m, k = a.shape
    n = b.shape[1]
    if algorithm == "persistent_tma":
        if (min(a.stride()) != 1 or min(b.stride()) != 1
                or max(a.stride()) * 2 % 16 or max(b.stride()) * 2 % 16):
            raise ValueError("TMA operands require an innermost stride of one and 16-byte-aligned rows")
        if a.storage_offset() * 2 % 16 or b.storage_offset() * 2 % 16:
            raise ValueError("TMA operands require 16-byte-aligned storage offsets")
        return _tma_mm(a, b, bm, bn, bk, warps, stages)
    output = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    programs = min(tiles, torch.cuda.get_device_properties(a.device).multi_processor_count) \
        if algorithm == "persistent" else tiles
    wrap_triton(_matmul_kernel)[(programs,)](
        a, b, output, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        bm, bn, bk, 8, programs, algorithm == "persistent",
        num_warps=warps, num_stages=stages)
    return output
