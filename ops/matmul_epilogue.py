"""BF16 paired FFN projections with a SwiGLU epilogue.

This module launches kernels only. The training adapter owns autograd and keeps
the original FP32 master parameters. Both projection halves are rounded to BF16
before activation, and the packed projection is returned for ordinary backward.
Fusing the forward therefore saves a projection read, not its required storage.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.library import wrap_triton


SUPPORTED_TILES = (
    (64, 64, 32, 4, 3),
    (64, 64, 64, 4, 2),
    (128, 64, 32, 8, 3),
    (64, 128, 32, 8, 3),
    (128, 64, 32, 4, 3),
    (64, 128, 32, 4, 3),
    (128, 128, 32, 8, 3),
)


@triton.jit
def _paired_swiglu_mm_kernel(
    A, B, PACKED, OUTPUT,
    M: tl.constexpr, H: tl.constexpr, K: tl.constexpr,
    STRIDE_AM: tl.constexpr, STRIDE_AK: tl.constexpr,
    STRIDE_BK: tl.constexpr, STRIDE_BN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr, GROUP_M: tl.constexpr, ROUND: tl.constexpr,
):
    tiles_m = tl.cdiv(M, BLOCK_M)
    tiles_n = tl.cdiv(H, BLOCK_N)
    total_tiles = tiles_m * tiles_n
    for tile_id in range(tl.program_id(0), total_tiles, NUM_PROGRAMS):
        # Group neighboring row tiles to reuse the two weight panels in L2.
        tiles_per_group = GROUP_M * tiles_n
        group = tile_id // tiles_per_group
        first_m = group * GROUP_M
        group_m = tl.minimum(tiles_m - first_m, GROUP_M)
        inner = tile_id % tiles_per_group
        pid_m = first_m + inner % group_m
        pid_n = inner // group_m
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        reduction = tl.arange(0, BLOCK_K)
        # Adjacent logical columns pair up/gate from different packed halves.
        # One wider dot shares A and lets the MMA scheduler handle both halves;
        # split the accumulator in registers before the rounded epilogue.
        pair_columns = tl.arange(0, 2 * BLOCK_N)
        paired_cols = pid_n * BLOCK_N + pair_columns // 2
        paired_offsets = paired_cols + (pair_columns % 2) * H
        accumulator = tl.zeros((BLOCK_M, 2 * BLOCK_N), tl.float32)
        for start in range(tl.cdiv(K, BLOCK_K)):
            ks = start * BLOCK_K + reduction
            x = tl.load(A + rows[:, None] * STRIDE_AM + ks[None, :] * STRIDE_AK,
                        (rows[:, None] < M) & (ks[None, :] < K), 0)
            paired_weights = tl.load(
                B + ks[:, None] * STRIDE_BK + paired_offsets[None, :] * STRIDE_BN,
                (ks[:, None] < K) & (paired_cols[None, :] < H), 0)
            accumulator = tl.dot(x, paired_weights, accumulator)
        up_acc, gate_acc = tl.split(tl.reshape(accumulator, (BLOCK_M, BLOCK_N, 2)))
        # These casts are essential, independently of ROUND. SwiGLU consumes
        # the BF16 GEMM output in both the eager and existing compiled model.
        up_bf16 = up_acc.to(tl.bfloat16)
        gate_bf16 = gate_acc.to(tl.bfloat16)
        mask = (rows[:, None] < M) & (cols[None, :] < H)
        packed_offsets = rows[:, None] * (2 * H) + cols[None, :]
        tl.store(PACKED + packed_offsets, up_bf16, mask)
        tl.store(PACKED + packed_offsets + H, gate_bf16, mask)
        up = up_bf16.to(tl.float32)
        gate = gate_bf16.to(tl.float32)
        silu = gate / (1.0 + libdevice.exp(-gate))
        if ROUND:
            silu = silu.to(tl.bfloat16).to(tl.float32)
        tl.store(OUTPUT + rows[:, None] * H + cols[None, :], silu * up, mask)


def launch_swiglu_mm(a: torch.Tensor, b: torch.Tensor, algorithm: str = "tiled",
                     tile=SUPPORTED_TILES[0], eager_rounding: bool = False
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed ``a @ b`` and SwiGLU, both BF16 and row-contiguous.

    ``a`` has shape [M,K], ``b`` [K,2H]. Positive-stride layouts, including
    transposed FP32-master weight casts, are supported. ``tile`` is
    (block_m, block_n, block_k, num_warps, num_stages); block_n counts columns
    in *each* projection half. ``persistent`` assigns one program per SM and
    strides over tiles; ``tiled`` launches one program per paired output tile.

    ``eager_rounding`` controls only the SiLU-output BF16 rounding before the
    multiply. Projection rounding always remains in place. ``aten`` is a
    reference execution path using the same explicit rounding choices.
    """
    from ops.matmul_kernels import normalize_tile

    if algorithm not in ("aten", "tiled", "persistent"):
        raise ValueError(f"Unknown SwiGLU matmul algorithm: {algorithm}")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("SwiGLU matmul expects compatible [M,K] and [K,2H] matrices")
    if min(a.shape) <= 0 or b.shape[1] <= 0 or b.shape[1] % 2:
        raise ValueError("SwiGLU matmul requires positive dimensions and an even packed width")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError("SwiGLU matmul requires BF16 operands")
    if a.device != b.device or any(stride <= 0 for tensor in (a, b) for stride in tensor.stride()):
        raise ValueError("SwiGLU matmul requires the same device and positive strides")
    block_m, block_n, block_k, warps, stages = normalize_tile(tile)
    # Two independent B panels share one A panel. Avoid configurations whose
    # pipeline alone exceeds the conservative SM120 96 KiB shared-memory cap.
    shared_bytes = (block_m + 2 * block_n) * block_k * 2 * stages
    if shared_bytes > 96 * 1024:
        raise ValueError("Paired SwiGLU tile exceeds the conservative 96 KiB shared-memory budget")
    if algorithm == "aten":
        packed = torch.mm(a, b)
        up, gate = packed.chunk(2, dim=-1)
        if eager_rounding:
            output = F.silu(gate) * up
        else:
            output = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        return packed, output
    if a.device.type != "cuda":
        raise ValueError("Custom SwiGLU matmul kernels require CUDA")
    rows, reduction = a.shape
    hidden = b.shape[1] // 2
    packed = torch.empty((rows, 2 * hidden), dtype=torch.bfloat16, device=a.device)
    output = torch.empty((rows, hidden), dtype=torch.bfloat16, device=a.device)
    total_tiles = triton.cdiv(rows, block_m) * triton.cdiv(hidden, block_n)
    programs = total_tiles
    if algorithm == "persistent":
        programs = min(total_tiles, torch.cuda.get_device_properties(a.device).multi_processor_count)
    wrap_triton(_paired_swiglu_mm_kernel)[(programs,)](
        a, b, packed, output, rows, hidden, reduction,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        block_m, block_n, block_k, programs, 8, eager_rounding,
        num_warps=warps, num_stages=stages, enable_fp_fusion=not eager_rounding)
    return packed, output
