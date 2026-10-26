"""Isolated BF16 attention-backward layout experiment, not a training adapter.

The reference boundary is the September 12 compiled backward's rotary kernel
followed by Q/K/V/gate/U packing. Both BF16 intermediate roundings are retained.
No reductions, GEMMs, parameter gradients or optimizer state are changed here.
All saved tensors use [batch,chunks,heads,chunk_size,head_dim] contiguous layout,
except transposed K-write gradients and token-major output gradients.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl


PROFILE_SHAPE = (4, 16, 32, 128, 64)
ALGORITHMS = ("profile", "paired", "single")


class LayoutInputs(NamedTuple):
    q_memory: torch.Tensor
    q_local: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor
    k_write: torch.Tensor
    tanh_k: torch.Tensor
    k_local: torch.Tensor
    v_local: torch.Tensor
    grad_y: torch.Tensor
    memory: torch.Tensor
    local: torch.Tensor
    gate: torch.Tensor
    u_write: torch.Tensor
    tanh_u: torch.Tensor


def validate_inputs(inputs: LayoutInputs, *, cuda=False):
    if not isinstance(inputs, LayoutInputs):
        raise TypeError("Expected named LayoutInputs")
    shape = tuple(inputs.q_memory.shape)
    if len(shape) != 5 or min(shape) < 1 or shape[-1] % 2:
        raise ValueError("Expected positive [B,N,H,C,D] with an even head dimension")
    b, n, h, c, d = shape
    device = inputs.q_memory.device
    if cuda and device.type != "cuda":
        raise ValueError("Triton layout kernels require CUDA")
    for name, tensor in zip(inputs._fields, inputs, strict=True):
        wanted = shape
        dtype = torch.bfloat16
        if name in ("sin", "cos"):
            wanted, dtype = (c, d), torch.float32
        elif name == "k_write":
            wanted = (b, n, h, d, c)
        elif name == "grad_y":
            wanted = (b, n * c, h * d)
        if tuple(tensor.shape) != wanted or tensor.dtype != dtype:
            raise ValueError(f"{name} must have shape {wanted} and dtype {dtype}")
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {device}")
    return shape


def make_inputs(shape=PROFILE_SHAPE, *, device="cpu", seed=817) -> LayoutInputs:
    """Deterministic synthetic saved activations/gradients, no model/data reads."""
    b, n, h, c, d = shape
    generator = torch.Generator(device=device).manual_seed(seed)

    def random(dims=shape, scale=0.1):
        return torch.randn(dims, device=device, dtype=torch.bfloat16,
                           generator=generator) * scale

    from warpstate.model import rope_cos_sin
    cos, sin = rope_cos_sin(c, d, torch.device(device), torch.float32)
    return LayoutInputs(random(), random(), sin, cos, random((b, n, h, d, c)),
                        random(scale=1).float().tanh().bfloat16(), random(), random(),
                        random((b, n * c, h * d)), random(), random(),
                        random(scale=1).float().sigmoid().bfloat16(), random(),
                        random(scale=1).float().tanh().bfloat16())


def reference_layout(inputs: LayoutInputs) -> torch.Tensor:
    """Independent tensor expression for both stages, including BF16 boundaries.

    FP32 operation contraction may differ between eager PyTorch and Triton;
    use the tracked profile kernels as the exact GPU rounding oracle.
    """
    b, n, h, c, d = validate_inputs(inputs)
    q = inputs.q_memory.float() + inputs.q_local.float()
    k = (inputs.k_write.transpose(-2, -1).float()
         * (1 - inputs.tanh_k.float().square()) + inputs.k_local.float())

    def inverse_rotate(value):
        return torch.stack((value[..., 1::2], -value[..., ::2]), dim=-1).flatten(-2)

    q_rounded = (inverse_rotate(q * inputs.sin) + q * inputs.cos).bfloat16()
    k_rotated_rounded = inverse_rotate(k * inputs.sin).bfloat16()
    k_packed = (k_rotated_rounded.float() + k * inputs.cos).bfloat16()
    dy = inputs.grad_y.view(b, n, c, h, d).permute(0, 1, 3, 2, 4).float()
    gate = inputs.gate.float()
    dg = ((-(dy * (inputs.memory.float() * d**-0.5))
           + dy * inputs.local.float()) * (gate * (1 - gate))).bfloat16()
    du = (inputs.u_write.float() * (1 - inputs.tanh_u.float().square())).bfloat16()

    def token_major(value):
        return value.permute(0, 1, 3, 2, 4).reshape(b, n * c, h * d)

    return torch.cat([token_major(value) for value in
                      (q_rounded, k_packed, inputs.v_local, dg, du)], dim=-1)


@triton.jit
def _rotary_store(QM, QL, SIN, COS, KW, TK, KL, OUT,
                  ELEMENTS: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
                  D: tl.constexpr, BLOCK: tl.constexpr, program):
    pairs = program * BLOCK + tl.arange(0, BLOCK)
    even = pairs * 2
    odd = even + 1
    mask = even < ELEMENTS
    token = (even // D) % C
    channel = even % D
    group = even // (C * D)
    rope = token * D + channel
    s0 = tl.load(SIN + rope, mask, 0)
    s1 = tl.load(SIN + rope + 1, mask, 0)
    c0 = tl.load(COS + rope, mask, 0)
    c1 = tl.load(COS + rope + 1, mask, 0)
    q0 = (tl.load(QM + even, mask, 0).to(tl.float32)
          + tl.load(QL + even, mask, 0).to(tl.float32))
    q1 = (tl.load(QM + odd, mask, 0).to(tl.float32)
          + tl.load(QL + odd, mask, 0).to(tl.float32))
    # The profiled interleave sums each active rotary lane with an inactive +0.
    # Retain that IEEE signed-zero boundary as well as its BF16 store below.
    q_even_rot = q1 * s1 + 0.0
    q_odd_rot = (-q0 * s0) + 0.0
    q_even = (q_even_rot + q0 * c0).to(tl.bfloat16)
    q_odd = (q_odd_rot + q1 * c1).to(tl.bfloat16)
    kw_offset = group * C * D + channel * C + token
    kw0 = tl.load(KW + kw_offset, mask, 0).to(tl.float32)
    kw1 = tl.load(KW + kw_offset + C, mask, 0).to(tl.float32)
    tk0 = tl.load(TK + even, mask, 0).to(tl.float32)
    tk1 = tl.load(TK + odd, mask, 0).to(tl.float32)
    k0 = kw0 * (1.0 - tk0 * tk0) + tl.load(KL + even, mask, 0).to(tl.float32)
    k1 = kw1 * (1.0 - tk1 * tk1) + tl.load(KL + odd, mask, 0).to(tl.float32)
    # Stage 17 stores these separately, before adding the cosine term in 18.
    # Retain the truncation even though the fused implementation has no scratch.
    k_even_rot = (k1 * s1 + 0.0).to(tl.bfloat16).to(tl.float32)
    k_odd_rot = ((-k0 * s0) + 0.0).to(tl.bfloat16).to(tl.float32)
    k_even = k_even_rot + k0 * c0
    k_odd = k_odd_rot + k1 * c1
    out = ((group // H) * C + token) * (5 * H * D) + (group % H) * D + channel
    tl.store(OUT + out, q_even, mask)
    tl.store(OUT + out + 1, q_odd, mask)
    tl.store(OUT + out + H * D, k_even, mask)
    tl.store(OUT + out + H * D + 1, k_odd, mask)


@triton.jit
def _other_store(VL, DY, MEMORY, LOCAL, GATE, UW, TU, OUT,
                 ELEMENTS: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
                 D: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr, program):
    index = program * BLOCK + tl.arange(0, BLOCK)
    mask = index < ELEMENTS
    token = (index // D) % C
    channel = index % D
    group = index // (C * D)
    token_row = (group // H) * C + token
    feature = (group % H) * D + channel
    dy = tl.load(DY + token_row * H * D + feature, mask, 0).to(tl.float32)
    memory = tl.load(MEMORY + index, mask, 0).to(tl.float32)
    local = tl.load(LOCAL + index, mask, 0).to(tl.float32)
    gate = tl.load(GATE + index, mask, 0).to(tl.float32)
    # Keep the original chain rather than algebraically factoring dy.
    dgate = (-(dy * (memory * SCALE)) + dy * local) * (gate * (1.0 - gate))
    uw = tl.load(UW + index, mask, 0).to(tl.float32)
    tu = tl.load(TU + index, mask, 0).to(tl.float32)
    du = uw * (1.0 - tu * tu)
    out = token_row * 5 * H * D + feature
    tl.store(OUT + out + 2 * H * D, tl.load(VL + index, mask, 0), mask)
    tl.store(OUT + out + 3 * H * D, dgate, mask)
    tl.store(OUT + out + 4 * H * D, du, mask)


@triton.jit
def _paired_kernel(QM, QL, SIN, COS, KW, TK, KL, VL, DY, MEMORY, LOCAL, GATE, UW, TU, OUT,
                   ELEMENTS: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr,
                   ROTARY: tl.constexpr):
    if ROTARY:
        _rotary_store(QM, QL, SIN, COS, KW, TK, KL, OUT,
                      ELEMENTS, H, C, D, BLOCK, tl.program_id(0))
    else:
        _other_store(VL, DY, MEMORY, LOCAL, GATE, UW, TU, OUT,
                     ELEMENTS, H, C, D, SCALE, BLOCK, tl.program_id(0))


@triton.jit
def _single_kernel(QM, QL, SIN, COS, KW, TK, KL, VL, DY, MEMORY, LOCAL, GATE, UW, TU, OUT,
                   ELEMENTS: tl.constexpr, H: tl.constexpr, C: tl.constexpr,
                   D: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    rotary_programs = tl.cdiv(ELEMENTS // 2, BLOCK)
    program = tl.program_id(0)
    if program < rotary_programs:
        _rotary_store(QM, QL, SIN, COS, KW, TK, KL, OUT,
                      ELEMENTS, H, C, D, BLOCK, program)
    else:
        _other_store(VL, DY, MEMORY, LOCAL, GATE, UW, TU, OUT,
                     ELEMENTS, H, C, D, SCALE, BLOCK, program - rotary_programs)


def prepare_layout(inputs: LayoutInputs, algorithm="paired", *, block=512, warps=4,
                   output=None, scratch=None):
    """Validate once and return an allocation-free launch callable.

    Callers must keep the inputs/output/scratch metadata and storage stable.
    No GPU kernel executes until the returned callable is invoked.
    """
    b, n, h, c, d = validate_inputs(inputs, cuda=True)
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown backward layout algorithm: {algorithm}")
    if block not in (128, 256, 512, 1024) or warps not in (4, 8):
        raise ValueError("Use block 128/256/512/1024 with 4/8 warps")
    shape = (b, n * c, 5 * h * d)
    if output is None:
        output = torch.empty(shape, dtype=torch.bfloat16, device=inputs.q_memory.device)
    if (tuple(output.shape) != shape or output.dtype != torch.bfloat16
            or output.device != inputs.q_memory.device or not output.is_contiguous()):
        raise ValueError("Output must be a contiguous BF16 packed gradient on the input device")
    if any(output.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
           for value in inputs):
        raise ValueError("Output may not alias a saved input")
    if algorithm == "profile":
        if (b, n, h, c, d) != PROFILE_SHAPE:
            raise ValueError("The exact profiled baseline supports PROFILE_SHAPE only")
        from ops.backward_layout_baseline import _profile_pack, _profile_rotary
        if scratch is None:
            scratch = (torch.empty_like(inputs.q_memory), torch.empty_like(inputs.q_memory))
        if (len(scratch) != 2 or any(value.shape != inputs.q_memory.shape
                or value.dtype != torch.bfloat16 or value.device != output.device
                or not value.is_contiguous() for value in scratch)):
            raise ValueError("The profile baseline needs two contiguous BF16 scratch tensors")
        q, kr = scratch
        storages = [value.untyped_storage().data_ptr() for value in (*inputs, output)]
        if (q.untyped_storage().data_ptr() in storages
                or kr.untyped_storage().data_ptr() in storages
                or q.untyped_storage().data_ptr() == kr.untyped_storage().data_ptr()):
            raise ValueError("Scratch tensors may not alias inputs, output, or one another")

        def operation():
            _profile_rotary[(1, 16384)](
                inputs.q_memory, inputs.q_local, inputs.sin, inputs.cos,
                inputs.k_write, inputs.tanh_k, inputs.k_local, q, kr,
                262144, 64, YBLOCK=16, XBLOCK=64, num_warps=4, enable_fp_fusion=True)
            _profile_pack[(160, 128)](
                q, kr, inputs.k_write, inputs.tanh_k, inputs.k_local, inputs.cos,
                inputs.v_local, inputs.grad_y, inputs.memory, inputs.local, inputs.gate,
                inputs.u_write, inputs.tanh_u, output, 8192, 10240,
                YBLOCK=64, XBLOCK=64, num_warps=8, enable_fp_fusion=True)
            return output
    else:
        elements = inputs.q_memory.numel()
        args = (*inputs, output, elements, h, c, d, d**-0.5, block)

        def operation():
            if algorithm == "paired":
                _paired_kernel[(triton.cdiv(elements // 2, block),)](
                    *args, True, num_warps=warps, enable_fp_fusion=True)
                _paired_kernel[(triton.cdiv(elements, block),)](
                    *args, False, num_warps=warps, enable_fp_fusion=True)
            else:
                _single_kernel[(triton.cdiv(elements // 2, block) + triton.cdiv(elements, block),)](
                    *args, num_warps=warps, enable_fp_fusion=True)
            return output
    return operation


def launch_layout(inputs: LayoutInputs, algorithm="paired", *, block=512, warps=4,
                  output=None, scratch=None) -> torch.Tensor:
    """One checked launch; timed loops should use prepare_layout once instead."""
    return prepare_layout(inputs, algorithm, block=block, warps=warps,
                          output=output, scratch=scratch)()
