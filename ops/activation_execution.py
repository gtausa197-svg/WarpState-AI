"""Opt-in BF16 SwiGLU and RMSNorm execution without replacing parameters.

Eager execution retains the model's BF16 rounding boundaries. Compiled execution
uses the current Inductor arithmetic, eliding intermediate BF16 casts while
retaining BF16 kernel inputs/outputs. Normalization statistics and master
parameter gradients remain FP32. FP32 arithmetic reductions can differ
slightly from Inductor's reduction order; this is an execution experiment, not a
claim of bitwise equivalence. CPU, evaluation and non-BF16 paths stay original.

Apply before compiling cores. Checkpoints and inference archives are unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
from torch.library import triton_op, wrap_triton


@triton.jit
def _swiglu_forward_kernel(X, Y, N: tl.constexpr, H: tl.constexpr,
                          BLOCK: tl.constexpr, ROUND: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    packed = (i // H) * (2 * H) + i % H
    up = tl.load(X + packed, mask, 0).to(tl.float32)
    gate = tl.load(X + packed + H, mask, 0).to(tl.float32)
    silu = gate / (1.0 + libdevice.exp(-gate))
    if ROUND:
        silu = silu.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + i, silu * up, mask)


@triton.jit
def _swiglu_backward_kernel(X, DY, DX, N: tl.constexpr, H: tl.constexpr,
                           BLOCK: tl.constexpr, ROUND: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    packed = (i // H) * (2 * H) + i % H
    up = tl.load(X + packed, mask, 0).to(tl.float32)
    gate = tl.load(X + packed + H, mask, 0).to(tl.float32)
    dy = tl.load(DY + i, mask, 0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate))
    silu = gate / (1.0 + libdevice.exp(-gate))
    if ROUND:
        silu = silu.to(tl.bfloat16).to(tl.float32)
    # aten's multiply backward rounds before aten's silu backward executes.
    dsilu = dy * up
    if ROUND:
        dsilu = dsilu.to(tl.bfloat16).to(tl.float32)
    dgate = (dsilu * sigmoid) * (1.0 + gate * (1.0 - sigmoid))
    tl.store(DX + packed, dy * silu, mask)
    tl.store(DX + packed + H, dgate, mask)


@triton_op("warpstate::bf16_swiglu", mutates_args={})
def bf16_swiglu(packed: torch.Tensor, eager_rounding: bool = True) -> torch.Tensor:
    """SwiGLU of a contiguous BF16 CUDA [*, 2*hidden] projection.

    Set ``eager_rounding=False`` to match the current compiled core arithmetic.
    """
    hidden = packed.shape[-1] // 2
    output = torch.empty((*packed.shape[:-1], hidden), device=packed.device,
                         dtype=packed.dtype)
    wrap_triton(_swiglu_forward_kernel)[(triton.cdiv(output.numel(), 1024),)](
        packed, output, output.numel(), hidden, 1024, eager_rounding,
        enable_fp_fusion=not eager_rounding)
    return output


@triton_op("warpstate::bf16_swiglu_backward", mutates_args={})
def _swiglu_backward(packed: torch.Tensor, grad: torch.Tensor,
                      eager_rounding: bool) -> torch.Tensor:
    grad = grad.contiguous()
    output = torch.empty_like(packed)
    wrap_triton(_swiglu_backward_kernel)[(triton.cdiv(grad.numel(), 1024),)](
        packed, grad, output, grad.numel(), packed.shape[-1] // 2, 1024, eager_rounding,
        enable_fp_fusion=not eager_rounding)
    return output


def _swiglu_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])
    ctx.eager_rounding = inputs[1]


def _swiglu_autograd(ctx, grad):
    return _swiglu_backward(ctx.saved_tensors[0], grad, ctx.eager_rounding), None


bf16_swiglu.register_autograd(_swiglu_autograd, setup_context=_swiglu_setup)


@triton.jit
def _rms_forward_kernel(X, W, Y, R, D: tl.constexpr, EPS: tl.constexpr,
                        BLOCK: tl.constexpr, ROUND: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    w = tl.load(W + col, col < D, 0)
    if ROUND:
        w = w.to(tl.bfloat16).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    z = x * r
    if ROUND:
        z = z.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + col, z * w, col < D)
    tl.store(R + row, r)


@triton.jit
def _rms_backward_kernel(X, W, R, DY, DX, DW_PART,
                         M: tl.constexpr, D: tl.constexpr,
                         ROWS: tl.constexpr, BLOCK: tl.constexpr, ROUND: tl.constexpr):
    group = tl.program_id(0)
    rows = group * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    mask = (rows[:, None] < M) & (cols[None, :] < D)
    x = tl.load(X + rows[:, None] * D + cols[None, :], mask, 0).to(tl.float32)
    dy = tl.load(DY + rows[:, None] * D + cols[None, :], mask, 0).to(tl.float32)
    w = tl.load(W + cols, cols < D, 0)
    if ROUND:
        w = w.to(tl.bfloat16).to(tl.float32)
    r = tl.load(R + rows, rows < M, 0)
    z = x * r[:, None]
    dz = dy * w[None, :]
    if ROUND:
        z = z.to(tl.bfloat16).to(tl.float32)
        dz = dz.to(tl.bfloat16).to(tl.float32)
    dr = tl.sum(dz * x, 1)
    # Keep the same explicit derivative chain as rsqrt(mean(x**2) + eps).
    dsquare = (-0.5 * dr * (r * r * r)) / D
    dx = dz * r[:, None] + dsquare[:, None] * (2.0 * x)
    dw = dy * z
    if ROUND:
        dw = dw.to(tl.bfloat16).to(tl.float32)
    partial = tl.sum(dw, 0)
    tl.store(DX + rows[:, None] * D + cols[None, :], dx, mask)
    tl.store(DW_PART + group * D + cols, partial, cols < D)


@triton.jit
def _rms_weight_reduce_kernel(PART, DW, D: tl.constexpr, GROUPS: tl.constexpr,
                             BLOCK_G: tl.constexpr, BLOCK_D: tl.constexpr, ROUND: tl.constexpr):
    cols = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    groups = tl.arange(0, BLOCK_G)
    partial = tl.load(PART + groups[:, None] * D + cols[None, :],
                      (groups[:, None] < GROUPS) & (cols[None, :] < D), 0)
    # Eager weight.to(BF16) makes sum-to-size return BF16 before its gradient is
    # cast back to FP32. The compiled path retains the FP32 reduction; Inductor's
    # own partitioning can retain a final BF16 cast at some large shapes, so this
    # comparison is numerically bounded rather than universally bitwise equal.
    total = tl.sum(partial, 0)
    if ROUND:
        total = total.to(tl.bfloat16).to(tl.float32)
    tl.store(DW + cols, total, cols < D)


@triton_op("warpstate::bf16_rmsnorm", mutates_args={})
def bf16_rmsnorm(x: torch.Tensor, weight: torch.Tensor,
                 eps: float, eager_rounding: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 CUDA RMSNorm with FP32 statistics; also return saved row rsqrts."""
    width = x.shape[-1]
    rows = x.numel() // width
    output = torch.empty_like(x)
    rstd = torch.empty((rows,), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(width)
    wrap_triton(_rms_forward_kernel)[(rows,)](
        x, weight, output, rstd, width, eps, block, eager_rounding,
        num_warps=4 if block <= 4096 else 8, enable_fp_fusion=not eager_rounding)
    return output, rstd


@triton_op("warpstate::bf16_rmsnorm_backward", mutates_args={})
def _rms_backward(x: torch.Tensor, weight: torch.Tensor, rstd: torch.Tensor,
                  grad: torch.Tensor, eager_rounding: bool) -> tuple[torch.Tensor, torch.Tensor]:
    grad = grad.contiguous()
    width = x.shape[-1]
    rows = x.numel() // width
    group_rows = 4
    groups = triton.cdiv(rows, group_rows)
    output = torch.empty_like(x)
    partial = torch.empty((groups, width), device=x.device, dtype=torch.float32)
    weight_grad = torch.empty_like(weight)
    block = triton.next_power_of_2(width)
    wrap_triton(_rms_backward_kernel)[(groups,)](
        x, weight, rstd, grad, output, partial, rows, width, group_rows, block, eager_rounding,
        num_warps=4 if block <= 2048 else 8, enable_fp_fusion=not eager_rounding)
    wrap_triton(_rms_weight_reduce_kernel)[(triton.cdiv(width, 8),)](
        partial, weight_grad, width, groups, triton.next_power_of_2(groups), 8, eager_rounding,
        num_warps=8, enable_fp_fusion=False)
    return output, weight_grad


def _rms_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1], output[1])
    ctx.mark_non_differentiable(output[1])
    ctx.eager_rounding = inputs[3]


def _rms_autograd(ctx, grad, _grad_rstd):
    x, weight, rstd = ctx.saved_tensors
    dx, dw = _rms_backward(x, weight, rstd, grad, ctx.eager_rounding)
    return dx, dw, None, None


bf16_rmsnorm.register_autograd(_rms_autograd, setup_context=_rms_setup)


def apply_activation_execution(model: torch.nn.Module, mode: str = "baseline") -> None:
    """Patch training forwards only, retaining canonical FP32 parameter objects.

    Modes select fused ``swiglu``, ``rmsnorm``, or ``all``; ``baseline`` is a no-op.
    Inputs outside the supported contiguous BF16 CUDA shapes use eager model
    code. Evaluation retains the original path, avoiding validation graph churn.
    Eager invocation preserves eager casts; core graph capture selects the
    current Inductor arithmetic with elided intermediate casts.
    """
    from warpstate.model import RMSNorm, SwiGLU

    if mode not in {"baseline", "swiglu", "rmsnorm", "all"}:
        raise ValueError(f"Unknown activation execution mode: {mode}")
    if getattr(model, "_activation_execution_mode", "baseline") != "baseline":
        raise ValueError("Activation execution has already been applied")
    if mode == "baseline":
        return
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Activation execution requires FP32 master parameters")

    for module in model.modules():
        original = module.forward
        if isinstance(module, SwiGLU) and mode in {"swiglu", "all"}:
            def swiglu_forward(x, _module=module, _original=original):
                if (_module.training and torch.is_grad_enabled() and x.is_cuda
                        and torch.is_autocast_enabled("cuda")
                        and torch.get_autocast_dtype("cuda") == torch.bfloat16):
                    packed = _module.up_gate(x)
                    return _module.down(bf16_swiglu(packed.contiguous(),
                        eager_rounding=not torch.compiler.is_compiling()))
                return _original(x)
            module.forward = swiglu_forward
        elif isinstance(module, RMSNorm) and mode in {"rmsnorm", "all"}:
            def rms_forward(x, _module=module, _original=original):
                if (_module.training and torch.is_grad_enabled() and x.is_cuda
                        and x.dtype == torch.bfloat16 and x.is_contiguous()
                        and 0 < x.shape[-1] <= 8192 and x.numel() > 0):
                    return bf16_rmsnorm(x, _module.weight, _module.eps,
                        eager_rounding=not torch.compiler.is_compiling())[0]
                return _original(x)
            module.forward = rms_forward
    model._activation_execution_mode = mode
