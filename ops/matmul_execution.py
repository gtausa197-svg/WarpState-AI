"""Opt-in, shape-specific BF16 matrix execution with ordinary autograd ownership.

Plans are validated and registered once, before compiling cores or constructing
DDP. FP32 parameter casts stay inside each visible forward graph. CUDA kernels
return BF16 values and gradients, and ordinary differentiable casts return the
gradients to the original FP32 leaves. No hook writes a parameter gradient.
Unsupported/unselected shapes use ATen; CPU and evaluation model paths stay
original. Checkpoint keys and archived model layouts are not changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch.library import custom_op, triton_op


PLAN_FORMAT = "warpstate-matmul-plan-v1"
_DEFAULT_TILE = (64, 64, 32, 4, 3)
_TILE_FIELDS = ("block_m", "block_n", "block_k", "num_warps", "num_stages")
_PLANS: dict[str, "MatmulPlan"] = {}


@dataclass(frozen=True)
class MatmulEntry:
    kind: str
    m: int
    n: int
    k: int
    a_stride: tuple[int, int]
    b_stride: tuple[int, int]
    algorithm: str
    tile: tuple[int, int, int, int, int]
    name: str = ""

    @property
    def key(self):
        return (self.kind, self.m, self.n, self.k, *self.a_stride, *self.b_stride)

    def as_dict(self):
        return {"kind": self.kind, "m": self.m, "n": self.n, "k": self.k,
                "a_stride": list(self.a_stride), "b_stride": list(self.b_stride),
                "dtype": "bfloat16", "algorithm": self.algorithm,
                "tile": dict(zip(_TILE_FIELDS, self.tile, strict=True)), "name": self.name}


@dataclass(frozen=True)
class MatmulPlan:
    entries: tuple[MatmulEntry, ...]
    sha256: str
    source_hashes: tuple[tuple[str, str], ...]
    registry_key: str
    description: str = ""

    @property
    def metadata(self):
        """Serializable provenance for runtime records and resume signatures."""
        return {"format": PLAN_FORMAT, "sha256": self.sha256,
                "source_sha256": dict(self.source_hashes),
                "entries": [entry.as_dict() for entry in self.entries]}

    def select(self, kind, m, n, k, a_stride, b_stride):
        """Pure shape/layout lookup; no device work or filesystem access."""
        key = (kind, m, n, k, *a_stride, *b_stride)
        for entry in self.entries:
            if entry.key == key:
                return entry
        return None


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _stride(value, name):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain two positive integer strides")
    return tuple(_positive_integer(item, name) for item in value)


def _no_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _implementation_hashes(has_fused_swiglu):
    names = ["matmul_execution.py", "matmul_kernels.py"]
    if has_fused_swiglu:
        names += ["matmul_epilogue.py", "activation_execution.py"]
    directory = Path(__file__).resolve().parent
    return tuple(("ops/" + name, hashlib.sha256((directory / name).read_bytes()).hexdigest())
                 for name in sorted(names))


def matmul_plan_from_dict(document: dict) -> MatmulPlan:
    """Validate a source recipe; subsequent selection only uses immutable data."""
    from ops.matmul_kernels import normalize_tile

    if not isinstance(document, dict):
        raise ValueError("Matmul plan must be a JSON object")
    if set(document) - {"format", "description", "entries"}:
        raise ValueError("Unknown matmul plan fields")
    if document.get("format") != PLAN_FORMAT:
        raise ValueError(f"Matmul plan format must be {PLAN_FORMAT}")
    description = document.get("description", "")
    if not isinstance(description, str):
        raise ValueError("Matmul plan description must be a string")
    if not isinstance(document.get("entries"), list):
        raise ValueError("Matmul plan entries must be a list")
    entries = []
    seen = set()
    fields = {"kind", "m", "n", "k", "a_stride", "b_stride", "dtype", "algorithm", "tile", "name"}
    for index, row in enumerate(document["entries"]):
        if not isinstance(row, dict) or set(row) - fields:
            raise ValueError(f"Unknown fields in matmul entry {index}")
        if row.get("kind") not in {"mm", "swiglu"}:
            raise ValueError("Matmul entry kind must be mm or swiglu")
        if row.get("dtype") != "bfloat16":
            raise ValueError("Matmul plans support BF16 operands only")
        if row.get("algorithm") not in {"aten", "tiled", "persistent", "persistent_tma"}:
            raise ValueError("Unknown matmul algorithm")
        if row["kind"] == "swiglu" and row["algorithm"] == "persistent_tma":
            raise ValueError("Fused SwiGLU does not yet support the persistent_tma algorithm")
        dimensions = tuple(_positive_integer(row.get(name), name) for name in ("m", "n", "k"))
        if row["kind"] == "swiglu" and dimensions[1] % 2:
            raise ValueError("Fused SwiGLU output width must be even")
        name = row.get("name", "")
        if not isinstance(name, str):
            raise ValueError("Matmul entry name must be a string")
        tile_document = row.get("tile", dict(zip(_TILE_FIELDS, _DEFAULT_TILE, strict=True)))
        if not isinstance(tile_document, dict) or set(tile_document) != set(_TILE_FIELDS):
            raise ValueError("Matmul tile must specify block_m/block_n/block_k/num_warps/num_stages")
        for field, value in tile_document.items():
            _positive_integer(value, field)
        tile = tuple(normalize_tile(tile_document))
        if row["kind"] == "swiglu":
            bm, bn, bk, _warps, stages = tile
            if (bm + 2 * bn) * bk * 2 * stages > 96 * 1024:
                raise ValueError("Fused SwiGLU tile exceeds the conservative shared-memory budget")
        entry = MatmulEntry(row["kind"], *dimensions,
                            _stride(row.get("a_stride"), "a_stride"),
                            _stride(row.get("b_stride"), "b_stride"),
                            row["algorithm"], tile, name)
        if entry.key in seen:
            raise ValueError("Duplicate matmul shape/layout entry")
        seen.add(entry.key)
        entries.append(entry)
    canonical = {"format": PLAN_FORMAT, "description": description,
                 "entries": [entry.as_dict() for entry in entries]}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    hashes = _implementation_hashes(any(entry.kind == "swiglu" for entry in entries))
    registry_key = hashlib.sha256((digest + json.dumps(hashes)).encode()).hexdigest()
    plan = MatmulPlan(tuple(entries), digest, hashes, registry_key, description)
    existing = _PLANS.get(registry_key)
    if existing is not None and existing != plan:
        raise RuntimeError("Matmul plan registry identity collision")
    _PLANS[registry_key] = plan
    return plan


def load_matmul_plan(path: str | Path) -> MatmulPlan:
    """Load once before training; no recipes or hashes are read during forward."""
    path = Path(path).expanduser().resolve()
    workspace = Path(__file__).resolve().parents[2]
    if not path.is_relative_to(workspace):
        raise ValueError("Matmul plan must remain inside the WarpState workspace")
    document = json.loads(path.read_text(), object_pairs_hook=_no_duplicate_json_keys)
    return matmul_plan_from_dict(document)


def _choice(a, b, plan_key, kind="mm"):
    if (a.ndim != 2 or b.ndim != 2 or not a.is_cuda or not b.is_cuda
            or a.device != b.device or a.dtype != torch.bfloat16
            or b.dtype != torch.bfloat16 or a.shape[1] != b.shape[0]
            or a.numel() == 0 or b.numel() == 0
            or any(stride <= 0 for stride in (*a.stride(), *b.stride()))):
        return None
    plan = _PLANS[plan_key]
    entry = plan.select(kind, a.shape[0], b.shape[1], a.shape[1], a.stride(), b.stride())
    if entry is not None and entry.algorithm == "persistent_tma":
        # Torch allocations have an aligned base; sliced views may not retain
        # descriptor alignment even when their shape/stride matches a plan.
        if (min(a.stride()) != 1 or min(b.stride()) != 1
                or max(a.stride()) * 2 % 16 or max(b.stride()) * 2 % 16
                or a.storage_offset() * 2 % 16 or b.storage_offset() * 2 % 16):
            return None
    return entry


@custom_op("warpstate::selected_matrix_kernel", mutates_args=())
def _selected_matrix_kernel(a: torch.Tensor, b: torch.Tensor, plan_key: str) -> torch.Tensor:
    """Opaque launch, retaining the outer operator's registered autograd.

    PyTorch 2.14 Compiled Autograd retraces the AOT backward graph, but Dynamo
    cannot retrace its functionalized Triton higher-order launch operator.
    Keep selected launches opaque in that graph; ordinary ATen fallbacks stay
    visible to the compiler. This also covers activation recomputation.
    """
    from ops.matmul_kernels import launch_mm

    entry = _choice(a, b, plan_key)
    if entry is None or entry.algorithm == "aten":
        return torch.mm(a, b)
    return launch_mm(a, b, entry.algorithm, entry.tile)


@_selected_matrix_kernel.register_fake
def _selected_matrix_kernel_fake(a, b, plan_key):
    return torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)


@triton_op("warpstate::planned_mm", mutates_args={})
def _planned_mm(a: torch.Tensor, b: torch.Tensor, plan_key: str) -> torch.Tensor:
    entry = _choice(a, b, plan_key)
    if entry is None or entry.algorithm == "aten":
        return torch.mm(a, b)
    return _selected_matrix_kernel(a, b, plan_key)


def _mm_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1])
    ctx.plan_key = inputs[2]


def _mm_backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad = grad.contiguous()
    da = _planned_mm(grad, b.T, ctx.plan_key) if ctx.needs_input_grad[0] else None
    # Match Linear's production dW orientation (dY.T @ X), then return the
    # gradient of the transposed BF16 weight operand. This avoids an extra
    # gradient-layout copy and lets the plan reuse measured dW shapes/strides.
    db = _planned_mm(grad.T, a, ctx.plan_key).T if ctx.needs_input_grad[1] else None
    return da, db, None


_planned_mm.register_autograd(_mm_backward, setup_context=_mm_setup)


def planned_matmul(a: torch.Tensor, b: torch.Tensor, plan: MatmulPlan) -> torch.Tensor:
    """Autograd-enabled entry point, also useful for isolated correctness checks."""
    return _planned_mm(a, b, plan.registry_key)


@custom_op("warpstate::selected_swiglu_kernel", mutates_args=())
def _selected_swiglu_kernel(a: torch.Tensor, weight: torch.Tensor, plan_key: str,
                            eager_rounding: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept native-layout BF16 W; form its projection view only at launch.

    Passing W.T across the opaque boundary makes Inductor exact-stride
    constraints materialize a second cast buffer alongside the W saved for
    backward. A native-layout input lets both consumers share the same visible
    FP32-to-BF16 cast. Autograd ownership and the plan still use the outer B=W.T.
    """
    from ops.matmul_epilogue import launch_swiglu_mm

    b = weight.T
    entry = _choice(a, b, plan_key, "swiglu")
    if entry is None or entry.algorithm == "aten":
        return launch_swiglu_mm(a, b, "aten", _DEFAULT_TILE, eager_rounding)
    return launch_swiglu_mm(a, b, entry.algorithm, entry.tile, eager_rounding)


@_selected_swiglu_kernel.register_fake
def _selected_swiglu_kernel_fake(a, weight, plan_key, eager_rounding):
    packed = torch.empty((a.shape[0], weight.shape[0]), dtype=a.dtype, device=a.device)
    activation = torch.empty((a.shape[0], weight.shape[0] // 2), dtype=a.dtype, device=a.device)
    return packed, activation


@custom_op("warpstate::selected_swiglu_gradient_kernel", mutates_args=())
def _selected_swiglu_gradient_kernel(packed: torch.Tensor, grad: torch.Tensor,
                                     eager_rounding: bool) -> torch.Tensor:
    from ops.activation_execution import _swiglu_backward

    return _swiglu_backward(packed, grad, eager_rounding)


@_selected_swiglu_gradient_kernel.register_fake
def _selected_swiglu_gradient_kernel_fake(packed, grad, eager_rounding):
    return torch.empty_like(packed)


@triton_op("warpstate::planned_swiglu_mm", mutates_args={})
def _planned_swiglu_mm(a: torch.Tensor, b: torch.Tensor, plan_key: str,
                       eager_rounding: bool) -> tuple[torch.Tensor, torch.Tensor]:
    from ops.matmul_epilogue import launch_swiglu_mm

    entry = _choice(a, b, plan_key, "swiglu")
    if entry is not None and entry.algorithm != "aten":
        return _selected_swiglu_kernel(a, b.T, plan_key, eager_rounding)
    algorithm = entry.algorithm if entry is not None else "aten"
    tile = entry.tile if entry is not None else _DEFAULT_TILE
    return launch_swiglu_mm(a, b, algorithm, tile, eager_rounding)


def _fused_setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1], output[0])
    ctx.plan_key = inputs[2]
    ctx.eager_rounding = inputs[3]
    ctx.mark_non_differentiable(output[0])


def _fused_backward(ctx, _packed_grad, grad):
    a, b, packed = ctx.saved_tensors
    projection_grad = _selected_swiglu_gradient_kernel(packed, grad, ctx.eager_rounding)
    da = _planned_mm(projection_grad, b.T, ctx.plan_key) if ctx.needs_input_grad[0] else None
    db = _planned_mm(projection_grad.T, a, ctx.plan_key).T if ctx.needs_input_grad[1] else None
    return da, db, None, None


_planned_swiglu_mm.register_autograd(_fused_backward, setup_context=_fused_setup)


def _linear_has_choices(module, plan):
    width, output = module.in_features, module.out_features
    return any(entry.kind == "mm" and (
        (entry.n == output and entry.k == width)  # Forward.
        or (entry.n == width and entry.k == output)  # dX.
        or (entry.m == output and entry.n == width)  # dW.
    ) for entry in plan.entries)


def apply_matmul_execution(model: torch.nn.Module, plan: MatmulPlan) -> None:
    """Apply before core compilation and DDP; CPU/evaluation paths stay original.

    Selected matrix calls do not replace parameters with externally cached BF16
    views. The ordinary FP32-to-BF16 casts are part of each visible forward graph
    so AOTAutograd/DDP retain their original parameter dependencies and hooks.
    """
    from warpstate.model import SwiGLU

    if not isinstance(plan, MatmulPlan):
        raise TypeError("Load a MatmulPlan before applying matrix execution")
    if getattr(model, "_matmul_execution_plan", None) is not None:
        raise ValueError("Matmul execution has already been applied")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Matmul execution requires FP32 master parameters")
    if plan.registry_key not in _PLANS:
        raise ValueError("Matmul plan is not registered in this process")
    model._matmul_execution_plan = plan
    if not plan.entries:
        return
    has_fused = any(entry.kind == "swiglu" and entry.algorithm != "aten" for entry in plan.entries)
    for core in model.cores:
        for module in core.modules():
            original = module.forward
            if (isinstance(module, torch.nn.Linear) and module.bias is None
                    and _linear_has_choices(module, plan)):
                def linear_forward(x, _module=module, _original=original):
                    if (not _module.training or not torch.is_grad_enabled() or not x.is_cuda
                            or not torch.is_autocast_enabled("cuda")
                            or torch.get_autocast_dtype("cuda") != torch.bfloat16):
                        return _original(x)
                    a = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
                    b = _module.weight.to(torch.bfloat16).T
                    output = _planned_mm(a, b, plan.registry_key)
                    return output.reshape(*x.shape[:-1], _module.out_features)
                module.forward = linear_forward
            elif (isinstance(module, SwiGLU) and has_fused and any(
                    entry.kind == "swiglu" and entry.n == module.up_gate.out_features
                    and entry.k == module.up_gate.in_features for entry in plan.entries)):
                def swiglu_forward(x, _module=module, _original=original):
                    if (not _module.training or not torch.is_grad_enabled() or not x.is_cuda
                            or not torch.is_autocast_enabled("cuda")
                            or torch.get_autocast_dtype("cuda") != torch.bfloat16
                            or _module.up_gate.bias is not None):
                        return _original(x)
                    a = x.reshape(-1, x.shape[-1]).to(torch.bfloat16)
                    b = _module.up_gate.weight.to(torch.bfloat16).T
                    entry = _choice(a, b, plan.registry_key, "swiglu")
                    if entry is None or entry.algorithm == "aten":
                        return _original(x)
                    _, activation = _planned_swiglu_mm(
                        a, b, plan.registry_key, not torch.compiler.is_compiling())
                    activation = activation.reshape(*x.shape[:-1], _module.down.in_features)
                    return _module.down(activation)
                module.forward = swiglu_forward
