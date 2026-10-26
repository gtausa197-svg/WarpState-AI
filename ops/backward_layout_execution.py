"""Opt-in replacement of the measured, normalized attention backward subgraph.

The boundary is the existing compiled BF16 kernel arithmetic, not eager BF16
ATen arithmetic: Inductor normally retains FP32 intermediates within a fused
kernel. Forward/autograd ownership, leaf accumulation and communication remain
unchanged. Unsupported graphs are retained; callers must require actual runtime
dispatches in opt-in validation. Configure once before compiling in a fresh process.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import operator
from pathlib import Path

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass
from torch.fx import Node
from torch.library import custom_op


BACKWARD_LAYOUT_MODES = ("baseline", "inspect", "paired")
_SHAPE = (4, 16, 32, 128, 64)
_NAMES = ("q_memory", "q_local", "sin", "cos", "k_write", "tanh_k", "k_local",
          "v_local", "grad_y", "memory", "local", "gate", "u_write", "tanh_u")
_NATIVE_SHAPES = (
    (2048, 128, 64), (64, 32, 128, 64), (128, 64), (128, 64),
    (2048, 64, 128), _SHAPE, (64, 32, 128, 64), (64, 32, 128, 64),
    (8192, 2048), (2048, 128, 64), (64, 32, 128, 64), _SHAPE,
    (2048, 128, 64), _SHAPE,
)
_COUNTS = Counter()
_MISSES = Counter()
_ACTIVE_MODE = "baseline"
_ACTIVE_PASS = None


def _aten_boundary(qm, ql, sin, cos, kw, tk, kl, vl, dy, memory, local, gate, uw, tu):
    """Exact normalized ATen dataflow of the real production backward boundary.

    This recipe is traced on fake tensors once to construct a structural pattern.
    Explicit operations retain every original cast, view, projection position,
    and zero-filled scatter; arbitrary algebraic equivalents are not accepted.
    It is never called to execute training or to provide a numeric GPU oracle.
    """
    a = torch.ops.aten
    cvt = torch.ops.prims.convert_element_type.default
    reshape = a.reshape.default
    head = lambda x: reshape(x, list(_SHAPE))
    sin = reshape(cvt(sin, torch.bfloat16), [1, 1, 1, 128, 64])
    cos = reshape(cvt(cos, torch.bfloat16), [1, 1, 1, 128, 64])
    zero = a.full.default(list(_SHAPE), 0, dtype=torch.bfloat16,
                          layout=torch.strided, device=qm.device, pin_memory=False)

    def rotary(value):
        lanes = reshape(a.mul.Tensor(value, sin), [4, 16, 32, 128, 32, 2])
        odd = a.slice_scatter.default(zero, a.neg.default(a.select.int(lanes, -1, 0)),
                                     4, 1, 9223372036854775807, 2)
        even = a.slice_scatter.default(zero, a.select.int(lanes, -1, 1),
                                      4, 0, 9223372036854775807, 2)
        return a.add.Tensor(a.add.Tensor(odd, even), a.mul.Tensor(value, cos))

    def tanh_backward(gradient, saved):
        gradient = cvt(gradient, torch.float32)
        saved = cvt(saved, torch.float32)
        return cvt(a.mul.Tensor(gradient, a.sub.Tensor(1, a.mul.Tensor(saved, saved))),
                   torch.bfloat16)

    q = rotary(a.add.Tensor(head(qm), head(ql)))
    kw = a.permute.default(reshape(kw, [4, 16, 32, 64, 128]), [0, 1, 2, 4, 3])
    k = rotary(a.add.Tensor(tanh_backward(kw, tk), head(kl)))
    v = head(vl)
    dy = a.permute.default(reshape(reshape(dy, [4, 2048, 2048]),
                                   [4, 16, 128, 32, 64]), [0, 1, 3, 2, 4])
    dg = a.add.Tensor(a.neg.default(a.mul.Tensor(dy, a.div.Tensor(head(memory), 8.0))),
                      a.mul.Tensor(dy, head(local)))
    gf = cvt(gate, torch.float32)
    dg = cvt(a.mul.Tensor(cvt(dg, torch.float32),
                          a.mul.Tensor(gf, a.sub.Tensor(1, gf))), torch.bfloat16)
    u = tanh_backward(head(uw), tu)

    def pack(value, clone=True):
        value = a.permute.default(value, [0, 1, 3, 2, 4])
        if clone:
            value = a.clone.default(value, memory_format=torch.contiguous_format)
        return reshape(reshape(value, [4, 2048, 32, 64]), [4, 2048, 2048])

    return a.cat.default([pack(q), pack(k), pack(v), pack(dg, False), pack(u)], 2)


@lru_cache(maxsize=1)
def _pattern():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    with FakeTensorMode():
        inputs = [torch.empty(shape, dtype=torch.float32 if name in ("sin", "cos")
                              else torch.bfloat16, device="cuda:0")
                  for name, shape in zip(_NAMES, _NATIVE_SHAPES, strict=True)]
        gm = make_fx(_aten_boundary, tracing_mode="fake")(*inputs)
    # make_fx normalizes explicit reshape to view; this hook runs after
    # Inductor's view-to-reshape normalization.
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.view.default:
            node.target = torch.ops.aten.reshape.default
    return gm


def _metadata(value):
    if not isinstance(value, torch.Tensor):
        return None
    # Symbolic/unsupported dimensions are rejected, not specialized implicitly.
    if any(type(x) is not int for x in (*value.shape, *value.stride())):
        return None
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device.type)


class _Mismatch(Exception):
    pass


def _match(candidate, pattern):
    binding, reverse = {}, {}

    def walk(actual, expected):
        if isinstance(expected, Node):
            if not isinstance(actual, Node):
                raise _Mismatch("node kind")
            if expected in binding:
                if binding[expected] is not actual:
                    raise _Mismatch("shared dataflow")
                return
            if actual in reverse:
                raise _Mismatch("distinct dataflow")
            actual_meta, expected_meta = _metadata(actual.meta.get("val")), _metadata(expected.meta.get("val"))
            if actual_meta is None or actual_meta != expected_meta:
                raise _Mismatch("tensor metadata")
            if expected.op != "placeholder":
                if actual.op != expected.op or actual.target != expected.target:
                    raise _Mismatch("operator")
                walk(actual.args, expected.args)
                walk(actual.kwargs, expected.kwargs)
            binding[expected] = actual
            reverse[actual] = expected
        elif isinstance(expected, (tuple, list)):
            if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
                raise _Mismatch("argument list")
            for av, ev in zip(actual, expected, strict=True):
                walk(av, ev)
        elif isinstance(expected, dict):
            if not isinstance(actual, dict) or set(actual) != set(expected):
                raise _Mismatch("keyword arguments")
            for key in expected:
                walk(actual[key], expected[key])
        elif isinstance(expected, torch.device):
            if not isinstance(actual, torch.device) or actual.type != expected.type:
                raise _Mismatch("device")
        elif type(actual) is not type(expected) or actual != expected:
            raise _Mismatch("literal")

    output = next(node for node in pattern.graph.nodes if node.op == "output").args[0]
    walk(candidate, output)
    inputs = tuple(binding[node] for node in pattern.graph.nodes if node.op == "placeholder")
    interior = {actual for expected, actual in binding.items() if expected.op != "placeholder"}
    retained = set()

    def retain_view(node):
        if node not in interior or node in retained:
            return
        # AOT CSE can share the exact zero-filled scatter base across cores.
        # Its complete literal recipe was matched above; preserve it while
        # another region still uses it, without retaining computed arithmetic.
        if node.target == torch.ops.aten.full.default and not node.all_input_nodes:
            retained.add(node)
            return
        if node.target not in (torch.ops.aten.reshape.default, torch.ops.aten.permute.default):
            raise _Mismatch("external intermediate consumer")
        retained.add(node)
        for parent in node.all_input_nodes:
            retain_view(parent)

    for node in interior - {candidate}:
        if any(user not in interior for user in node.users):
            # The raw output-gradient's head view also feeds the independent
            # attention/memory VJPs. Retain that metadata-only chain in place;
            # the opaque op takes its original contiguous MM result directly.
            retain_view(node)
    if len({value.meta["val"].device for value in inputs}) != 1:
        raise _Mismatch("mixed input devices")
    _validate_producers(inputs)
    return inputs, interior - retained


def _validate_producers(inputs):
    """The local branches must belong to one causal dropout-zero Flash VJP."""
    qm, ql, _sin, _cos, kw, _tk, kl, vl, dy, memory, local, _gate, uw, _tu = inputs
    flash = None
    for node, index in ((ql, 0), (kl, 1), (vl, 2)):
        if (node.op != "call_function" or node.target is not operator.getitem
                or len(node.args) != 2 or node.args[1] != index):
            raise _Mismatch("Flash gradient ordering")
        if flash is None:
            flash = node.args[0]
        elif node.args[0] is not flash:
            raise _Mismatch("separate Flash VJPs")
    if (not isinstance(flash, Node)
            or flash.target != torch.ops.aten._scaled_dot_product_flash_attention_backward.default
            or len(flash.args) != 14 or flash.args[8:12] != (128, 128, 0.0, True)
            or dict(flash.kwargs) != {"scale": 0.125} or flash.args[4] is not local):
        raise _Mismatch("Flash attention settings")
    for node in (qm, kw, uw):
        if node.op != "call_function" or node.target != torch.ops.aten.bmm.default:
            raise _Mismatch("native backward GEMM producer")
    if dy.op != "call_function" or dy.target != torch.ops.aten.mm.default:
        raise _Mismatch("native output gradient producer")


@custom_op("warpstate::paired_backward_layout", mutates_args=())
def _paired_backward_layout(q_memory: torch.Tensor, q_local: torch.Tensor,
                            sin: torch.Tensor, cos: torch.Tensor, k_write: torch.Tensor,
                            tanh_k: torch.Tensor, k_local: torch.Tensor, v_local: torch.Tensor,
                            grad_y: torch.Tensor, memory: torch.Tensor, local: torch.Tensor,
                            gate: torch.Tensor, u_write: torch.Tensor,
                            tanh_u: torch.Tensor) -> torch.Tensor:
    """Opaque pure launch using native input storage, with no layout copies.

    Fake/compile guards already require native shapes and strides. The entrypoint
    is internal to this strict rewrite; shape views and fresh output allocation
    are the only tensor operations. Runtime evidence remains valid on cache hits.
    """
    from ops.backward_layout import _paired_kernel

    args = (q_memory, q_local, sin, cos, k_write, tanh_k, k_local, v_local,
            grad_y, memory, local, gate, u_write, tanh_u)
    _validate_native_inputs(args)
    output = torch.empty((4, 2048, 10240), dtype=torch.bfloat16, device=q_memory.device)
    # All contiguous native shapes have exactly the raw layouts consumed by the
    # prototype kernel; passing native pointers needs no view or contiguous call.
    launch = (*args, output, 16777216, 32, 128, 64, 0.125, 256)
    _paired_kernel[(32768,)](*launch, True, num_warps=8, enable_fp_fusion=True)
    _paired_kernel[(65536,)](*launch, False, num_warps=8, enable_fp_fusion=True)
    _COUNTS["runtime_calls"] += 1
    return output


@_paired_backward_layout.register_fake
def _paired_backward_layout_fake(*inputs):
    _validate_native_inputs(inputs)
    return inputs[0].new_empty((4, 2048, 10240))


def _validate_native_inputs(inputs):
    for name, value, shape in zip(_NAMES, inputs, _NATIVE_SHAPES, strict=True):
        dtype = torch.float32 if name in ("sin", "cos") else torch.bfloat16
        if (tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous()
                or value.device != inputs[0].device or value.device.type != "cuda"):
            raise ValueError(f"Unsupported paired backward input {name}")


class BackwardLayoutPass(CustomGraphPass):
    def __init__(self, mode, identity):
        self.mode, self.identity = mode, identity

    def uuid(self):
        return self.identity

    def __call__(self, graph):
        _COUNTS["graphs"] += 1
        candidates = [node for node in graph.nodes if node.op == "call_function"
                      and node.target == torch.ops.aten.cat.default
                      and isinstance(node.args[0], (tuple, list)) and len(node.args[0]) == 5]
        for candidate in candidates:
            _COUNTS["candidates"] += 1
            try:
                inputs, interior = _match(candidate, _pattern())
            except _Mismatch as error:
                _MISSES[str(error)] += 1
                continue
            _COUNTS["matches"] += 1
            if self.mode == "inspect":
                continue
            with graph.inserting_before(candidate):
                replacement = graph.call_function(torch.ops.warpstate.paired_backward_layout.default, inputs)
                replacement.meta = dict(candidate.meta)
            candidate.replace_all_uses_with(replacement)
            for node in reversed(list(graph.nodes)):
                if node in interior:
                    graph.erase_node(node)
            _COUNTS["replacements"] += 1


def _identity(mode):
    directory = Path(__file__).resolve().parent
    sources = {"ops/" + name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
               for name in ("backward_layout_execution.py", "backward_layout.py")}
    metadata = {"format": "warpstate-backward-layout-v1", "mode": mode,
                "source_sha256": sources, "torch_version": torch.__version__,
                "shape": list(_SHAPE), "algorithm": "paired", "block": 256, "warps": 8}
    metadata["identity"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return metadata


def validate_backward_layout_config(cfg):
    """Reject unsupported model recipes before allocation or resume mutation."""
    required = {"d_model": 2048, "n_heads": 32, "chunk_size": 128,
                "parallel_memory": True, "dropout": 0.0}
    for name, value in required.items():
        if getattr(cfg, name, None) != value:
            raise ValueError(f"Paired backward layout requires {name}={value}")


def configure_backward_layout(mode="baseline"):
    """Configure once before compilation, returning checkpoint/benchmark identity.

    A pre-existing custom pass is rejected, never replaced or silently composed;
    composition could change the arithmetic boundary. Use a fresh process when
    changing modes, matching Compiled Autograd's process-wide configuration.
    """
    global _ACTIVE_MODE, _ACTIVE_PASS
    if mode not in BACKWARD_LAYOUT_MODES:
        raise ValueError(f"Unknown backward layout mode: {mode}")
    from torch._inductor import config
    if _ACTIVE_PASS is not None:
        raise RuntimeError("Backward layout is already configured; use a fresh process")
    if mode == "baseline":
        return None
    if config.post_grad_custom_post_pass is not None:
        raise RuntimeError("A post-gradient custom pass already exists; refusing to replace it")
    if (config.emulate_precision_casts or not config.triton.codegen_upcast_to_fp32
            or config.eager_numerics.disable_ftz):
        raise ValueError("Paired backward layout requires the profiled FP32 pointwise fusion settings")
    metadata = _identity(mode)
    _ACTIVE_MODE = mode
    _ACTIVE_PASS = BackwardLayoutPass(mode, metadata["identity"])
    config.post_grad_custom_post_pass = _ACTIVE_PASS
    return metadata


def backward_layout_stats():
    """Process-local compilation and runtime evidence; no device synchronization."""
    return {"mode": _ACTIVE_MODE, **{name: _COUNTS[name] for name in
            ("graphs", "candidates", "matches", "replacements", "runtime_calls")},
            "miss_reasons": dict(_MISSES)}


def require_backward_layout_dispatch(min_calls=1):
    if isinstance(min_calls, bool) or not isinstance(min_calls, int) or min_calls < 1:
        raise ValueError("min_calls must be a positive integer")
    if _ACTIVE_MODE != "paired" or _COUNTS["runtime_calls"] < min_calls:
        raise RuntimeError(f"Paired backward layout expected at least {min_calls} runtime calls; "
                           f"observed {backward_layout_stats()}")
    return backward_layout_stats()
