"""Structural and configuration guards for the opt-in backward graph rewrite.

The pattern fixtures isolate graph safety; actual compiled model numerical and
continuation checks live in the separate integration validation harness.
"""

import copy
import hashlib
import operator
import os
from pathlib import Path

import pytest
import torch
from torch._inductor import config

from ops import backward_layout_execution as layout
from warpstate.config import WarpStateConfig


@pytest.fixture(autouse=True)
def restore_configuration(monkeypatch):
    monkeypatch.setattr(layout, "_ACTIVE_PASS", None)
    monkeypatch.setattr(layout, "_ACTIVE_MODE", "baseline")
    monkeypatch.setattr(layout, "_COUNTS", layout.Counter())
    monkeypatch.setattr(layout, "_MISSES", layout.Counter())
    with config.patch({"post_grad_custom_post_pass": None, "emulate_precision_casts": False,
                       "triton.codegen_upcast_to_fp32": True, "eager_numerics.disable_ftz": False}):
        yield


def candidate_graph():
    """Attach explicit native GEMM and ordered Flash VJP boundary producers."""
    gm = copy.deepcopy(layout._pattern())
    graph = gm.graph
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    values = dict(zip(layout._NAMES, placeholders, strict=True))
    first = next(node for node in graph.nodes if node.op == "call_function")
    with graph.inserting_before(first):
        flash = graph.call_function(torch.ops.aten._scaled_dot_product_flash_attention_backward.default,
                                    (None, None, None, None, values["local"], None, None, None,
                                     128, 128, 0.0, True, None, None), {"scale": 0.125})
        for name, index in (("q_local", 0), ("k_local", 1), ("v_local", 2)):
            old = values[name]
            new = graph.call_function(operator.getitem, (flash, index))
            new.meta = dict(old.meta)
            old.replace_all_uses_with(new)
        for name in ("q_memory", "k_write", "u_write", "grad_y"):
            old = values[name]
            users = list(old.users)
            target = torch.ops.aten.mm.default if name == "grad_y" else torch.ops.aten.bmm.default
            new = graph.call_function(target, (old, old))
            new.meta = dict(old.meta)
            for user in users:
                user.replace_input_with(old, new)
    graph.lint()
    root = next(node for node in graph.nodes if node.op == "output").args[0]
    return gm, root, flash


def test_pattern_is_independent_of_node_names_and_preserves_native_inputs():
    gm, root, _ = candidate_graph()
    for index, node in enumerate(gm.graph.nodes):
        node.name = f"renamed_{index}"
    inputs, interior = layout._match(root, layout._pattern())
    assert len(inputs) == 14
    assert len(interior) > 70
    assert tuple(inputs[4].meta["val"].shape) == (2048, 64, 128)
    assert tuple(inputs[8].meta["val"].shape) == (8192, 2048)


def test_inspection_retains_graph_and_replacement_is_pure():
    gm, root, _ = candidate_graph()
    original = str(gm.graph)
    layout.BackwardLayoutPass("inspect", "inspect-test")(gm.graph)
    assert str(gm.graph) == original
    assert layout.backward_layout_stats()["matches"] == 1
    layout.BackwardLayoutPass("paired", "paired-test")(gm.graph)
    gm.graph.lint()
    calls = [node for node in gm.graph.nodes
             if node.target == torch.ops.warpstate.paired_backward_layout.default]
    assert len(calls) == 1
    assert root not in gm.graph.nodes
    assert len(calls[0].args) == 14
    assert layout.backward_layout_stats()["replacements"] == 1
    assert layout.backward_layout_stats()["runtime_calls"] == 0


@pytest.mark.parametrize("mutation", ["gate_scale", "projection_order", "rotary_lane", "rounding",
                                      "dtype", "stride", "dropout", "flash_scale", "flash_order",
                                      "input_producer", "missing_meta", "mixed_devices"])
def test_unsupported_graph_is_left_unchanged(mutation):
    gm, root, flash = candidate_graph()
    nodes = list(gm.graph.nodes)
    if mutation == "gate_scale":
        node = next(n for n in nodes if n.target == torch.ops.aten.div.Tensor)
        node.args = (node.args[0], 16.0)
    elif mutation == "projection_order":
        projections = list(root.args[0]); projections[0], projections[1] = projections[1], projections[0]
        root.args = (projections, 2)
    elif mutation == "rotary_lane":
        node = next(n for n in nodes if n.target == torch.ops.aten.select.int)
        node.args = (*node.args[:-1], 1 - node.args[-1])
    elif mutation == "rounding":
        node = next(n for n in nodes if n.target == torch.ops.prims.convert_element_type.default)
        node.args = (node.args[0], torch.float32)
    elif mutation in ("dtype", "stride", "missing_meta"):
        if mutation == "missing_meta":
            root.meta.pop("val")
        elif mutation == "dtype":
            root.meta["val"] = root.meta["val"].float()
        else:
            root.meta["val"] = root.meta["val"].transpose(1, 2).contiguous().transpose(1, 2)
            assert tuple(root.meta["val"].shape) == (4, 2048, 10240)
    elif mutation == "mixed_devices":
        from torch._subclasses.fake_tensor import FakeTensor

        node = next(n for n in nodes if n.target == torch.ops.aten.bmm.default)
        value = node.meta["val"]
        node.meta["val"] = FakeTensor(value.fake_mode,
                                      torch.empty_strided(value.shape, value.stride(),
                                                          dtype=value.dtype, device="meta"),
                                      torch.device("cuda:1"))
    elif mutation == "dropout":
        args = list(flash.args); args[10] = 0.1; flash.args = tuple(args)
    elif mutation == "flash_scale":
        flash.kwargs = {"scale": 0.25}
    elif mutation == "flash_order":
        node = next(n for n in nodes if n.target is operator.getitem)
        node.args = (flash, 2)
    else:
        node = next(n for n in nodes if n.target == torch.ops.aten.bmm.default)
        node.target = torch.ops.aten.mm.default
    original = str(gm.graph)
    layout.BackwardLayoutPass("paired", "test")(gm.graph)
    assert str(gm.graph) == original
    assert layout.backward_layout_stats()["replacements"] == 0
    assert sum(layout.backward_layout_stats()["miss_reasons"].values()) == 1


def test_external_arithmetic_consumer_rejected():
    gm, root, _ = candidate_graph()
    node = next(n for n in gm.graph.nodes if n.target == torch.ops.aten.add.Tensor)
    with gm.graph.inserting_before(root):
        gm.graph.call_function(torch.ops.aten.neg.default, (node,))
    with pytest.raises(layout._Mismatch, match="external intermediate consumer"):
        layout._match(root, layout._pattern())


def test_shared_gradient_view_retained_without_copy():
    gm, root, _ = candidate_graph()
    node = next(n for n in gm.graph.nodes if n.target == torch.ops.aten.permute.default
                and tuple(n.meta["val"].stride()) == (4194304, 262144, 64, 2048, 1))
    with gm.graph.inserting_before(root):
        external = gm.graph.call_function(torch.ops.aten.neg.default, (node,))
    _, interior = layout._match(root, layout._pattern())
    assert node not in interior
    layout.BackwardLayoutPass("paired", "test")(gm.graph)
    assert external.args[0] is node
    gm.graph.lint()


def test_cse_shared_exact_zero_across_two_regions():
    from torch._functorch.compile_utils import fx_graph_cse

    graph = torch.fx.Graph()
    regions = [candidate_graph(), candidate_graph()]
    mappings = []
    for index, (gm, _, _) in enumerate(regions):
        mapping = {}
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                new = graph.placeholder(f"core_{index}_{node.name}")
                new.meta = dict(node.meta)
                mapping[node] = new
        mappings.append(mapping)
    roots = []
    for (gm, root, _), mapping in zip(regions, mappings, strict=True):
        for node in gm.graph.nodes:
            if node.op not in ("placeholder", "output"):
                mapping[node] = graph.node_copy(node, lambda value: mapping[value])
        roots.append(mapping[root])
    graph.output(tuple(roots))
    graph = fx_graph_cse(graph)
    zeros = [node for node in graph.nodes if node.target == torch.ops.aten.full.default]
    assert len(zeros) == 1 and len(zeros[0].users) == 8
    layout.BackwardLayoutPass("paired", "test")(graph)
    graph.lint()
    assert sum(node.target == torch.ops.warpstate.paired_backward_layout.default
               for node in graph.nodes) == 2
    assert not any(node.target == torch.ops.aten.full.default for node in graph.nodes)
    assert layout.backward_layout_stats()["replacements"] == 2


def test_configuration_identity_covers_sources():
    result = layout.configure_backward_layout("paired")
    assert config.post_grad_custom_post_pass.uuid() == result["identity"]
    for name, digest in result["source_sha256"].items():
        assert hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() == digest
    with pytest.raises(RuntimeError, match="already configured"):
        layout.configure_backward_layout("paired")


def test_configuration_rejects_existing_pass_and_precision_changes():
    existing = lambda graph: None
    with config.patch(post_grad_custom_post_pass=existing):
        with pytest.raises(RuntimeError, match="already exists"):
            layout.configure_backward_layout("paired")
        assert config.post_grad_custom_post_pass is existing
    for setting, value in (("emulate_precision_casts", True), ("triton.codegen_upcast_to_fp32", False),
                           ("eager_numerics.disable_ftz", True)):
        with config.patch({setting: value}):
            with pytest.raises(ValueError, match="FP32 pointwise"):
                layout.configure_backward_layout("paired")


def test_baseline_has_no_compiler_hook_or_metadata():
    assert layout.configure_backward_layout() is None
    assert config.post_grad_custom_post_pass is None


def test_dispatch_evidence_requires_actual_runtime_calls():
    layout.configure_backward_layout("paired")
    layout._COUNTS["replacements"] = 100
    with pytest.raises(RuntimeError, match="runtime calls"):
        layout.require_backward_layout_dispatch()
    layout._COUNTS["runtime_calls"] = 2
    assert layout.require_backward_layout_dispatch(2)["runtime_calls"] == 2
    with pytest.raises(RuntimeError, match="at least 3"):
        layout.require_backward_layout_dispatch(3)
    with pytest.raises(ValueError):
        layout.require_backward_layout_dispatch(True)


def test_native_entry_rejects_unsupported_inputs_before_launch():
    with pytest.raises(ValueError, match="q_memory"):
        layout._validate_native_inputs([torch.empty(1)] * 14)


@pytest.mark.skipif(os.environ.get("WARPSTATE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
                    reason="requires an explicitly reserved CUDA device")
def test_opaque_boundary_schema_and_fake_registration():
    from ops.backward_layout import make_inputs

    saved = make_inputs(device="cuda", seed=9221)
    native = tuple(value.view(shape) for value, shape in
                   zip(saved, layout._NATIVE_SHAPES, strict=True))
    # This is a backward-only internal operator; schema purity and accurate fake
    # metadata apply, whereas registering another autograd VJP would be wrong.
    result = torch.library.opcheck(torch.ops.warpstate.paired_backward_layout.default, native,
                                   test_utils=("test_schema", "test_faketensor"))
    assert set(result.values()) == {"SUCCESS"}


@pytest.mark.parametrize("field,value", [("d_model", 1024), ("n_heads", 16), ("chunk_size", 64),
                                        ("parallel_memory", False), ("dropout", 0.1)])
def test_model_configuration_guard(field, value):
    cfg = WarpStateConfig(d_model=2048, n_heads=32, chunk_size=128, parallel_memory=True)
    layout.validate_backward_layout_config(cfg)
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        layout.validate_backward_layout_config(cfg)
