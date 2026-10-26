"""Optional CUDA graph replay evidence for disposable benchmarks.

Uses private PyTorch 2.14 APIs, imported only when enabled. Counts describe host
capture/replay calls, not GPU durations or kernel coverage. No CUDA work or
synchronization is added. Use only one context at a time in a benchmark process.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Iterator


@dataclass
class CudagraphDiagnostics:
    _replays: Counter[int]
    _read_counters: Callable[[], dict[str, int]]
    _initial_counters: dict[str, int]

    def snapshot(self) -> dict:
        """Return cumulative counts since context entry; subtract for an interval."""
        current = self._read_counters()
        return {
            **{key: value - self._initial_counters[key] for key, value in current.items()},
            "replay_calls": sum(self._replays.values()),
            "replay_calls_by_graph_id": {
                str(key): value for key, value in sorted(self._replays.items())
            },
        }


@contextmanager
def collect_cudagraph_diagnostics() -> Iterator[CudagraphDiagnostics]:
    """Count successful, non-kernel-free tree replays and restore the wrapper.

    ``capture_calls`` counts entries in PyTorch's timed recording metrics.
    ``cudagraph_skips`` and ``recorded_non_static_inputs`` are compiler counter
    deltas. These alone do not establish replay. ``replay_calls`` increments
    after a real ``CUDAGraphNode.run_graph`` returns successfully; its initial
    constructor replay is outside this method and is deliberately excluded.
    Existing benchmark synchronization establishes completion of queued work.
    """
    try:
        trees = import_module("torch._inductor.cudagraph_trees")
        utils = import_module("torch._dynamo.utils")
        node_type = trees.CUDAGraphNode
        original = node_type.run_graph
        if not callable(original):
            raise AttributeError("CUDAGraphNode.run_graph is not callable")
        # Validate availability before changing any process-wide method.
        utils.compilation_time_metrics.get("CUDAGraphNode.record", [])
        utils.counters.get("inductor", {})
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "CUDA graph diagnostics require the private PyTorch 2.14 "
            "CUDAGraphNode and compiler counter APIs"
        ) from error

    def read_counters() -> dict[str, int]:
        inductor = utils.counters.get("inductor", {})
        return {
            "capture_calls": len(utils.compilation_time_metrics.get("CUDAGraphNode.record", [])),
            "cudagraph_skips": int(inductor.get("cudagraph_skips", 0)),
            "recorded_non_static_inputs": int(inductor.get("cudagraph_recorded_non_static_inputs", 0)),
        }

    replays: Counter[int] = Counter()
    diagnostics = CudagraphDiagnostics(replays, read_counters, read_counters())

    def counted_run_graph(node):
        try:
            kernel_free = node.wrapped_function.kernel_free_cudagraph
            graph_id = node.id.id
        except AttributeError as error:
            raise RuntimeError("CUDA graph diagnostics encountered an unsupported node API") from error
        result = original(node)
        if not kernel_free:
            replays[graph_id] += 1
        return result

    node_type.run_graph = counted_run_graph
    try:
        yield diagnostics
    finally:
        node_type.run_graph = original
