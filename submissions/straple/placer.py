"""
Straple Placer (C++ core) — Legalize + SA + LNS, fast native implementation.

Hot loops (legalize, SA refinement, LNS destroy/repair) live in
submissions/straple/cpp/placer_core.cpp and are loaded via pybind11.
Python here orchestrates: load the benchmark, build edges, run the C++
pipeline, and call the (Python) TILOS proxy_cost for LNS accept/reject.

Usage:
    uv run evaluate submissions/straple/placer.py
    uv run evaluate submissions/straple/placer.py --all

Build the native module once (or after editing the C++ source):
    submissions/straple/cpp/build.sh
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


_CPP_DIR = Path(__file__).resolve().parent / "cpp"
if str(_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(_CPP_DIR))


def _import_native_or_build():
    try:
        import _placer_core
        return _placer_core
    except ImportError:
        build_script = _CPP_DIR / "build.sh"
        if not build_script.exists():
            raise
        os.system(f"bash {build_script}")
        import _placer_core
        return _placer_core


_placer_core = _import_native_or_build()


def _load_plc(name):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        _accelerate_plc(plc)
        return plc
    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            _accelerate_plc(plc)
            return plc
    return None


def _accelerate_plc(plc):
    if not isinstance(plc.soft_macro_pin_indices, set):
        plc.soft_macro_pin_indices = set(plc.soft_macro_pin_indices)
    if not isinstance(plc.hard_macro_pin_indices, set):
        plc.hard_macro_pin_indices = set(plc.hard_macro_pin_indices)


def _extract_edges(benchmark, plc):
    n_hard = benchmark.num_hard_macros
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx
    edge_dict = {}
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            macro_list = sorted(macros)
            weight = 1.0 / (len(macro_list) - 1)
            for i in range(len(macro_list)):
                for j in range(i + 1, len(macro_list)):
                    pair = (macro_list[i], macro_list[j])
                    edge_dict[pair] = edge_dict.get(pair, 0) + weight
    if not edge_dict:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float64)
    edges = np.array(list(edge_dict.keys()), dtype=np.int32)
    weights = np.array([edge_dict[e] for e in edge_dict], dtype=np.float64)
    return edges, weights


class StraplePlacer:
    def __init__(
        self,
        seed: int = 42,
        refine_iters: int = 3000,
        lns_outer_iters: int = 30,
        lns_destroy_size: int = 8,
    ):
        self.seed = seed
        self.refine_iters = refine_iters
        self.lns_outer_iters = lns_outer_iters
        self.lns_destroy_size = lns_destroy_size

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        n_hard = benchmark.num_hard_macros
        plc = _load_plc(benchmark.name)
        if plc is not None:
            edges, edge_weights = _extract_edges(benchmark, plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int32)
            edge_weights = np.zeros(0, dtype=np.float64)

        initial_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64).copy()
        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        movable_mask = benchmark.get_movable_mask()[:n_hard].numpy().astype(np.bool_)
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        state = _placer_core.PlacerState()
        state.initialize(
            initial_pos, sizes, movable_mask,
            edges, edge_weights,
            canvas_w, canvas_h, int(self.seed),
        )

        state.legalize()
        state.sa_refine(self.refine_iters)

        if plc is not None and self.lns_outer_iters > 0:
            self._lns_loop(state, benchmark, plc, n_hard)

        final_pos = state.current_positions()
        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(final_pos, dtype=torch.float32)
        return full

    def _lns_loop(self, state, benchmark, plc, n_hard):
        def proxy_of(positions_np):
            full = benchmark.macro_positions.clone()
            full[:n_hard] = torch.tensor(positions_np, dtype=torch.float32)
            return compute_proxy_cost(full, benchmark, plc)['proxy_cost']

        best_pos = state.current_positions()
        best_cost = proxy_of(best_pos)

        for _ in range(self.lns_outer_iters):
            saved = state.current_positions()
            trial = state.destroy_and_repair(self.lns_destroy_size)
            new_cost = proxy_of(trial)
            if new_cost < best_cost:
                best_cost = new_cost
                best_pos = trial
            else:
                state.set_positions(saved)

        state.set_positions(best_pos)
