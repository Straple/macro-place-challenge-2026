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

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


_CPP_DIR = Path(__file__).resolve().parent / "cpp"
if str(_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(_CPP_DIR))


def _import_native_or_build():
    try:
        import _placer_core
        import _proxy_cost
        return _placer_core, _proxy_cost
    except ImportError:
        build_script = _CPP_DIR / "build.sh"
        if not build_script.exists():
            raise
        os.system(f"bash {build_script}")
        import _placer_core
        import _proxy_cost
        return _placer_core, _proxy_cost


_placer_core, _proxy_cost = _import_native_or_build()


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


_PIN_KIND_PORT = 0
_PIN_KIND_HARD = 1
_PIN_KIND_SOFT = 2


def _build_proxy_evaluator(benchmark, plc):
    hard_name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        hard_name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx
    soft_name_to_sidx = {}
    for sidx, idx in enumerate(plc.soft_macro_indices):
        soft_name_to_sidx[plc.modules_w_pins[idx].get_name()] = sidx
    port_name_to_pidx = {}
    for pidx, idx in enumerate(plc.port_indices):
        port_name_to_pidx[plc.modules_w_pins[idx].get_name()] = pidx

    pin_kinds = []
    pin_owners = []
    pin_offsets_x = []
    pin_offsets_y = []
    net_starts = [0]
    net_weights = []
    net_source_slots = []

    for driver, sinks in plc.nets.items():
        net_pins = [driver] + sinks
        net_pin_data = []
        source_slot = 0
        slot = 0
        for pin_name in net_pins:
            parent_name = pin_name.split("/")[0]
            pin_idx = plc.mod_name_to_indices.get(pin_name)
            if pin_idx is None:
                slot += 1
                continue
            pin_node = plc.modules_w_pins[pin_idx]
            pin_type = pin_node.get_type()
            if pin_type == "PORT":
                if parent_name not in port_name_to_pidx:
                    slot += 1
                    continue
                kind = _PIN_KIND_PORT
                owner = port_name_to_pidx[parent_name]
                offset_x = 0.0
                offset_y = 0.0
            elif pin_type == "MACRO_PIN":
                ox, oy = pin_node.get_offset()
                if parent_name in hard_name_to_bidx:
                    kind = _PIN_KIND_HARD
                    owner = hard_name_to_bidx[parent_name]
                elif parent_name in soft_name_to_sidx:
                    kind = _PIN_KIND_SOFT
                    owner = soft_name_to_sidx[parent_name]
                else:
                    slot += 1
                    continue
                offset_x = float(ox)
                offset_y = float(oy)
            else:
                slot += 1
                continue
            if pin_name == driver:
                source_slot = len(net_pin_data)
            net_pin_data.append((kind, owner, offset_x, offset_y))
            slot += 1
        if not net_pin_data:
            continue
        driver_node = plc.modules_w_pins[plc.mod_name_to_indices[driver]]
        weight = float(driver_node.get_weight())
        for kind, owner, ox, oy in net_pin_data:
            pin_kinds.append(kind)
            pin_owners.append(owner)
            pin_offsets_x.append(ox)
            pin_offsets_y.append(oy)
        net_starts.append(len(pin_kinds))
        net_weights.append(weight)
        net_source_slots.append(source_slot)

    n_hard = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros

    hard_sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    soft_sizes = benchmark.macro_sizes[n_hard:].numpy().astype(np.float64)
    soft_positions = benchmark.macro_positions[n_hard:].numpy().astype(np.float64)
    port_positions = benchmark.port_positions.numpy().astype(np.float64) if benchmark.port_positions.numel() > 0 else np.zeros((0, 2), dtype=np.float64)

    evaluator = _proxy_cost.ProxyEvaluator()
    evaluator.initialize(
        n_hard, n_soft,
        float(plc.width), float(plc.height),
        int(plc.grid_col), int(plc.grid_row),
        float(plc.hroutes_per_micron), float(plc.vroutes_per_micron),
        float(plc.hrouting_alloc), float(plc.vrouting_alloc),
        int(plc.smooth_range),
        float(plc.net_cnt) if plc.net_cnt > 0 else 1.0,
        hard_sizes, soft_sizes, soft_positions, port_positions,
        np.asarray(pin_kinds, dtype=np.int32),
        np.asarray(pin_owners, dtype=np.int32),
        np.asarray(pin_offsets_x, dtype=np.float64),
        np.asarray(pin_offsets_y, dtype=np.float64),
        np.asarray(net_starts, dtype=np.int32),
        np.asarray(net_weights, dtype=np.float64),
        np.asarray(net_source_slots, dtype=np.int32),
    )
    return evaluator


class StraplePlacer:
    def __init__(
        self,
        seed: int = 42,
        refine_iters: int = 3000,
        lns_outer_iters: int = 30,
        lns_destroy_size: int = 8,
        verbose: int = 0,
        analytical_steps: int = 0,
        analytical_lr: float = 0.3,
        analytical_lambda_density: float = 50000.0,
        analytical_target_util: float = 0.2,
    ):
        self.seed = seed
        self.refine_iters = refine_iters
        self.lns_outer_iters = lns_outer_iters
        self.lns_destroy_size = lns_destroy_size
        self.verbose = verbose if verbose else int(os.environ.get("STRAPLE_VERBOSE", "0"))
        self.analytical_steps = analytical_steps
        self.analytical_lr = analytical_lr
        self.analytical_lambda_density = analytical_lambda_density
        self.analytical_target_util = analytical_target_util

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        import time
        t_place_start = time.time()
        bench_label = getattr(benchmark, "name", "?")
        self._log(f"[{bench_label}] === StraplePlacer.place() ===")

        n_hard = benchmark.num_hard_macros
        plc = _load_plc(benchmark.name)
        t_after_plc = time.time()
        self._log(f"[{bench_label}] _load_plc: {t_after_plc - t_place_start:.2f}s "
                  f"(n_hard={n_hard})")

        if plc is not None:
            edges, edge_weights = _extract_edges(benchmark, plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int32)
            edge_weights = np.zeros(0, dtype=np.float64)
        t_after_edges = time.time()
        self._log(f"[{bench_label}] _extract_edges: {t_after_edges - t_after_plc:.2f}s "
                  f"(edges={len(edges)})")

        initial_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64).copy()
        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        movable_mask = benchmark.get_movable_mask()[:n_hard].numpy().astype(np.bool_)
        num_movable = int(movable_mask.sum())
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        self._log(f"[{bench_label}] num_movable={num_movable} canvas={canvas_w:.0f}x{canvas_h:.0f}")

        analytical_pos = None
        if plc is not None and self.analytical_steps > 0:
            _STRAPLE_DIR = str(Path(__file__).resolve().parent)
            if _STRAPLE_DIR not in sys.path:
                sys.path.insert(0, _STRAPLE_DIR)
            from analytical_seed import analytical_seed
            t0 = time.time()
            analytical_pos = analytical_seed(
                benchmark, plc,
                num_steps=self.analytical_steps,
                lr=self.analytical_lr,
                lambda_density=self.analytical_lambda_density,
                target_util=self.analytical_target_util,
                log_every=(max(1, self.analytical_steps // 5) if self.verbose else 0),
                log_proxy=bool(self.verbose),
                label=bench_label,
            )
            self._log(f"[{bench_label}] analytical_seed({self.analytical_steps} steps): "
                      f"{time.time()-t0:.2f}s")

        num_orig_starts = 3 if num_movable >= 300 else 1
        num_starts = num_orig_starts + (1 if analytical_pos is not None else 0)

        evaluator = None
        if plc is not None and self.lns_outer_iters > 0:
            t0 = time.time()
            evaluator = _build_proxy_evaluator(benchmark, plc)
            self._log(f"[{bench_label}] _build_proxy_evaluator: {time.time()-t0:.2f}s")

        best_pos = None
        best_cost = float("inf")

        for start_idx in range(num_starts):
            is_analytical_start = (analytical_pos is not None
                                   and start_idx == num_starts - 1)
            seed = self.seed + start_idx
            t_start_iter = time.time()
            state = _placer_core.PlacerState()
            seed_pos = analytical_pos if is_analytical_start else initial_pos
            state.initialize(
                seed_pos, sizes, movable_mask,
                edges, edge_weights,
                canvas_w, canvas_h, int(seed),
            )

            t0 = time.time()
            state.legalize()
            t_legalize = time.time() - t0
            cost_after_legal = evaluator.evaluate(state.current_positions()) if evaluator else 0.0

            sa_iters_to_run = self.refine_iters if (num_movable < 300 and not is_analytical_start) else 0
            t0 = time.time()
            if self.verbose and sa_iters_to_run > 0:
                snap = max(1, sa_iters_to_run // 10)
                sa_stats = state.sa_refine_with_stats(sa_iters_to_run, snap)
            elif sa_iters_to_run > 0:
                state.sa_refine(sa_iters_to_run)
                sa_stats = None
            else:
                sa_stats = None
            t_sa = time.time() - t0
            cost_after_sa = evaluator.evaluate(state.current_positions()) if evaluator else 0.0

            self._log(f"[{bench_label}] start#{start_idx} seed={seed}: "
                      f"legalize={t_legalize:.2f}s proxy={cost_after_legal:.4f} | "
                      f"sa_refine({sa_iters_to_run})={t_sa*1000:.1f}ms proxy={cost_after_sa:.4f} "
                      f"(delta {cost_after_sa-cost_after_legal:+.4f})")
            if sa_stats and not sa_stats.get("skipped", True):
                acc = sa_stats["num_accepted"]
                rej = sa_stats["num_rejected"]
                bolt = sa_stats["num_accepted_boltzmann"]
                rej_ovr = sa_stats["num_rejected_overlap"]
                ni = sa_stats["num_iters"]
                self._log(f"[{bench_label}] start#{start_idx} SA stats: "
                          f"iters={ni} acc={acc} ({acc*100//ni}%) "
                          f"acc_boltzmann={bolt} ({bolt*100//max(acc,1)}% of acc) "
                          f"rej={rej} (rej_overlap={rej_ovr}) "
                          f"shift={sa_stats['num_shift']} swap={sa_stats['num_swap']} "
                          f"toward={sa_stats['num_toward_neighbor']} "
                          f"WL: init={sa_stats['initial_wl']:.4f} -> "
                          f"best@step{sa_stats['best_step']}={sa_stats['best_wl']:.4f} "
                          f"final={sa_stats['final_wl']:.4f} "
                          f"T: {sa_stats['t_start']:.2f}->{sa_stats['t_end']:.4f}")
                if "trajectory_steps" in sa_stats:
                    traj_s = sa_stats["trajectory_steps"]
                    traj_w = sa_stats["trajectory_wl"]
                    snapshots_str = " ".join(
                        f"step{s}={w:.4f}" for s, w in zip(traj_s, traj_w))
                    self._log(f"[{bench_label}] start#{start_idx} SA trajectory: {snapshots_str}")

            if evaluator is not None:
                t0 = time.time()
                lns_log = self._lns_loop(state, evaluator, plc, num_movable, bench_label, start_idx)
                t_lns = time.time() - t0
                self._log(f"[{bench_label}] start#{start_idx} LNS: {t_lns:.2f}s "
                          f"iters={lns_log['iters']} accepted={lns_log['accepted']} "
                          f"cost {cost_after_sa:.4f} -> {lns_log['final_cost']:.4f} "
                          f"(delta {lns_log['final_cost']-cost_after_sa:+.4f})")

            trial_pos = state.current_positions()
            if evaluator is not None:
                trial_cost = evaluator.evaluate(trial_pos)
            else:
                trial_cost = 0.0 if num_starts == 1 else float(start_idx)

            if trial_cost < best_cost:
                best_cost = trial_cost
                best_pos = trial_pos
                self._log(f"[{bench_label}] start#{start_idx} new best={trial_cost:.4f} "
                          f"(iter_total={time.time()-t_start_iter:.2f}s)")
            else:
                self._log(f"[{bench_label}] start#{start_idx} cost={trial_cost:.4f} "
                          f"(not best, current best={best_cost:.4f})")

        if evaluator is not None and best_pos is not None:
            t0 = time.time()
            state2 = _placer_core.PlacerState()
            state2.initialize(
                best_pos, sizes, movable_mask,
                edges, edge_weights,
                canvas_w, canvas_h, int(self.seed + 9999),
            )
            lns2_log = self._lns_loop(state2, evaluator, plc, num_movable,
                                      bench_label, 99)
            refined_pos = state2.current_positions()
            refined_cost = evaluator.evaluate(refined_pos)
            self._log(f"[{bench_label}] REFINE pass: {time.time()-t0:.2f}s "
                      f"cost {best_cost:.4f} -> {refined_cost:.4f} "
                      f"(delta {refined_cost-best_cost:+.4f})")
            if refined_cost < best_cost:
                best_cost = refined_cost
                best_pos = refined_pos

        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        self._log(f"[{bench_label}] === DONE total={time.time()-t_place_start:.2f}s "
                  f"final_cost={best_cost:.4f} ===")
        return full

    def _lns_loop(self, state, evaluator, plc, num_movable, bench_label="?", start_idx=0):
        import time
        best_pos = state.current_positions()
        best_cost = evaluator.evaluate(best_pos)
        initial_cost = best_cost

        grid_rows = int(plc.grid_row)
        grid_cols = int(plc.grid_col)
        congested_percent = 0.05

        adaptive_destroy = max(self.lns_destroy_size, min(16, math.ceil(0.025 * num_movable)))
        adaptive_outer = max(self.lns_outer_iters, min(50000, math.ceil(60.0 * num_movable)))

        accepted = 0
        accepted_random = 0
        accepted_congested = 0
        gain_random = 0.0
        gain_congested = 0.0
        log_step = max(1, adaptive_outer // 10) if self.verbose else 0

        accepted_swap = 0
        gain_swap = 0.0
        accepted_cluster = 0
        gain_cluster = 0.0
        no_improve_count = 0
        early_term_threshold = max(500, adaptive_outer // 5)

        ops = ["rand", "cong", "swap", "cluster"]
        op_weights = {o: 1.0 for o in ops}
        op_attempts = {o: 0 for o in ops}
        op_accepts = {o: 0 for o in ops}
        op_gains = {o: 0.0 for o in ops}
        rng = np.random.default_rng(self.seed + start_idx * 1000)
        warmup = min(40, adaptive_outer // 10)
        weight_decay = 0.95
        weight_reaction = 0.3

        shake_threshold = max(200, adaptive_outer // 20)
        shake_count = 0

        for iteration in range(adaptive_outer):
            saved = state.current_positions()
            t_iter = time.time()
            if iteration < warmup:
                op = ops[iteration % len(ops)]
            else:
                total_w = sum(op_weights.values())
                r = rng.random() * total_w
                acc = 0.0
                op = ops[-1]
                for o in ops:
                    acc += op_weights[o]
                    if r <= acc:
                        op = o
                        break
            op_attempts[op] += 1

            if op == "rand":
                trial = state.destroy_and_repair(adaptive_destroy)
            elif op == "cong":
                evaluator.evaluate(saved)
                hot_cells = evaluator.get_top_congested_cells(congested_percent)
                if hot_cells.shape[0] > 0:
                    cong_grid = evaluator.get_congestion_grid()
                    trial = state.destroy_congested_and_repair(
                        hot_cells, grid_rows, grid_cols, adaptive_destroy,
                        cong_grid, grid_rows, grid_cols,
                    )
                else:
                    trial = state.destroy_and_repair(adaptive_destroy)
            elif op == "swap":
                trial = state.swap_two_macros(max(2, adaptive_destroy // 2))
            else:
                trial = state.destroy_cluster_and_repair(adaptive_destroy)
            new_cost = evaluator.evaluate(trial)
            if new_cost < best_cost:
                gain = best_cost - new_cost
                best_cost = new_cost
                best_pos = trial
                accepted += 1
                no_improve_count = 0
                op_accepts[op] += 1
                op_gains[op] += gain
                op_weights[op] = op_weights[op] * weight_decay + (
                    weight_reaction * (1.0 + 100.0 * gain))
                if op == "rand":
                    accepted_random += 1
                    gain_random += gain
                elif op == "swap":
                    accepted_swap += 1
                    gain_swap += gain
                elif op == "cluster":
                    accepted_cluster += 1
                    gain_cluster += gain
                else:
                    accepted_congested += 1
                    gain_congested += gain
                if log_step and (iteration % log_step == 0 or iteration == adaptive_outer - 1):
                    self._log(f"[{bench_label}] start#{start_idx} LNS iter={iteration:>3} "
                              f"op={op:<7} ACCEPT cost={new_cost:.4f} (-{gain:.4f}) "
                              f"k={adaptive_destroy} t={(time.time()-t_iter)*1000:.0f}ms")
            else:
                state.set_positions(saved)
                no_improve_count += 1
                op_weights[op] = op_weights[op] * weight_decay
                if op_weights[op] < 0.05:
                    op_weights[op] = 0.05
                if log_step and iteration % (log_step * 2) == 0:
                    self._log(f"[{bench_label}] start#{start_idx} LNS iter={iteration:>3} "
                              f"op={op:<7} reject (cost={new_cost:.4f} > {best_cost:.4f})")
                if no_improve_count >= shake_threshold and shake_count < 10:
                    state.set_positions(best_pos)
                    shake_k = min(num_movable // 4 + shake_count * 10, max(20, num_movable // 2))
                    state.swap_two_macros(shake_k)
                    state.destroy_and_repair(shake_k)
                    no_improve_count = 0
                    shake_count += 1
                    op_weights = {o: 1.0 for o in ops}
                    if self.verbose:
                        self._log(f"[{bench_label}] start#{start_idx} LNS SHAKE-UP @ iter={iteration} "
                                  f"k={shake_k} count={shake_count}")
                    continue
                if no_improve_count >= early_term_threshold:
                    if self.verbose:
                        self._log(f"[{bench_label}] start#{start_idx} LNS early term @ iter={iteration} "
                                  f"({no_improve_count} consec rejects)")
                    break

        state.set_positions(best_pos)
        if self.verbose:
            ops_summary = " | ".join(
                f"{o}: att={op_attempts[o]} acc={op_accepts[o]} "
                f"gain={op_gains[o]:.4f} w={op_weights[o]:.2f}"
                for o in ops
            )
            self._log(f"[{bench_label}] start#{start_idx} LNS ALNS: {ops_summary}")
            self._log(f"[{bench_label}] start#{start_idx} LNS summary: "
                      f"{accepted}/{adaptive_outer} accepted "
                      f"k={adaptive_destroy} initial={initial_cost:.4f} -> final={best_cost:.4f}")
        return {"iters": adaptive_outer, "accepted": accepted, "final_cost": best_cost}
