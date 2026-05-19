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


def _np_isfinite_any(arr):
    import numpy as _np
    return bool(_np.isfinite(arr).any())


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


def _build_proxy_evaluator(benchmark, plc, soft_positions_override=None):
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
    if soft_positions_override is not None:
        soft_positions = np.asarray(soft_positions_override, dtype=np.float64)
        if soft_positions.shape != (n_soft, 2):
            raise ValueError(
                f"soft_positions_override shape {soft_positions.shape} != ({n_soft}, 2)"
            )
    else:
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


def _worker_run_start(start_idx, args, seed_base, refine_iters, lns_outer_iters,
                      lns_destroy_size, benchmark, plc):
    placer = StraplePlacer(
        seed=seed_base,
        refine_iters=refine_iters,
        lns_outer_iters=lns_outer_iters,
        lns_destroy_size=lns_destroy_size,
        verbose=0,
        analytical_steps=0,
    )
    evaluator = _build_proxy_evaluator(
        benchmark, plc,
        soft_positions_override=args.get("gradient_soft_pos"),
    )
    return placer._run_one_start(start_idx, args, evaluator, plc)


class StraplePlacer:
    def __init__(
        self,
        seed: int = 42,
        refine_iters: int = 3000,
        lns_outer_iters: int = 30,
        lns_destroy_size: int = 8,
        verbose: int = 0,
        analytical_steps: int = None,
        analytical_lr: float = 0.3,
        analytical_lambda_density: float = 50000.0,
        analytical_target_util: float = 0.2,
        analytical_lambda_schedule=None,
        analytical_gamma_schedule=None,
        analytical_gamma_frac: float = 0.05,
    ):
        if analytical_steps is None:
            analytical_steps = int(os.environ.get("STRAPLE_ANALYTICAL_STEPS", "0"))
        self.seed = seed
        self.refine_iters = refine_iters
        self.lns_outer_iters = lns_outer_iters
        self.lns_destroy_size = lns_destroy_size
        self.verbose = verbose if verbose else int(os.environ.get("STRAPLE_VERBOSE", "0"))
        self.analytical_steps = analytical_steps
        self.analytical_lr = analytical_lr
        self.analytical_lambda_density = analytical_lambda_density
        self.analytical_target_util = analytical_target_util
        self.analytical_lambda_schedule = analytical_lambda_schedule
        self.analytical_gamma_schedule = analytical_gamma_schedule
        self.analytical_gamma_frac = analytical_gamma_frac
        env_preset = os.environ.get("STRAPLE_ANALYTICAL_PRESET", "")
        if env_preset == "cold_start":
            self.analytical_lambda_schedule = [(0.0, 0.0), (0.2, 100.0),
                                               (0.5, 5000.0), (1.0, 50000.0)]
            self.analytical_gamma_schedule = [(0.0, 1.5), (1.0, 0.3)]

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _estimate_peak_per_K_mb(self, n_total, n_hard, num_nets, max_pins,
                                  grid_rows, grid_cols, eplace_grid):
        """Estimate per-seed peak GPU memory (MB) of one gradient_batch step.

        Calibrated against a known operating point — K=384 with ibm01
        (n_total=1140, n_hard=246, num_nets=5747, max_pins=18,
        grid_rows=41, grid_cols=45, eplace_grid=128) consumed ≈9.3 GB peak
        on T4 (≈25 MB/seed).  Calibrated multiplier captures autograd
        doubling, optimizer state, temp tensors, and FFT scratch.
        """
        f4 = 4.0     # bytes per fp32 cell
        bytes_per_K = (
            # pin_xy [num_nets, max_pins, 2] (forward) + autograd dup
            num_nets * max_pins * 2 * f4 * 2.0
            # density grid + scatter buffers (multi-phase ε)
            + grid_rows * grid_cols * f4 * 6.0
            # ePlace FFT chunked (complex64 = 8B)
            + eplace_grid * eplace_grid * 8.0 * 2.0
            # pos + Adam state (exp_avg, exp_avg_sq) + grad
            + n_total * 2 * f4 * 4.0
            # overlap pair tables [n_hard, n_hard] (diff_x, diff_y, ovlap_x/y)
            + (n_hard * n_hard) * f4 * 4.0
            # congestion smooth bbox in_x/in_y (when cong_w>0)
            + num_nets * (grid_rows + grid_cols) * f4 * 2.0
        )
        # Empirical multiplier — covers autograd doubling, optimizer state,
        # ephemeral 4D density tensors, FFT scratch, framework overhead.
        # Calibrated against an OOM at K=448 on T4 (where peak hit ≈11 GB
        # on the cell_density allocation step).  5.5× lands at ≈33 MB/seed.
        multiplier = 5.5
        return (bytes_per_K * multiplier) / (1024 ** 2)

    def _place_gradient_batch(self, benchmark, plc, bench_label):
        """Submission entry: K parallel gradient seeds with GPU proxy fitness.

        K is sized from free VRAM and the per-seed memory footprint of
        gradient_batch (pin_xy + overlap pair-table + ePlace FFT + density
        + autograd doubling).  See ``_estimate_peak_per_K_mb`` for the model.

        ENV overrides:
            STRAPLE_BATCH_K              — fixed K (skips the auto-sizer)
            STRAPLE_BATCH_K_MIN          — minimum K (default 32)
            STRAPLE_BATCH_K_MAX          — upper cap (default 1024)
            STRAPLE_BATCH_VRAM_SAFETY_GB — reserve free GB (default 1.5)
            STRAPLE_BATCH_TIME_BUDGET    — gradient budget (sec, default 3000)
            STRAPLE_BATCH_LEGALIZE_TOPN  — legalize top-N candidates only
        """
        import time as _time
        import multiprocessing as _mp
        _STRAPLE_DIR = str(Path(__file__).resolve().parent)
        if _STRAPLE_DIR not in sys.path:
            sys.path.insert(0, _STRAPLE_DIR)

        from gradient_batch import gradient_batch
        from analytical_seed import (_build_net_pin_tensors_full,
                                      _build_padded_net_tensors)
        from gpu_proxy import (build_routing_edges_full,
                                build_smooth_matrices,
                                build_routing_consts, build_wl_pkg_full)
        from macro_place.objective import compute_proxy_cost

        n_hard = benchmark.num_hard_macros
        n_soft = benchmark.num_soft_macros
        n_total = benchmark.num_macros
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)

        # ---- Adaptive K from free VRAM ----
        # 55 min default — judging TL is 1 h per bench; we leave 5 min for
        # probe (~10-30 s), legalize-all (3-30 s), GPU proxy select (1 s)
        # and the evaluator's own compute_proxy_cost reporting (a few s).
        time_budget = float(os.environ.get(
            "STRAPLE_BATCH_TIME_BUDGET", "3300"))
        legalize_topn = int(os.environ.get(
            "STRAPLE_BATCH_LEGALIZE_TOPN", "0"))   # 0 = all

        # Probe net structure to estimate per-K cost accurately.
        net_macro_idx, net_pin_offsets = _build_net_pin_tensors_full(
            benchmark, plc)
        padded_probe = _build_padded_net_tensors(
            net_macro_idx, net_pin_offsets)
        if padded_probe is not None:
            num_nets_probe = int(padded_probe[0].shape[0])
            max_pins_probe = int(padded_probe[0].shape[1])
        else:
            num_nets_probe = max(1, len(plc.nets))
            max_pins_probe = 18
        ep_n = int(os.environ.get("STRAPLE_BATCH_EPLACE_GRID", "128"))
        grid_rows = int(benchmark.grid_rows)
        grid_cols = int(benchmark.grid_cols)

        peak_per_K_mb = self._estimate_peak_per_K_mb(
            n_total=n_total, n_hard=n_hard,
            num_nets=num_nets_probe, max_pins=max_pins_probe,
            grid_rows=grid_rows, grid_cols=grid_cols, eplace_grid=ep_n)

        K_min = int(os.environ.get("STRAPLE_BATCH_K_MIN", "32"))
        # Upper cap is intentionally generous — probe handles VRAM ceiling.
        K_max = int(os.environ.get("STRAPLE_BATCH_K_MAX", "8192"))
        # Initial conservative safety used for the *probe* — large enough
        # that the probe itself never OOMs even on a poorly-calibrated GPU.
        probe_safety_gb = float(os.environ.get(
            "STRAPLE_BATCH_VRAM_SAFETY_GB", "1.5"))
        # Final fill fraction: how close to total VRAM the real run is
        # allowed to push.  0.80 leaves ~20 % margin to cover the gap
        # between a short probe and the long real run (phase transitions,
        # lambda ramp, FFT plan re-allocation, allocator fragmentation).
        fill_frac = float(os.environ.get(
            "STRAPLE_BATCH_VRAM_FILL_FRAC", "0.80"))
        # Probe runs a small fixed number of gradient steps; 4 is enough
        # to allocate the Adam optimizer state buffers and one full
        # forward+backward+step+clamp pass, which dominates peak memory.
        probe_steps = int(os.environ.get(
            "STRAPLE_BATCH_PROBE_STEPS", "4"))
        K_align = int(os.environ.get("STRAPLE_BATCH_K_ALIGN", "32"))

        def _round_K(K_raw, fits_pred):
            K_floor = (K_raw // K_align) * K_align
            K_ceil = ((K_raw + K_align - 1) // K_align) * K_align
            if K_ceil > 0 and fits_pred(K_ceil):
                return K_ceil, "↑"
            return K_floor, "↓"

        try:
            import torch as _t
            cuda_avail = _t.cuda.is_available()
            if cuda_avail:
                free_b, total_b = _t.cuda.mem_get_info()
                free_gb = free_b / 1e9
                total_gb = total_b / 1e9
            else:
                free_gb = total_gb = 0.0
        except Exception:
            cuda_avail = False
            free_gb = total_gb = 0.0

        k_env = os.environ.get("STRAPLE_BATCH_K", "")
        if k_env:
            K_initial = int(k_env)
            self._log(f"[{bench_label}] K={K_initial} (env override; "
                      f"probe will validate but not grow)")
            do_probe = bool(cuda_avail) and (free_gb > 0)
            env_override = True
        else:
            env_override = False
            usable_probe_mb = max(0.0, (free_gb - probe_safety_gb) * 1024.0)
            K_raw = int(usable_probe_mb // max(peak_per_K_mb, 1.0))
            K_init_aligned, _dir = _round_K(
                K_raw,
                lambda K: K * peak_per_K_mb <= usable_probe_mb)
            K_init_aligned = max(K_align, K_init_aligned)
            K_initial = max(K_min, min(K_max, K_init_aligned))
            self._log(
                f"[{bench_label}] auto-K initial: VRAM free={free_gb:.1f} GB "
                f"/ {total_gb:.1f} GB, probe_safety={probe_safety_gb} GB → "
                f"per-K≈{peak_per_K_mb:.1f} MB est → K_initial={K_initial}")
            do_probe = bool(cuda_avail) and (free_gb > 0)
        K = K_initial

        if not cuda_avail:
            self._log(f"[{bench_label}] CUDA unavailable — falling back to "
                      f"single-thread CPU gradient (K reduced to 8)")
            K = min(K, 8)

        # ---- Build name_to_global, proxy packages ----
        name_to_global = {}
        for bidx, idx in enumerate(plc.hard_macro_indices):
            name_to_global[plc.modules_w_pins[idx].get_name()] = bidx
        for sidx, idx in enumerate(plc.soft_macro_indices):
            name_to_global[plc.modules_w_pins[idx].get_name()] = n_hard + sidx
        edges_pkg = build_routing_edges_full(plc, name_to_global, n_total)
        routing_consts = build_routing_consts(
            plc, canvas_w, canvas_h,
            int(benchmark.grid_rows), int(benchmark.grid_cols))
        smooth_matrices = build_smooth_matrices(
            int(benchmark.grid_rows), int(benchmark.grid_cols),
            routing_consts["smooth_range"])
        wl_pkg = build_wl_pkg_full(plc, name_to_global, n_total)
        proxy_pkgs = {
            "edges_pkg": edges_pkg,
            "smooth_matrices": smooth_matrices,
            "routing_consts": routing_consts,
            "wl_pkg": wl_pkg,
        }

        self._log(f"[{bench_label}] gradient_batch preset: K={K} "
                  f"time_budget={time_budget:.0f}s n_total={n_total} "
                  f"(hard={n_hard} soft={n_soft})")

        # ---- Gradient batch: K parallel seeds with GPU proxy fitness ----
        # Default knobs tuned on ibm01 — match best-of submission run.
        os.environ.setdefault("STRAPLE_BATCH_EPLACE", "1")
        os.environ.setdefault("STRAPLE_BATCH_EPLACE_GRID", "128")
        os.environ.setdefault("STRAPLE_BATCH_CONG_W", "10")
        os.environ.setdefault("STRAPLE_BATCH_COHESION_START", "5")
        os.environ.setdefault("STRAPLE_BATCH_COHESION_END", "0.001")
        os.environ.setdefault("STRAPLE_BATCH_DIVERSITY", "1")
        os.environ.setdefault("STRAPLE_BATCH_OVERLAP_FORM", "rect_quad")
        os.environ.setdefault("STRAPLE_BATCH_OVERLAP_SOFT", "1")
        # Winning config of 2026-05-09/10/11 exploration: schedule tuning +
        # L-BFGS finisher + CD/pair-swap/triple-cycle polish. See
        # submissions/straple/best_placements/README.md.
        os.environ.setdefault("STRAPLE_BATCH_OVERFLOW_LAMBDA", "1")
        os.environ.setdefault("STRAPLE_BATCH_OVERFLOW_TARGET", "0.13")
        os.environ.setdefault("STRAPLE_BATCH_OVERFLOW_EXP", "0.7")
        os.environ.setdefault("STRAPLE_BATCH_OVERFLOW_COEF_HI", "1.5")
        os.environ.setdefault("STRAPLE_BATCH_BLOCKAGE_W", "50")
        os.environ.setdefault("STRAPLE_BATCH_OVERLAP_W_MAX", "50000")
        os.environ.setdefault("STRAPLE_BATCH_OVERLAP_W_GROWTH", "1.004")
        os.environ.setdefault("STRAPLE_BATCH_LBFGS_FROM_STEP", "1000")
        os.environ.setdefault("STRAPLE_BATCH_LBFGS_ALPHA", "1.0")
        os.environ.setdefault("STRAPLE_BATCH_LBFGS_CLIP", "0.3")
        os.environ.setdefault("STRAPLE_BATCH_CD_POLISH", "1")
        os.environ.setdefault("STRAPLE_BATCH_CD_GPU_FILTER", "1")
        os.environ.setdefault("STRAPLE_BATCH_CD_GPU_APPROX", "1")
        os.environ.setdefault("STRAPLE_BATCH_CD_DIRS", "8")
        os.environ.setdefault("STRAPLE_BATCH_CD_ROUNDS", "8")
        os.environ.setdefault("STRAPLE_BATCH_PAIR_SWAP", "1")
        os.environ.setdefault("STRAPLE_BATCH_PAIR_SWAP_NEIGHBORS", "12")
        os.environ.setdefault("STRAPLE_BATCH_PAIR_SWAP_ROUNDS", "8")
        os.environ.setdefault("STRAPLE_BATCH_TRIPLE_CYCLE", "1")
        os.environ.setdefault("STRAPLE_BATCH_TRIPLE_CYCLE_NEIGHBORS", "6")
        os.environ.setdefault("STRAPLE_BATCH_TRIPLE_CYCLE_ROUNDS", "4")
        # Plateau-detect + per-seed crossover OFF by default for submission —
        # plain gradient with multi-start diversity has been more reliable.
        # Re-enable with STRAPLE_BATCH_PLATEAU_OPS=1.
        os.environ.setdefault("STRAPLE_BATCH_PLATEAU_OPS", "0")
        os.environ.setdefault("STRAPLE_BATCH_PLATEAU_PATIENCE", "30")
        os.environ.setdefault("STRAPLE_BATCH_PLATEAU_INTERVAL", "20")
        os.environ.setdefault("STRAPLE_BATCH_PLATEAU_EPS", "0.005")
        os.environ.setdefault("STRAPLE_BATCH_GA_ELITE_PCT", "0.25")
        os.environ.setdefault("STRAPLE_BATCH_GA_MUTATION_RATE", "0.01")
        os.environ.setdefault("STRAPLE_BATCH_GA_MUTATION_SIGMA", "0.005")
        # GPU proxy: used ONLY for end-of-run selection by default.
        #   STRAPLE_BATCH_USE_GPU_PROXY=1 — proxy as in-loop fitness (slow,
        #     and sparse refresh causes false plateaus — experimental).
        #   STRAPLE_BATCH_RECORD_PROXY=1  — record proxy_history for plots
        #     (adds overhead).  Submission has both OFF.
        os.environ.setdefault("STRAPLE_BATCH_USE_GPU_PROXY", "0")
        os.environ.setdefault("STRAPLE_BATCH_RECORD_PROXY", "0")
        os.environ.setdefault("STRAPLE_BATCH_PROXY_INTERVAL", "50")
        # Snapshots only needed for the HTML viz / plots; submission skips.
        os.environ.setdefault("STRAPLE_BATCH_SNAPSHOT_EVERY", "999")

        anchor_beta_start = float(os.environ.get(
            "STRAPLE_BATCH_ANCHOR_BETA_START", "0"))
        anchor_beta_end = float(os.environ.get(
            "STRAPLE_BATCH_ANCHOR_BETA_END", "0"))
        use_eplace = os.environ.get("STRAPLE_BATCH_EPLACE", "0") == "1"
        eplace_grid = int(os.environ.get("STRAPLE_BATCH_EPLACE_GRID", "128"))
        cong_weight = float(os.environ.get("STRAPLE_BATCH_CONG_W", "0"))
        per_k_div = os.environ.get("STRAPLE_BATCH_DIVERSITY", "0") == "1"
        cohesion_start = float(os.environ.get(
            "STRAPLE_BATCH_COHESION_START", "0"))
        cohesion_end = float(os.environ.get(
            "STRAPLE_BATCH_COHESION_END", "0"))

        gb_kwargs_common = dict(
            seed=self.seed,
            device="cuda" if cuda_avail else "cpu",
            anchor_strategy=os.environ.get(
                "STRAPLE_BATCH_ANCHOR_STRATEGY", "centroid"),
            spawn_radius_frac=0.05, spawn_adaptive=True,
            anchor_jitter_frac=0.05,
            anchor_loss_beta_start=anchor_beta_start,
            anchor_loss_beta_end=anchor_beta_end,
            cohesion_beta_start=cohesion_start,
            cohesion_beta_end=cohesion_end,
            use_eplace_density=use_eplace,
            eplace_grid_size=eplace_grid,
            cong_weight=cong_weight,
            per_k_diversity=per_k_div,
        )

        # ---- Probe step: measure real per-seed peak, then size K to
        # ---- fill_frac of total VRAM (default 0.92 — close to OOM but with
        # ---- ~8 % jitter margin).  On OOM we shrink K by 20 % per retry
        # ---- (rather than halving) and try again, up to a bounded number
        # ---- of attempts.  This keeps us close to the real capacity even
        # ---- when the initial estimate was just slightly optimistic.
        if do_probe:
            import torch as _t
            probe_shrink = float(os.environ.get(
                "STRAPLE_BATCH_PROBE_SHRINK", "0.8"))
            t_probe_total = _time.time()
            attempt = 0
            lo_K = None         # largest K known to fit
            hi_K = None         # smallest K known to OOM
            best_per_K_mb = None
            best_target_mb = None

            def _try_probe(K_try):
                """Run a probe at K_try.  Returns (peak_mb, real_per_K_mb,
                fill_target_mb) on success, or None on OOM.  Cleans up
                allocations either way.
                """
                nonlocal attempt
                attempt += 1
                _t.cuda.empty_cache()
                _t.cuda.reset_peak_memory_stats()
                t_attempt = _time.time()
                try:
                    pp, ps = gradient_batch(
                        benchmark, plc, K=K_try, num_steps=probe_steps,
                        time_budget=0.0,
                        proxy_pkgs=None,
                        verbose=False,
                        **gb_kwargs_common,
                    )
                    peak_b = _t.cuda.max_memory_allocated()
                    del pp, ps
                    _t.cuda.empty_cache()
                    _t.cuda.reset_peak_memory_stats()
                    peak_mb = peak_b / (1024 ** 2)
                    per_K = peak_mb / max(K_try, 1)
                    _, total_b2 = _t.cuda.mem_get_info()
                    total_gb_real = total_b2 / 1e9
                    # Absolute target: fill_frac of TOTAL VRAM.  We don't
                    # subtract "current free" because the PyTorch allocator
                    # pool stays reserved after empty_cache() and would
                    # falsely shrink the target.
                    fill_target = total_gb_real * fill_frac * 1024.0
                    self._log(
                        f"[{bench_label}] probe attempt #{attempt} "
                        f"({_time.time()-t_attempt:.2f}s): K={K_try} OK → "
                        f"peak={peak_mb:.0f} MB, per-K={per_K:.1f} MB, "
                        f"fill_target={fill_target:.0f} MB")
                    return (peak_mb, per_K, fill_target)
                except _t.cuda.OutOfMemoryError:
                    _t.cuda.empty_cache()
                    self._log(
                        f"[{bench_label}] probe attempt #{attempt} "
                        f"({_time.time()-t_attempt:.2f}s): K={K_try} OOM")
                    return None

            # Phase 1: shrink until fits.
            attempt_K = K_initial
            while True:
                r = _try_probe(attempt_K)
                if r is not None:
                    lo_K = attempt_K
                    best_per_K_mb = r[1]
                    best_target_mb = r[2]
                    break
                hi_K = attempt_K
                next_K = max(
                    K_align,
                    (int(attempt_K * probe_shrink) // K_align) * K_align)
                if next_K >= attempt_K or next_K < K_min:
                    break
                attempt_K = next_K

            # Phase 1.5: adaptive growth — if probe at lo_K fit easily and
            # the formula says we have headroom (real per-K < expected),
            # try a bigger K to discover an upper bound.  This kicks in
            # whenever Phase 1 succeeded on first attempt (hi_K is None).
            if lo_K is not None and hi_K is None:
                while True:
                    K_target_grow = int(
                        best_target_mb // max(best_per_K_mb, 1.0))
                    K_target_grow = (K_target_grow // K_align) * K_align
                    K_target_grow = min(K_max, K_target_grow)
                    if K_target_grow <= lo_K:
                        break
                    r = _try_probe(K_target_grow)
                    if r is not None:
                        lo_K = K_target_grow
                        best_per_K_mb = r[1]
                        best_target_mb = r[2]
                        # If at K_max, we can't grow further.
                        if lo_K >= K_max:
                            break
                    else:
                        hi_K = K_target_grow
                        break

            # Phase 2: binary-search upward between lo_K and hi_K to find
            # the largest aligned K that still fits.  This recovers capacity
            # lost to the coarse shrink step.
            if lo_K is not None and hi_K is not None and hi_K - lo_K > K_align:
                while True:
                    mid_raw = (lo_K + hi_K) // 2
                    mid = (mid_raw // K_align) * K_align
                    if mid <= lo_K or mid >= hi_K:
                        break
                    r = _try_probe(mid)
                    if r is not None:
                        lo_K = mid
                        best_per_K_mb = r[1]
                        best_target_mb = r[2]
                    else:
                        hi_K = mid

            if lo_K is None:
                # Probe never succeeded — fall back to estimate-shrink result.
                K = max(K_min, attempt_K)
                self._log(
                    f"[{bench_label}] probe failed after {attempt} "
                    f"attempts; fallback K={K} "
                    f"(probe wall {_time.time()-t_probe_total:.1f}s)")
            else:
                # Pick K_final from real per-K under the fill_frac target,
                # capped by the known-OK lo_K (since anything above it has
                # not been verified during this run except by binary search,
                # which already advanced lo_K).
                K_raw_real = int(
                    best_target_mb // max(best_per_K_mb, 1.0))
                K_aligned_real, dir2 = _round_K(
                    K_raw_real,
                    lambda Kx: Kx * best_per_K_mb <= best_target_mb)
                K_aligned_real = max(K_align, K_aligned_real)
                K_final = max(K_min, min(K_max, K_aligned_real))
                # Never exceed the largest probed-OK K.
                K_final = min(K_final, lo_K)
                if env_override:
                    K_final = min(K_final, K_initial)
                self._log(
                    f"[{bench_label}] probe DONE in "
                    f"{_time.time()-t_probe_total:.1f}s, {attempt} attempts: "
                    f"lo_K={lo_K} hi_K={hi_K} per_K={best_per_K_mb:.1f} MB "
                    f"target={best_target_mb:.0f} MB → K={K_final} "
                    f"(est peak={K_final * best_per_K_mb / 1024.0:.2f} GB)")
                K = K_final

        t_grad = _time.time()
        pos_K, stats = gradient_batch(
            benchmark, plc, K=K, num_steps=20000,
            time_budget=time_budget,
            proxy_pkgs=proxy_pkgs,
            verbose=bool(self.verbose),
            **gb_kwargs_common,
        )
        self._log(f"[{bench_label}] gradient: {_time.time()-t_grad:.1f}s "
                  f"plateau={stats.get('plateau_events',0)} "
                  f"recombined={stats.get('seeds_recombined_total',0)}")

        # ---- Pick best by GPU proxy fitness ----
        proxy_K_last = stats.get("proxy_history", None)
        if proxy_K_last is not None and len(proxy_K_last) > 0:
            cand_proxy = proxy_K_last[-1]
        elif "fitness_history" in stats and stats["fitness_history"] is not None:
            cand_proxy = stats["fitness_history"][-1]
        else:
            ova = stats.get("overlap_area_K")
            cand_proxy = (ova if ova is not None
                          else np.zeros(K, dtype=np.float32))
        topn = legalize_topn if legalize_topn > 0 else K
        cand_idx = np.argsort(cand_proxy)[:topn].tolist()

        # ---- Parallel legalize candidates and pick the best valid ----
        sys.path.insert(0, str(_CPP_DIR))
        bench_dir_str = str(Path(
            "external/MacroPlacement/Testcases/ICCAD04") / benchmark.name)
        if not Path(bench_dir_str).exists():
            # NG45 fallback: use ng45_dir mapping (same as _load_plc).
            ng45 = {
                "ariane133_ng45": "ariane133",
                "ariane136_ng45": "ariane136",
                "nvdla_ng45": "nvdla",
                "mempool_tile_ng45": "mempool_tile",
            }
            d = ng45.get(benchmark.name)
            if d:
                bench_dir_str = str(
                    Path("external/MacroPlacement/Flows/NanGate45") / d
                    / "netlist" / "output_CT_Grouping")

        scripts_dir = str(
            Path(__file__).resolve().parent.parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from gpu_run_one import _legalize_only as _leg_only

        sizes_np = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        movable_np = (benchmark.get_movable_mask()[:n_hard]
                       .cpu().numpy().astype(bool))
        pos_hard_list = [
            pos_K[k, :n_hard].astype(np.float64).copy() for k in cand_idx]
        n_workers = min(_mp.cpu_count(), 16)
        t_leg = _time.time()
        with _mp.get_context("fork").Pool(n_workers) as pool:
            results_leg = pool.starmap(_leg_only, [
                (cand_idx[j], pos_hard_list[j], sizes_np, movable_np,
                 float(canvas_w), float(canvas_h),
                 int(self.seed) + cand_idx[j])
                for j in range(len(cand_idx))])
        self._log(f"[{bench_label}] legalize-only {len(cand_idx)} candidates: "
                  f"{_time.time()-t_leg:.1f}s ({n_workers} workers)")

        # ---- Build legalized full pos for ALL candidates and proxy on GPU ----
        import torch as _t
        dev = _t.device("cuda" if _t.cuda.is_available() else "cpu")
        Nc = len(results_leg)
        pos_full_K = (benchmark.macro_positions.clone()
                      .unsqueeze(0).expand(Nc, -1, -1).contiguous()
                      .to(_t.float32))
        cand_idx_after = []
        for slot, (k, leg_hard) in enumerate(results_leg):
            pos_full_K[slot, :n_hard] = _t.tensor(leg_hard, dtype=_t.float32)
            if pos_K.shape[1] > n_hard:
                pos_full_K[slot, n_hard:n_hard + n_soft] = _t.tensor(
                    pos_K[k, n_hard:n_hard + n_soft], dtype=_t.float32)
            cand_idx_after.append(k)
        pos_full_K = pos_full_K.to(dev)
        sizes_t = benchmark.macro_sizes[:n_total].to(dev, _t.float32)

        # GPU pairwise overlap_count to flag invalid candidates
        half_w = sizes_t[:n_hard, 0] / 2.0
        half_h = sizes_t[:n_hard, 1] / 2.0
        sx = (sizes_t[:n_hard, 0:1] + sizes_t[:n_hard, 0].unsqueeze(0)) * 0.5
        sy = (sizes_t[:n_hard, 1:2] + sizes_t[:n_hard, 1].unsqueeze(0)) * 0.5
        eye = (1.0 - _t.eye(n_hard, dtype=_t.float32, device=dev))
        ph = pos_full_K[:, :n_hard]
        dx = ph[:, :, 0:1] - ph[:, :, 0].unsqueeze(1)
        dy = ph[:, :, 1:2] - ph[:, :, 1].unsqueeze(1)
        ox = _t.relu(sx[None, :, :] - _t.abs(dx))
        oy = _t.relu(sy[None, :, :] - _t.abs(dy))
        ov_area_pair = ox * oy * eye[None, :, :]
        # Google's overlap threshold is 0.4% of cell area; legalize should
        # bring everything to 0, but we use 1e-6 here as a safety floor.
        invalid_mask = (ov_area_pair > 1e-6).any(dim=(1, 2))
        n_invalid = int(invalid_mask.sum().item())

        # GPU batched proxy_cost (Google PlacementCost reproduction).
        from gpu_proxy import gpu_proxy_batched
        macro_idx_p = padded_probe[0].to(dev)
        offsets_p = padded_probe[1].to(dev)
        mask_p = padded_probe[2].to(dev)
        t_proxy = _time.time()
        proxy_K_eval, comp_eval = gpu_proxy_batched(
            pos_full_K, sizes_t, macro_idx_p, offsets_p, mask_p,
            canvas_w, canvas_h, grid_rows, grid_cols,
            macro_idx_p.shape[0],
            n_hard=n_hard,
            edges_pkg=edges_pkg, smooth_matrices=smooth_matrices,
            routing_consts=routing_consts, wl_pkg=wl_pkg,
        )
        proxy_np = proxy_K_eval.cpu().numpy()
        proxy_np_for_pick = proxy_np.copy()
        proxy_np_for_pick[invalid_mask.cpu().numpy()] = float("inf")
        self._log(f"[{bench_label}] GPU proxy on {Nc} legalized: "
                  f"{_time.time()-t_proxy:.2f}s, invalid={n_invalid}")

        if not _np_isfinite_any(proxy_np_for_pick):
            self._log(f"[{bench_label}] WARN: every legalized candidate "
                      f"still has overlap; returning lowest-overlap raw")
            ova = stats.get("overlap_area_K")
            order = np.argsort(ova) if ova is not None else np.arange(K)
            k = int(order[0])
            full_b = benchmark.macro_positions.clone()
            full_b[:n_hard] = torch.tensor(
                pos_K[k, :n_hard], dtype=torch.float32)
            if pos_K.shape[1] > n_hard:
                full_b[n_hard:n_hard + n_soft] = torch.tensor(
                    pos_K[k, n_hard:n_hard + n_soft], dtype=torch.float32)
            return full_b

        best_slot = int(np.argmin(proxy_np_for_pick))
        best_proxy = float(proxy_np_for_pick[best_slot])
        best_k = int(cand_idx_after[best_slot])
        self._log(f"[{bench_label}] BEST proxy={best_proxy:.4f} "
                  f"(legalized seed k={best_k})")

        valid_np = ~invalid_mask.cpu().numpy()
        n_valid = int(valid_np.sum())
        if n_valid > 0:
            def _qstats(name: str, arr: np.ndarray) -> str:
                v = arr[valid_np]
                qs = np.quantile(v, [0.0, 0.25, 0.5, 0.75, 1.0])
                return (f"{name:>5}: min={qs[0]:.4f} p25={qs[1]:.4f} "
                        f"p50={qs[2]:.4f} p75={qs[3]:.4f} max={qs[4]:.4f} "
                        f"mean={float(v.mean()):.4f} std={float(v.std()):.4f}")
            wl_np = comp_eval["wl"].cpu().numpy() if "wl" in comp_eval else None
            den_np = comp_eval["density"].cpu().numpy() if "density" in comp_eval else None
            cong_np = comp_eval["congestion"].cpu().numpy() if "congestion" in comp_eval else None
            self._log(f"[{bench_label}] DIST over {n_valid}/{Nc} valid legalized seeds:")
            self._log(f"[{bench_label}]   {_qstats('proxy', proxy_np)}")
            if wl_np is not None:
                self._log(f"[{bench_label}]   {_qstats('wl', wl_np)}")
            if den_np is not None:
                self._log(f"[{bench_label}]   {_qstats('dens', den_np)}")
            if cong_np is not None:
                self._log(f"[{bench_label}]   {_qstats('cong', cong_np)}")

        # ---- Optional save of final placement for offline debugging ----
        save_path = os.environ.get("STRAPLE_BATCH_SAVE_FINAL_PATH", "")
        if save_path:
            try:
                import numpy as _np
                final_np = pos_full_K[best_slot].cpu().numpy()
                _np.savez(save_path, pos=final_np,
                           bench=str(benchmark.name),
                           best_proxy_gpu=float(proxy_K_eval[best_slot].item()))
                self._log(f"[{bench_label}] saved final → {save_path}")
            except Exception as e:
                self._log(f"[{bench_label}] save failed: {e}")

        # ---- CPU verification of GPU proxy (env-gated debug) ----
        if os.environ.get("STRAPLE_BATCH_VERIFY_CPU", "0") == "1":
            try:
                from macro_place.objective import compute_proxy_cost
                full_check = pos_full_K[best_slot].cpu()
                cpu_cost = compute_proxy_cost(full_check, benchmark, plc)
                gpu_p = float(proxy_K_eval[best_slot].item())
                # Pull GPU per-component for the same slot.
                wl_g = float(comp_eval["wl"][best_slot].item()) \
                    if "wl" in comp_eval else float("nan")
                den_g = float(comp_eval["density"][best_slot].item()) \
                    if "density" in comp_eval else float("nan")
                cong_g = float(comp_eval["congestion"][best_slot].item()) \
                    if "congestion" in comp_eval else float("nan")
                self._log(
                    f"[{bench_label}] VERIFY:\n"
                    f"  wl   GPU={wl_g:.6f}  CPU={cpu_cost['wirelength_cost']:.6f}  "
                    f"diff={wl_g - cpu_cost['wirelength_cost']:+.6f}\n"
                    f"  den  GPU={den_g:.6f}  CPU={cpu_cost['density_cost']:.6f}  "
                    f"diff={den_g - cpu_cost['density_cost']:+.6f}\n"
                    f"  cong GPU={cong_g:.6f}  CPU={cpu_cost['congestion_cost']:.6f}  "
                    f"diff={cong_g - cpu_cost['congestion_cost']:+.6f}\n"
                    f"  PROXY GPU={gpu_p:.6f}  CPU={cpu_cost['proxy_cost']:.6f}  "
                    f"diff={gpu_p - cpu_cost['proxy_cost']:+.6f} "
                    f"({100*(gpu_p - cpu_cost['proxy_cost'])/cpu_cost['proxy_cost']:+.2f}%)")
            except Exception as e:
                self._log(f"[{bench_label}] verify failed: {e}")

        # ---- Optional evolution plot (env-gated) ----
        if os.environ.get("STRAPLE_BATCH_PLOT", "0") == "1":
            try:
                self._save_evolution_plot(
                    benchmark, bench_label, stats,
                    final_proxy_K=proxy_np, final_overlap_K=invalid_mask.cpu().numpy(),
                    best_slot=best_slot, best_k=best_k, best_proxy=best_proxy,
                )
            except Exception as e:
                self._log(f"[{bench_label}] plot skipped: {e}")

        # ---- Polish stack: CD + pair-swap + triple-cycle ----
        # Winning addition from 2026-05-09/10/11 exploration. Operates on
        # the legalized best seed. Each stage is env-gated; the polish is
        # incremental and respects STRAPLE_BATCH_WALL_TL.
        best_pos_full = pos_full_K[best_slot].cpu().numpy()
        proxy_pkgs_cd = {
            "edges_pkg": edges_pkg,
            "smooth_matrices": smooth_matrices,
            "routing_consts": routing_consts,
            "wl_pkg": wl_pkg,
        }
        try:
            best_pos_full = self._run_polish_stack(
                benchmark, plc, best_pos_full, bench_label,
                t_place_start=t_place_start, proxy_pkgs_cd=proxy_pkgs_cd)
        except Exception as e:
            import traceback
            self._log(f"[{bench_label}] polish stack failed: {e}\n"
                      f"{traceback.format_exc()}")
            self._log(f"[{bench_label}] returning unpolished placement")
        return torch.tensor(best_pos_full, dtype=torch.float32)

    def _run_polish_stack(self, benchmark, plc, best_pos_full, bench_label,
                           t_place_start: float, proxy_pkgs_cd):
        """Apply CD + pair-swap + triple-cycle polish to best legalized seed."""
        import time as _time
        import numpy as _np

        _STRAPLE_DIR = str(Path(__file__).resolve().parent)
        if _STRAPLE_DIR not in sys.path:
            sys.path.insert(0, _STRAPLE_DIR)

        wall_tl = float(os.environ.get("STRAPLE_BATCH_WALL_TL", "3300"))
        wall_reserve = float(os.environ.get("STRAPLE_BATCH_WALL_RESERVE", "60"))

        def _remaining():
            return wall_tl - (_time.time() - t_place_start) - wall_reserve

        cd_polish_enable = os.environ.get("STRAPLE_BATCH_CD_POLISH", "0") == "1"
        cd_gpu_enable = os.environ.get("STRAPLE_BATCH_CD_GPU_FILTER", "0") == "1"
        pswap_enable = os.environ.get("STRAPLE_BATCH_PAIR_SWAP", "0") == "1" and cd_gpu_enable
        tcyc_enable = os.environ.get("STRAPLE_BATCH_TRIPLE_CYCLE", "0") == "1" and cd_gpu_enable

        if not (cd_polish_enable or pswap_enable or tcyc_enable):
            return best_pos_full
        if _remaining() <= 0:
            self._log(f"[{bench_label}] polish: wall_remaining<=0 — skipping all")
            return best_pos_full

        from cd_polish import (cd_polish_gpu, pair_swap_polish_gpu,
                                triple_cycle_polish_gpu)
        cd_proxy_chunk_n = int(os.environ.get(
            "STRAPLE_BATCH_CD_PROXY_CHUNK_N", "32"))

        if cd_polish_enable and _remaining() > 0:
            t0 = _time.time()
            cd_rounds = int(os.environ.get("STRAPLE_BATCH_CD_ROUNDS", "8"))
            cd_dirs = int(os.environ.get("STRAPLE_BATCH_CD_DIRS", "8"))
            cd_topk = int(os.environ.get("STRAPLE_BATCH_CD_TOPK_VERIFY", "3"))
            cd_macro_chunk = int(os.environ.get("STRAPLE_BATCH_CD_MACRO_CHUNK", "64"))
            sf_str = os.environ.get(
                "STRAPLE_BATCH_CD_SF",
                "0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625")
            cd_sf = tuple(float(x) for x in sf_str.split(",") if x.strip())
            cd_budget = max(1.0, _remaining())
            self._log(f"[{bench_label}] CD polish: rounds={cd_rounds} "
                      f"dirs={cd_dirs} budget={cd_budget:.0f}s")
            new_pos, new_proxy = cd_polish_gpu(
                benchmark, plc, best_pos_full,
                proxy_pkgs=proxy_pkgs_cd,
                rounds=cd_rounds, step_factors=cd_sf,
                n_directions=cd_dirs, topk_verify=cd_topk,
                macro_chunk=cd_macro_chunk,
                time_budget=cd_budget,
                proxy_chunk_n=cd_proxy_chunk_n,
                approx_verify=True, approx_threshold=1e-5,
                approx_refresh_per_accept=False,
                verbose=False)
            self._log(f"[{bench_label}] CD polish: proxy={new_proxy:.4f} "
                      f"({_time.time()-t0:.1f}s)")
            best_pos_full = new_pos.astype(_np.float32)

        if pswap_enable and _remaining() > 0:
            t0 = _time.time()
            ps_n = int(os.environ.get("STRAPLE_BATCH_PAIR_SWAP_NEIGHBORS", "12"))
            ps_r = int(os.environ.get("STRAPLE_BATCH_PAIR_SWAP_ROUNDS", "8"))
            ps_chunk = int(os.environ.get("STRAPLE_BATCH_PAIR_SWAP_CHUNK", "256"))
            ps_budget = max(1.0, _remaining())
            self._log(f"[{bench_label}] PAIR_SWAP: n={ps_n} rounds={ps_r} "
                      f"budget={ps_budget:.0f}s")
            new_pos, new_proxy = pair_swap_polish_gpu(
                benchmark, plc, best_pos_full,
                proxy_pkgs=proxy_pkgs_cd,
                n_neighbors=ps_n, n_rounds=ps_r,
                verbose=False, time_budget=ps_budget,
                proxy_chunk_n=cd_proxy_chunk_n, chunk_pairs=ps_chunk)
            self._log(f"[{bench_label}] PAIR_SWAP: proxy={new_proxy:.4f} "
                      f"({_time.time()-t0:.1f}s)")
            best_pos_full = new_pos.astype(_np.float32)

        if tcyc_enable and _remaining() > 0:
            t0 = _time.time()
            tc_n = int(os.environ.get("STRAPLE_BATCH_TRIPLE_CYCLE_NEIGHBORS", "6"))
            tc_r = int(os.environ.get("STRAPLE_BATCH_TRIPLE_CYCLE_ROUNDS", "4"))
            tc_chunk = int(os.environ.get("STRAPLE_BATCH_TRIPLE_CYCLE_CHUNK", "256"))
            tc_budget = max(1.0, _remaining())
            self._log(f"[{bench_label}] TRIPLE_CYCLE: n={tc_n} rounds={tc_r} "
                      f"budget={tc_budget:.0f}s")
            new_pos, new_proxy = triple_cycle_polish_gpu(
                benchmark, plc, best_pos_full,
                proxy_pkgs=proxy_pkgs_cd,
                n_neighbors=tc_n, n_rounds=tc_r,
                verbose=False, time_budget=tc_budget,
                proxy_chunk_n=cd_proxy_chunk_n, chunk_triples=tc_chunk)
            self._log(f"[{bench_label}] TRIPLE_CYCLE: proxy={new_proxy:.4f} "
                      f"({_time.time()-t0:.1f}s)")
            best_pos_full = new_pos.astype(_np.float32)

        return best_pos_full

    def _save_evolution_plot(self, benchmark, bench_label, stats,
                              final_proxy_K, final_overlap_K,
                              best_slot, best_k, best_proxy):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        repo_root = Path(__file__).resolve().parent.parent.parent
        out_dir = repo_root / "vis"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{benchmark.name}_submission_evolution.png"

        fit_hist = stats.get("fitness_history", None)
        ov_hist = stats.get("overlap_area_history", None)
        proxy_hist = stats.get("proxy_history", None)
        proxy_steps = stats.get("proxy_history_steps", []) or []

        n_panels = 1 + (1 if ov_hist is not None and len(ov_hist) > 0 else 0)
        n_panels += 1 if proxy_hist is not None and len(proxy_hist) > 0 else 0
        # Always add a final-distribution panel (after legalize) if we have it.
        n_panels += 1
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels),
                                  sharex=False)
        if n_panels == 1:
            axes = [axes]
        ai = 0

        if fit_hist is not None and len(fit_hist) > 0:
            steps = np.arange(1, fit_hist.shape[0] + 1)
            f_min = fit_hist.min(axis=1)
            f_p25 = np.percentile(fit_hist, 25, axis=1)
            f_med = np.median(fit_hist, axis=1)
            f_p75 = np.percentile(fit_hist, 75, axis=1)
            f_max = fit_hist.max(axis=1)
            ax = axes[ai]; ai += 1
            ax.fill_between(steps, f_min, f_max, alpha=0.15,
                            color="C0", label="min..max")
            ax.fill_between(steps, f_p25, f_p75, alpha=0.30,
                            color="C0", label="p25..p75")
            ax.plot(steps, f_med, color="C0", lw=1.5, label="median")
            ax.plot(steps, f_min, color="C2", lw=1.0, label="min")
            ax.set_yscale("log")
            ax.set_ylabel("fitness (gradient loss)")
            ax.set_title(f"{benchmark.name}: per-step fitness across "
                         f"K={fit_hist.shape[1]} seeds")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("step")

        if ov_hist is not None and len(ov_hist) > 0:
            steps = np.arange(1, ov_hist.shape[0] + 1)
            o_min = ov_hist.min(axis=1)
            o_med = np.median(ov_hist, axis=1)
            o_max = ov_hist.max(axis=1)
            ax = axes[ai]; ai += 1
            ax.fill_between(steps, o_min, o_max, alpha=0.20,
                            color="C3", label="min..max")
            ax.plot(steps, o_med, color="C3", lw=1.2, label="median")
            ax.plot(steps, o_min, color="C2", lw=1.0, label="min")
            ax.set_yscale("symlog")
            ax.set_ylabel("overlap area (raw)")
            ax.set_xlabel("step")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        if proxy_hist is not None and len(proxy_hist) > 0 and proxy_steps:
            ps = np.array(proxy_steps)
            P = proxy_hist
            p_min = np.nanmin(P, axis=1)
            p_p25 = np.nanpercentile(P, 25, axis=1)
            p_med = np.nanmedian(P, axis=1)
            p_p75 = np.nanpercentile(P, 75, axis=1)
            p_max = np.nanmax(P, axis=1)
            ax = axes[ai]; ai += 1
            ax.fill_between(ps, p_min, p_max, alpha=0.15,
                            color="C4", label="min..max")
            ax.fill_between(ps, p_p25, p_p75, alpha=0.30,
                            color="C4", label="p25..p75")
            ax.plot(ps, p_med, color="C4", lw=1.5, label="median")
            ax.plot(ps, p_min, color="#117733", lw=1.5,
                    marker="o", ms=3, label="min (best raw)")
            ax.set_ylabel("proxy_cost (GPU repro, raw)")
            ax.set_xlabel("step")
            ax.set_title("In-loop proxy across all K seeds (no legalize)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Final distribution after legalize
        ax = axes[ai]; ai += 1
        valid_mask = ~final_overlap_K.astype(bool)
        proxy_valid = final_proxy_K[valid_mask]
        proxy_invalid = final_proxy_K[~valid_mask]
        if len(proxy_valid):
            ax.hist(proxy_valid, bins=40, alpha=0.7, color="C2",
                    label=f"valid ({len(proxy_valid)})")
        if len(proxy_invalid):
            ax.hist(proxy_invalid, bins=20, alpha=0.5, color="C3",
                    label=f"invalid ({len(proxy_invalid)})")
        ax.axvline(best_proxy, color="black", lw=1.5,
                   label=f"BEST k={best_k} → {best_proxy:.4f}")
        ax.set_xlabel("proxy_cost (after legalize, GPU)")
        ax.set_ylabel("seeds")
        ax.set_title(f"Post-legalize proxy distribution "
                     f"(K={len(final_proxy_K)})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        self._log(f"[{bench_label}] evolution plot → {out_path}")

    def _perturb_initial(self, args, start_idx):
        rng = np.random.default_rng(self.seed + start_idx * 1000 + 7)
        canvas_min = min(args["canvas_w"], args["canvas_h"])
        perturb_idx = start_idx - args["num_orig_starts"]
        env_scales = os.environ.get("STRAPLE_PERTURB_SCALES", "")
        if env_scales:
            scales = [float(s) for s in env_scales.split(",") if s.strip()]
            sigma_factor = scales[perturb_idx % len(scales)] if scales else 0.05
        else:
            default_scales = [0.05, 0.10, 0.15, 0.20]
            sigma_factor = default_scales[perturb_idx % len(default_scales)]
        sigma = sigma_factor * canvas_min
        movable = args["movable_mask"]
        perturb = rng.normal(0.0, sigma, size=args["initial_pos"].shape)
        perturb[~movable] = 0.0
        seed_pos = args["initial_pos"] + perturb
        sizes = args["sizes"]
        half_w = sizes[:, 0] / 2.0
        half_h = sizes[:, 1] / 2.0
        seed_pos[:, 0] = np.clip(seed_pos[:, 0], half_w, args["canvas_w"] - half_w)
        seed_pos[:, 1] = np.clip(seed_pos[:, 1], half_h, args["canvas_h"] - half_h)
        seed_pos[~movable] = args["initial_pos"][~movable]
        if self.verbose:
            self._log(f"[{args['bench_label']}] start#{start_idx} perturbed: "
                      f"sigma={sigma:.2f} (factor={sigma_factor:.3f})")
        return seed_pos

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        import time
        t_place_start = time.time()
        bench_label = getattr(benchmark, "name", "?")
        self._log(f"[{bench_label}] === StraplePlacer.place() ===")

        # Default to the GPU batched gradient path — that's our submission
        # entry point.  Override with STRAPLE_PRESET=high_effort (legacy
        # ALNS) or =""/something else to disable.
        preset = os.environ.get("STRAPLE_PRESET", "gradient_batch")
        if preset == "high_effort":
            # Adaptive num_starts по размеру bench, чтобы вписаться в 1ч лимит
            # per bench. На сервере (16 vCPU) baseline ALNS ibm17 = ~26-39 min,
            # значит 16 starts там нереально (~10ч). Scale down для big benches.
            n_total_for_preset = benchmark.num_macros
            if n_total_for_preset < 1500:        # ibm01-04
                n_orig, n_pert = 4, 12           # 16 starts
                lns_factor, lns_cap = 120, 80000
            elif n_total_for_preset < 2200:      # ibm14
                n_orig, n_pert = 2, 4            # 6 starts
                lns_factor, lns_cap = 80, 60000
            elif n_total_for_preset < 2700:      # ibm10/15/17
                n_orig, n_pert = 2, 3            # 5 starts
                lns_factor, lns_cap = 70, 55000
            else:                                # ibm18+
                n_orig, n_pert = 2, 2            # 4 starts
                lns_factor, lns_cap = 60, 50000
            os.environ.setdefault("STRAPLE_NUM_STARTS", str(n_orig))
            os.environ.setdefault("STRAPLE_PERTURB_EXTRA_STARTS", str(n_pert))
            os.environ.setdefault(
                "STRAPLE_PERTURB_SCALES",
                "0.10,0.20,0.30,0.50,0.10,0.20,0.30,0.50,0.10,0.20,0.30,0.50",
            )
            os.environ.setdefault("STRAPLE_LNS_OUTER_FACTOR", str(lns_factor))
            os.environ.setdefault("STRAPLE_LNS_OUTER_CAP", str(lns_cap))
            os.environ.setdefault("STRAPLE_DESTROY_CAP", "32")
            self._log(f"[{bench_label}] PRESET=high_effort: "
                      f"starts={n_orig}+{n_pert} factor={lns_factor} cap={lns_cap}")

        n_hard = benchmark.num_hard_macros
        plc = _load_plc(benchmark.name)

        recorder = None
        vis_path = os.environ.get("STRAPLE_VIS_VIDEO", "")
        if vis_path and plc is not None:
            _STRAPLE_DIR = str(Path(__file__).resolve().parent)
            if _STRAPLE_DIR not in sys.path:
                sys.path.insert(0, _STRAPLE_DIR)
            from visualizer import PlacementRecorder
            interval = int(os.environ.get("STRAPLE_VIS_INTERVAL", "100"))
            max_frames = int(os.environ.get("STRAPLE_VIS_MAX_FRAMES", "60"))
            recorder = PlacementRecorder(benchmark, plc, vis_path,
                                         interval=interval, max_frames=max_frames)
            self._log(f"[{bench_label}] visualizer ENABLED → {vis_path} "
                      f"(interval={interval}, max_frames={max_frames})")
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

        # ===== GPU batch placement preset: K parallel seeds + GA + GPU proxy =====
        # Activated by STRAPLE_PRESET=gradient_batch (or STRAPLE_BATCH_PRESET=1).
        # Uses gradient_batch.py as the primary placer, scoring inside the
        # loop with the GPU reproduction of Google's PlacementCost.
        gb_preset = (preset == "gradient_batch"
                      or os.environ.get("STRAPLE_BATCH_PRESET", "0") == "1")
        if gb_preset and plc is not None:
            full = self._place_gradient_batch(benchmark, plc, bench_label)
            self._log(f"[{bench_label}] === gradient_batch preset done "
                      f"total={time.time()-t_place_start:.2f}s ===")
            return full

        demo_mode = os.environ.get("STRAPLE_DEMO", "")
        if demo_mode in ("force", "gradient") and plc is not None:
            _STRAPLE_DIR = str(Path(__file__).resolve().parent)
            if _STRAPLE_DIR not in sys.path:
                sys.path.insert(0, _STRAPLE_DIR)
            demo_iters = int(os.environ.get("STRAPLE_DEMO_ITERS", "300"))
            demo_seed = int(os.environ.get("STRAPLE_DEMO_SEED", "42"))
            time_budget = float(os.environ.get("STRAPLE_DEMO_TIME_BUDGET", "0"))
            score_png = os.environ.get("STRAPLE_DEMO_SCORE_PNG", "")
            score_sample_every = float(
                os.environ.get("STRAPLE_DEMO_SCORE_SAMPLE_S", "1.0"))
            self._log(f"[{bench_label}] DEMO mode={demo_mode} "
                      f"iters={demo_iters} time_budget={time_budget}s seed={demo_seed} "
                      f"score_png={score_png or 'none'}")
            t0 = time.time()
            if demo_mode == "force":
                from force_demo import force_directed_demo
                result = force_directed_demo(
                    benchmark, plc, edges, edge_weights,
                    recorder=recorder, num_iters=demo_iters, seed=demo_seed,
                    time_budget=time_budget, score_png=score_png,
                    score_sample_every_s=score_sample_every,
                )
            else:
                from gradient_demo import gradient_demo
                result = gradient_demo(
                    benchmark, plc,
                    recorder=recorder, num_steps=demo_iters, seed=demo_seed,
                    time_budget=time_budget, score_png=score_png,
                    score_sample_every_s=score_sample_every,
                )
            if isinstance(result, tuple):
                final_pos = result[0]
            else:
                final_pos = result
            self._log(f"[{bench_label}] {demo_mode} demo: {time.time()-t0:.2f}s")
            full = benchmark.macro_positions.clone()
            n_demo = final_pos.shape[0]
            full[:n_demo] = torch.tensor(final_pos, dtype=torch.float32)
            if recorder is not None:
                try:
                    recorder.render()
                except Exception as exc:
                    print(f"[visualizer] render failed: {exc}", file=sys.stderr)
            self._log(f"[{bench_label}] === DEMO DONE total={time.time()-t_place_start:.2f}s ===")
            return full

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
                gamma_frac=self.analytical_gamma_frac,
                target_util=self.analytical_target_util,
                lambda_schedule=self.analytical_lambda_schedule,
                gamma_schedule=self.analytical_gamma_schedule,
                log_every=(max(1, self.analytical_steps // 5) if self.verbose else 0),
                log_proxy=bool(self.verbose),
                label=bench_label,
            )
            self._log(f"[{bench_label}] analytical_seed({self.analytical_steps} steps): "
                      f"{time.time()-t0:.2f}s")

        gradient_hard_pos = None
        gradient_soft_pos = None

        # External gradient seed file (from scripts/gpu_batch_search.py)
        seed_file = os.environ.get("STRAPLE_GRADIENT_SEED_FILE", "")
        if seed_file and Path(seed_file).exists():
            try:
                import pickle
                with open(seed_file, "rb") as f:
                    seed_data = pickle.load(f)
                hp = np.asarray(seed_data["hard"], dtype=np.float64)
                sp = np.asarray(seed_data["soft"], dtype=np.float64)
                if hp.shape[0] == n_hard:
                    gradient_hard_pos = hp.copy()
                    self._log(f"[{bench_label}] loaded external gradient hard "
                              f"from {seed_file} (proxy={seed_data.get('proxy', '?')})")
                if sp.shape[0] == benchmark.num_soft_macros and \
                        os.environ.get("STRAPLE_GRADIENT_SEED_USE_SOFT", "0") == "1":
                    gradient_soft_pos = sp.copy()
                    self._log(f"[{bench_label}] loaded external gradient soft")
            except Exception as exc:
                print(f"[{bench_label}] failed to load external seed: {exc}",
                      file=sys.stderr)

        use_gradient_seed = (
            os.environ.get("STRAPLE_USE_GRADIENT_SEED", "0") == "1"
            and plc is not None
            and gradient_hard_pos is None
        )
        if use_gradient_seed:
            _STRAPLE_DIR = str(Path(__file__).resolve().parent)
            if _STRAPLE_DIR not in sys.path:
                sys.path.insert(0, _STRAPLE_DIR)
            try:
                from gradient_demo import gradient_demo
                env_snapshot = {}
                for k, v in (("STRAPLE_DEMO_INIT", "anchor_soft"),
                             ("STRAPLE_DEMO_PLACE_ALL", "1"),
                             ("STRAPLE_DEMO_FINISH_LEGALIZE", "1")):
                    env_snapshot[k] = os.environ.get(k)
                    if env_snapshot[k] is None:
                        os.environ[k] = v
                grad_steps = int(os.environ.get("STRAPLE_GRADIENT_SEED_STEPS", "500"))
                grad_seed = self.seed
                grad_n = max(1, int(
                    os.environ.get("STRAPLE_GRADIENT_SEED_NUM", "1")))
                self._log(f"[{bench_label}] gradient seed pre-pass "
                          f"({grad_n}x{grad_steps} steps)...")
                t0 = time.time()
                from macro_place.objective import compute_proxy_cost
                best_grad_pos = None
                best_grad_proxy = float("inf")
                gp = None
                for k in range(grad_n):
                    sd = grad_seed + k * 1009
                    grad_result = gradient_demo(
                        benchmark, plc, recorder=None,
                        num_steps=grad_steps, seed=sd,
                        time_budget=0.0, score_png="",
                        score_sample_every_s=1.0,
                    )
                    gp = grad_result[0] if isinstance(grad_result, tuple) else grad_result
                    gp = np.asarray(gp, dtype=np.float64)
                    full_test = benchmark.macro_positions.clone()
                    full_test[:n_hard] = torch.tensor(gp[:n_hard], dtype=torch.float32)
                    if gp.shape[0] > n_hard:
                        nsl = benchmark.num_soft_macros
                        full_test[n_hard:n_hard + nsl] = torch.tensor(
                            gp[n_hard:n_hard + nsl], dtype=torch.float32)
                    cost = compute_proxy_cost(full_test, benchmark, plc)
                    self._log(f"[{bench_label}] grad_seed#{k} sd={sd} "
                              f"proxy={cost['proxy_cost']:.4f} "
                              f"ovrlp={cost['overlap_count']}")
                    if (cost['overlap_count'] == 0
                            and cost['proxy_cost'] < best_grad_proxy):
                        best_grad_proxy = cost['proxy_cost']
                        best_grad_pos = gp
                grad_pos = best_grad_pos if best_grad_pos is not None else gp
                if grad_pos is None:
                    raise RuntimeError("all gradient seeds failed")
                if grad_pos.shape[0] >= n_hard:
                    gradient_hard_pos = grad_pos[:n_hard].copy()
                use_grad_soft = (
                    os.environ.get("STRAPLE_GRADIENT_SEED_USE_SOFT", "0") == "1"
                )
                if use_grad_soft and grad_pos.shape[0] > n_hard:
                    n_soft_local = benchmark.num_soft_macros
                    gradient_soft_pos = grad_pos[n_hard:n_hard + n_soft_local].copy()
                self._log(f"[{bench_label}] best grad proxy={best_grad_proxy:.4f}")
                for k, prev in env_snapshot.items():
                    if prev is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = prev
                self._log(f"[{bench_label}] gradient seed: {time.time()-t0:.2f}s "
                          f"(hard={gradient_hard_pos is not None}, "
                          f"soft={gradient_soft_pos is not None})")
            except Exception as exc:
                print(f"[{bench_label}] gradient seed FAILED: {exc}", file=sys.stderr)
                gradient_hard_pos = None
                gradient_soft_pos = None

        num_orig_starts = 3 if num_movable >= 300 else 1
        env_n = int(os.environ.get("STRAPLE_NUM_STARTS", "0"))
        if env_n > 0:
            num_orig_starts = env_n
        num_perturbed_starts = int(os.environ.get("STRAPLE_PERTURB_EXTRA_STARTS", "0"))
        if num_movable < 300:
            num_perturbed_starts = 0
        num_gradient_starts = 1 if gradient_hard_pos is not None else 0
        num_starts = (num_orig_starts + num_perturbed_starts
                      + num_gradient_starts
                      + (1 if analytical_pos is not None else 0))

        evaluator = None
        if plc is not None and self.lns_outer_iters > 0:
            t0 = time.time()
            evaluator = _build_proxy_evaluator(
                benchmark, plc, soft_positions_override=gradient_soft_pos)
            self._log(f"[{bench_label}] _build_proxy_evaluator: {time.time()-t0:.2f}s "
                      f"(soft_override={gradient_soft_pos is not None})")

        best_pos = None
        best_cost = float("inf")

        start_args = {
            "initial_pos": initial_pos,
            "sizes": sizes,
            "movable_mask": movable_mask,
            "edges": edges,
            "edge_weights": edge_weights,
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "num_movable": num_movable,
            "analytical_pos": analytical_pos,
            "gradient_hard_pos": gradient_hard_pos,
            "gradient_soft_pos": gradient_soft_pos,
            "num_starts": num_starts,
            "num_orig_starts": num_orig_starts,
            "num_perturbed_starts": num_perturbed_starts,
            "num_gradient_starts": num_gradient_starts,
            "bench_label": bench_label,
        }

        if recorder is not None:
            recorder.add(initial_pos, "initial")

        parallel_workers = int(os.environ.get("STRAPLE_PARALLEL_STARTS", "0"))
        if parallel_workers > 1 and evaluator is not None and num_starts > 1:
            self._log(f"[{bench_label}] === parallel multi-start ({parallel_workers} workers, "
                      f"{num_starts} starts) ===")
            results = self._run_starts_parallel(start_args, benchmark, plc,
                                                min(parallel_workers, num_starts))
            for trial_pos, trial_cost, start_idx in results:
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_pos = trial_pos
                    self._log(f"[{bench_label}] start#{start_idx} new best={trial_cost:.4f}")
                else:
                    self._log(f"[{bench_label}] start#{start_idx} cost={trial_cost:.4f} "
                              f"(not best, best={best_cost:.4f})")
        else:
            for start_idx in range(num_starts):
                trial_pos, trial_cost = self._run_one_start(start_idx, start_args, evaluator, plc,
                                                            recorder=recorder)
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_pos = trial_pos
                    self._log(f"[{bench_label}] start#{start_idx} new best={trial_cost:.4f}")
                else:
                    self._log(f"[{bench_label}] start#{start_idx} cost={trial_cost:.4f} "
                              f"(not best, best={best_cost:.4f})")

        if evaluator is not None and best_pos is not None:
            for refine_iter in range(3):
                t0 = time.time()
                state_r = _placer_core.PlacerState()
                state_r.initialize(
                    best_pos, sizes, movable_mask,
                    edges, edge_weights,
                    canvas_w, canvas_h, int(self.seed + 9999 + refine_iter * 100),
                )
                self._lns_loop(state_r, evaluator, plc, num_movable,
                               bench_label, 99 + refine_iter, recorder=recorder)
                refined_pos = state_r.current_positions()
                refined_cost = evaluator.evaluate(refined_pos)
                self._log(f"[{bench_label}] REFINE pass#{refine_iter}: "
                          f"{time.time()-t0:.2f}s "
                          f"cost {best_cost:.4f} -> {refined_cost:.4f} "
                          f"(delta {refined_cost-best_cost:+.4f})")
                if refined_cost < best_cost:
                    best_cost = refined_cost
                    best_pos = refined_pos
                    if recorder is not None:
                        recorder.add(best_pos, f"refine#{refine_iter} cost={best_cost:.4f}")
                else:
                    break

        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        if gradient_soft_pos is not None:
            full[n_hard:n_hard + gradient_soft_pos.shape[0]] = torch.tensor(
                gradient_soft_pos, dtype=torch.float32)

        refine_soft = (
            os.environ.get("STRAPLE_REFINE_SOFT", "0") == "1"
            and plc is not None
            and benchmark.num_soft_macros > 0
        )
        if refine_soft and best_pos is not None:
            try:
                from copy import deepcopy
                _STRAPLE_DIR = str(Path(__file__).resolve().parent)
                if _STRAPLE_DIR not in sys.path:
                    sys.path.insert(0, _STRAPLE_DIR)
                from gradient_demo import gradient_demo
                refine_steps = int(
                    os.environ.get("STRAPLE_REFINE_SOFT_STEPS", "300"))
                bench_refine = deepcopy(benchmark)
                bench_refine.macro_positions = full.clone()
                bench_refine.macro_fixed = bench_refine.macro_fixed.clone()
                bench_refine.macro_fixed[:n_hard] = True
                env_snap = {}
                for k, v in (("STRAPLE_DEMO_INIT", "current"),
                             ("STRAPLE_DEMO_PLACE_ALL", "1"),
                             ("STRAPLE_DEMO_FINISH_LEGALIZE", "0"),
                             ("STRAPLE_DEMO_ANCHOR_LOSS", "0")):
                    env_snap[k] = os.environ.get(k)
                    os.environ[k] = v
                self._log(f"[{bench_label}] post-ALNS soft refine "
                          f"({refine_steps} steps)...")
                t0 = time.time()
                refined = gradient_demo(
                    bench_refine, plc, recorder=None,
                    num_steps=refine_steps, seed=self.seed + 31337,
                    time_budget=0.0, score_png="",
                )
                refined_pos = refined[0] if isinstance(refined, tuple) else refined
                refined_pos = np.asarray(refined_pos, dtype=np.float64)
                for k, prev in env_snap.items():
                    if prev is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = prev
                full_refined = full.clone()
                if refined_pos.shape[0] >= n_hard:
                    full_refined[n_hard:] = torch.tensor(
                        refined_pos[n_hard:], dtype=torch.float32)
                from macro_place.objective import compute_proxy_cost
                cost_before = compute_proxy_cost(full, benchmark, plc)
                cost_after = compute_proxy_cost(full_refined, benchmark, plc)
                self._log(f"[{bench_label}] soft refine: "
                          f"{time.time()-t0:.2f}s "
                          f"proxy {cost_before['proxy_cost']:.4f} -> "
                          f"{cost_after['proxy_cost']:.4f} "
                          f"(ovrlp {cost_before['overlap_count']} -> "
                          f"{cost_after['overlap_count']})")
                if (cost_after['proxy_cost'] < cost_before['proxy_cost']
                        and cost_after['overlap_count'] == 0):
                    full = full_refined
            except Exception as exc:
                print(f"[{bench_label}] soft refine FAILED: {exc}", file=sys.stderr)

        if recorder is not None and best_pos is not None:
            recorder.add(full[:n_hard].numpy().astype(np.float64),
                         f"FINAL best={best_cost:.4f}")
            try:
                recorder.render()
            except Exception as exc:
                print(f"[visualizer] render failed: {exc}", file=sys.stderr)

        self._log(f"[{bench_label}] === DONE total={time.time()-t_place_start:.2f}s "
                  f"final_cost={best_cost:.4f} ===")
        return full

    def _run_one_start(self, start_idx, args, evaluator, plc, recorder=None):
        import time
        is_analytical_start = (args["analytical_pos"] is not None
                               and start_idx == args["num_starts"] - 1)
        num_orig = args["num_orig_starts"]
        num_perturbed = args.get("num_perturbed_starts", 0)
        num_gradient = args.get("num_gradient_starts", 0)
        is_perturbed_start = (not is_analytical_start
                              and start_idx >= num_orig
                              and start_idx < num_orig + num_perturbed)
        is_gradient_start = (not is_analytical_start
                             and not is_perturbed_start
                             and num_gradient > 0
                             and start_idx >= num_orig + num_perturbed
                             and start_idx < num_orig + num_perturbed + num_gradient)
        seed = self.seed + start_idx
        t_start_iter = time.time()

        state = _placer_core.PlacerState()
        if is_analytical_start:
            seed_pos = args["analytical_pos"]
        elif is_gradient_start:
            seed_pos = args["gradient_hard_pos"]
        elif is_perturbed_start:
            seed_pos = self._perturb_initial(args, start_idx)
        else:
            seed_pos = args["initial_pos"]
        state.initialize(
            seed_pos, args["sizes"], args["movable_mask"],
            args["edges"], args["edge_weights"],
            args["canvas_w"], args["canvas_h"], int(seed),
        )

        if recorder is not None and (is_analytical_start or is_perturbed_start
                                     or is_gradient_start):
            recorder.add(state.current_positions(), f"start#{start_idx} pre-legalize")

        if is_analytical_start or is_gradient_start:
            state.legalize_min_displacement(500)
            state.legalize()
        else:
            state.legalize()
        cost_after_legal = evaluator.evaluate(state.current_positions()) if evaluator else 0.0
        if recorder is not None:
            recorder.add(state.current_positions(),
                         f"start#{start_idx} post-legalize cost={cost_after_legal:.4f}")

        sa_iters_to_run = self.refine_iters if (
            args["num_movable"] < 300
            and not is_analytical_start
            and not is_gradient_start) else 0
        if sa_iters_to_run > 0:
            state.sa_refine(sa_iters_to_run)
        cost_after_sa = evaluator.evaluate(state.current_positions()) if evaluator else 0.0
        if recorder is not None and sa_iters_to_run > 0:
            recorder.add(state.current_positions(),
                         f"start#{start_idx} post-SA cost={cost_after_sa:.4f}")

        self._log(f"[{args['bench_label']}] start#{start_idx} seed={seed}: "
                  f"legalize proxy={cost_after_legal:.4f} | "
                  f"sa_refine({sa_iters_to_run}) proxy={cost_after_sa:.4f}")

        if evaluator is not None:
            t0 = time.time()
            lns_log = self._lns_loop(state, evaluator, plc, args["num_movable"],
                                     args["bench_label"], start_idx, recorder=recorder)
            self._log(f"[{args['bench_label']}] start#{start_idx} LNS: {time.time()-t0:.2f}s "
                      f"cost {cost_after_sa:.4f} -> {lns_log['final_cost']:.4f}")

        trial_pos = state.current_positions()
        trial_cost = evaluator.evaluate(trial_pos) if evaluator else float(start_idx)
        self._log(f"[{args['bench_label']}] start#{start_idx} done in "
                  f"{time.time()-t_start_iter:.1f}s cost={trial_cost:.4f}")
        return trial_pos, trial_cost

    def _run_starts_parallel(self, args, benchmark, plc, num_workers):
        import multiprocessing as mp
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        with ctx.Pool(num_workers) as pool:
            futures = []
            for start_idx in range(args["num_starts"]):
                futures.append(pool.apply_async(
                    _worker_run_start,
                    (start_idx, args, self.seed, self.refine_iters,
                     self.lns_outer_iters, self.lns_destroy_size, benchmark, plc),
                ))
            results = []
            for start_idx, fut in enumerate(futures):
                trial_pos, trial_cost = fut.get()
                results.append((trial_pos, trial_cost, start_idx))
        return results

    def _lns_loop(self, state, evaluator, plc, num_movable, bench_label="?", start_idx=0,
                  recorder=None):
        import time
        best_pos = state.current_positions()
        best_cost = evaluator.evaluate(best_pos)
        initial_cost = best_cost

        grid_rows = int(plc.grid_row)
        grid_cols = int(plc.grid_col)
        congested_percent = 0.05

        destroy_cap = int(os.environ.get("STRAPLE_DESTROY_CAP", "16"))
        destroy_pct = float(os.environ.get("STRAPLE_DESTROY_PCT", "0.025"))
        adaptive_destroy = max(self.lns_destroy_size,
                               min(destroy_cap, math.ceil(destroy_pct * num_movable)))
        outer_cap = int(os.environ.get("STRAPLE_LNS_OUTER_CAP", "50000"))
        outer_factor = float(os.environ.get("STRAPLE_LNS_OUTER_FACTOR", "60.0"))
        adaptive_outer = max(self.lns_outer_iters, min(outer_cap, math.ceil(outer_factor * num_movable)))

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
            if recorder is not None:
                recorder.maybe_add_lns(trial, iteration, op,
                                       new_cost < best_cost, new_cost)
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
