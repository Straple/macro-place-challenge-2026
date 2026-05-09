"""GPU multi-seed gradient на одном bench с time budget + HTML viz лучшего сида.

Pipeline:
1. gradient_batch(K seeds, time_budget) на GPU — все seeds параллельно
2. Eval all K seeds via compute_proxy_cost, find best valid
3. Save best pos to vis/{bench}_gpu_best.html — replay best seed через
   gradient_demo (single, with recorder), HTML render на сервере.

Usage:
    uv run python scripts/gpu_run_one.py --bench ibm01 --K 384 --time-budget 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple"))


# Worker для proxy_cost trajectory: load bench один раз per process,
# затем считаем proxy для batch снимков (snapshot_idx, seed_idx, pos_full).
_PROXY_WORKER_BENCH = None
_PROXY_WORKER_PLC = None


def _proxy_worker_init(bench_dir_str):
    global _PROXY_WORKER_BENCH, _PROXY_WORKER_PLC
    sys.path.insert(0, str(REPO_ROOT))
    from macro_place.loader import load_benchmark_from_dir as _ldb
    _PROXY_WORKER_BENCH, _PROXY_WORKER_PLC = _ldb(bench_dir_str)


def _proxy_worker_compute(args):
    snap_idx, k, pos_full_np = args
    import torch as _torch
    from macro_place.objective import compute_proxy_cost as _cpc
    full = _torch.tensor(pos_full_np, dtype=_torch.float32)
    cost = _cpc(full, _PROXY_WORKER_BENCH, _PROXY_WORKER_PLC)
    return (snap_idx, k, float(cost["proxy_cost"]),
            int(cost["overlap_count"]))


# Worker: ТОЛЬКО legalize (без compute_proxy_cost).  Используется когда
# proxy считается батчем на GPU после pool.  ~150 ms vs ~2 sec для full eval.
def _legalize_only(k, pos_hard, sizes_np, movable_np,
                   canvas_w, canvas_h, seed):
    import numpy as _np
    sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple" / "cpp"))
    import _placer_core
    state = _placer_core.PlacerState()
    state.initialize(
        pos_hard.copy(), sizes_np, movable_np,
        _np.zeros((0, 2), dtype=_np.int32),
        _np.zeros(0, dtype=_np.float64),
        float(canvas_w), float(canvas_h),
        int(seed),
    )
    state.legalize_min_displacement(500)
    state.legalize()
    return (k, state.current_positions())


# Worker для multiprocessing — legalize одного seed и compute proxy.
# Импорты внутри чтобы fork не дублировал большие tensor'ы.
def _legalize_and_eval(k, pos_hard, pos_soft, bench_dir_str):
    import numpy as _np
    import torch as _torch
    sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple" / "cpp"))
    import _placer_core
    from macro_place.loader import load_benchmark_from_dir as _ldb
    from macro_place.objective import compute_proxy_cost as _cpc
    bench, plc = _ldb(bench_dir_str)
    n_hard = bench.num_hard_macros
    n_soft = bench.num_soft_macros
    sizes_np = bench.macro_sizes[:n_hard].cpu().numpy().astype(_np.float64)
    movable_np = bench.get_movable_mask()[:n_hard].cpu().numpy().astype(bool)
    state = _placer_core.PlacerState()
    state.initialize(
        pos_hard.copy(), sizes_np, movable_np,
        _np.zeros((0, 2), dtype=_np.int32),
        _np.zeros(0, dtype=_np.float64),
        float(bench.canvas_width), float(bench.canvas_height),
        int(42 + k),
    )
    state.legalize_min_displacement(500)
    state.legalize()
    leg_hard = state.current_positions()
    full = bench.macro_positions.clone()
    full[:n_hard] = _torch.tensor(leg_hard, dtype=_torch.float32)
    if pos_soft is not None:
        full[n_hard:n_hard + n_soft] = _torch.tensor(pos_soft, dtype=_torch.float32)
    cost = _cpc(full, bench, plc)
    return (k, leg_hard, float(cost["proxy_cost"]), int(cost["overlap_count"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--K", type=int, default=384)
    parser.add_argument("--time-budget", type=float, default=300.0,
                        help="seconds for gradient batch phase")
    parser.add_argument("--vis-frames", type=int, default=80)
    parser.add_argument("--vis-fps", type=int, default=15)
    parser.add_argument("--no-vis", action="store_true",
                        help="skip HTML render (faster)")
    parser.add_argument("--vis-budget", type=float, default=0.0,
                        help="seconds for visualizer replay (default: same iters as best)")
    args = parser.parse_args()

    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost

    bdir = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / args.bench
    print(f"[gpu_run_one] loading {args.bench}...", flush=True)
    benchmark, plc = load_benchmark_from_dir(str(bdir))
    n_hard = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros
    print(f"[gpu_run_one] {args.bench}: hard={n_hard} soft={n_soft} "
          f"nets={benchmark.num_nets} canvas={benchmark.canvas_width:.1f}x"
          f"{benchmark.canvas_height:.1f}", flush=True)

    c0 = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
    print(f"[gpu_run_one] INITIAL proxy={c0['proxy_cost']:.4f} "
          f"ovrlp={c0['overlap_count']}", flush=True)

    # Phase 1: GPU batch K seeds with time budget
    import torch
    from gradient_batch import gradient_batch

    if torch.cuda.is_available():
        print(f"[gpu_run_one] GPU: {torch.cuda.get_device_name(0)} "
              f"(K={args.K} parallel seeds, budget={args.time_budget}s)", flush=True)
        torch.cuda.reset_peak_memory_stats()
    else:
        print("[gpu_run_one] CUDA not available — fallback CPU", flush=True)

    t0 = time.time()
    # Estimate max steps for time budget. Per-step time grows with K and n.
    # Default to 5000 steps then time_budget will cap.
    anchor_beta_start = float(os.environ.get("STRAPLE_BATCH_ANCHOR_BETA_START", "0"))
    anchor_beta_end = float(os.environ.get("STRAPLE_BATCH_ANCHOR_BETA_END", "0"))
    use_eplace = os.environ.get("STRAPLE_BATCH_EPLACE", "0") == "1"
    eplace_grid_size = int(os.environ.get("STRAPLE_BATCH_EPLACE_GRID", "256"))
    cong_weight = float(os.environ.get("STRAPLE_BATCH_CONG_W", "0"))
    per_k_diversity = os.environ.get("STRAPLE_BATCH_DIVERSITY", "0") == "1"
    cohesion_beta_start = float(os.environ.get("STRAPLE_BATCH_COHESION_START", "0"))
    cohesion_beta_end = float(os.environ.get("STRAPLE_BATCH_COHESION_END", "0"))
    grad_lr = float(os.environ.get("STRAPLE_BATCH_LR", "0.3"))
    grad_overlap_w_max = float(os.environ.get(
        "STRAPLE_BATCH_OVERLAP_W_MAX", "500000"))
    grad_overlap_w_growth = float(os.environ.get(
        "STRAPLE_BATCH_OVERLAP_W_GROWTH", "1.008"))
    pos_K, stats = gradient_batch(
        benchmark, plc, K=args.K,
        num_steps=20000,
        time_budget=args.time_budget,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        anchor_strategy=os.environ.get("STRAPLE_BATCH_ANCHOR_STRATEGY", "centroid"),
        spawn_radius_frac=0.05,
        spawn_adaptive=True,
        anchor_jitter_frac=0.05,
        anchor_loss_beta_start=anchor_beta_start,
        anchor_loss_beta_end=anchor_beta_end,
        cohesion_beta_start=cohesion_beta_start,
        cohesion_beta_end=cohesion_beta_end,
        use_eplace_density=use_eplace,
        eplace_grid_size=eplace_grid_size,
        cong_weight=cong_weight,
        per_k_diversity=per_k_diversity,
        lr=grad_lr,
        overlap_w_max=grad_overlap_w_max,
        overlap_w_growth=grad_overlap_w_growth,
    )
    grad_time = time.time() - t0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"[gpu_run_one] GPU peak memory: {peak_mb:.0f} MB", flush=True)
    print(f"[gpu_run_one] gradient batch: {grad_time:.1f}s "
          f"({grad_time / args.K * 1000:.1f}ms per seed) "
          f"[wall_elapsed={time.time()-t0:.1f}s]", flush=True)
    plat_evts = stats.get("plateau_events", 0)
    seeds_recomb = stats.get("seeds_recombined_total", 0)
    if plat_evts:
        print(f"[gpu_run_one] evolution: plateau_events={plat_evts} "
              f"seeds_recombined={seeds_recomb}", flush=True)

    fit_hist = stats.get("fitness_history", None)
    ov_hist = stats.get("overlap_area_history", None)
    snapshots_pos_all = stats.get("snapshots_pos", None)
    snapshots_step_all = stats.get("snapshots_step", [])
    proxy_traj = None  # populated later after legalize-all-K (full_template/bench_dir_str)

    # Phase 2: filter top-N candidates by overlap_area, eval proxy on top-N
    overlap_area_K = stats.get("overlap_area_K", None)
    eval_topn = min(args.K, 32)
    if overlap_area_K is not None:
        sorted_by_oa = np.argsort(overlap_area_K)
        candidates = sorted_by_oa[:eval_topn].tolist()
    else:
        candidates = list(range(args.K))

    print(f"[gpu_run_one] eval top-{eval_topn} candidates by overlap_area...",
          flush=True)
    sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple" / "cpp"))
    import _placer_core
    sizes_np = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
    movable_np = benchmark.get_movable_mask()[:n_hard].cpu().numpy().astype(bool)

    full_template = benchmark.macro_positions.clone()
    best_proxy = float("inf")
    best_idx = -1
    best_pos_full = None  # full [n_total, 2] including legalized hard + gradient soft

    t_eval = time.time()
    for k in candidates:
        # 1) try without legalize first
        full_raw = full_template.clone()
        full_raw[:n_hard] = torch.tensor(pos_K[k, :n_hard], dtype=torch.float32)
        if pos_K.shape[1] > n_hard:
            full_raw[n_hard:n_hard + n_soft] = torch.tensor(
                pos_K[k, n_hard:n_hard + n_soft], dtype=torch.float32)
        c_raw = compute_proxy_cost(full_raw, benchmark, plc)

        # 2) always run C++ legalize on hard (fixes overlaps without disturbing soft)
        state = _placer_core.PlacerState()
        state.initialize(
            pos_K[k, :n_hard].astype(np.float64).copy(), sizes_np, movable_np,
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.float64),
            float(benchmark.canvas_width), float(benchmark.canvas_height),
            int(42 + k),
        )
        state.legalize_min_displacement(500)
        state.legalize()
        leg_hard = state.current_positions()
        full_leg = full_raw.clone()
        full_leg[:n_hard] = torch.tensor(leg_hard, dtype=torch.float32)
        c_leg = compute_proxy_cost(full_leg, benchmark, plc)

        if c_raw["overlap_count"] == 0 and c_raw["proxy_cost"] < best_proxy:
            best_proxy = float(c_raw["proxy_cost"])
            best_idx = int(k)
            best_pos_full = full_raw.clone().cpu().numpy()
            tag = "raw"
        if c_leg["overlap_count"] == 0 and c_leg["proxy_cost"] < best_proxy:
            best_proxy = float(c_leg["proxy_cost"])
            best_idx = int(k)
            best_pos_full = full_leg.clone().cpu().numpy()
            tag = "leg"
    print(f"[gpu_run_one] eval+legalize: {time.time()-t_eval:.1f}s "
          f"[wall_elapsed={time.time()-t0:.1f}s]", flush=True)
    print(f"\n[gpu_run_one] BEST proxy={best_proxy:.4f} (k={best_idx})", flush=True)

    if best_pos_full is None:
        print(f"[gpu_run_one] no valid solution, exit", flush=True)
        return

    # Save best to results
    import pickle
    seed_path = REPO_ROOT / "results" / f"gpu_seed_{args.bench}.pkl"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(seed_path, "wb") as f:
        pickle.dump({
            "hard": best_pos_full[:n_hard].astype(np.float64),
            "soft": best_pos_full[n_hard:n_hard + n_soft].astype(np.float64),
            "proxy": float(best_proxy),
            "best_idx": int(best_idx),
            "K": int(args.K),
        }, f)
    print(f"[gpu_run_one] saved {seed_path}", flush=True)

    # Phase 2.5: legalize ALL K seeds in parallel (mp.Pool), compute proxy
    # for each. This makes 100% of seeds valid for grid render.
    print(f"[gpu_run_one] legalize ALL {args.K} seeds in parallel...", flush=True)
    t_all = time.time()
    import multiprocessing as mp
    bench_dir_str = str(bdir)
    pos_hard_list = [pos_K[k, :n_hard].astype(np.float64).copy()
                     for k in range(args.K)]
    pos_soft_list = [pos_K[k, n_hard:n_hard + n_soft].astype(np.float64).copy()
                     if pos_K.shape[1] > n_hard else None
                     for k in range(args.K)]
    n_workers = min(mp.cpu_count(), 16)
    with mp.get_context("fork").Pool(n_workers) as pool:
        results_legalize = pool.starmap(
            _legalize_and_eval,
            [(k, pos_hard_list[k], pos_soft_list[k], bench_dir_str)
             for k in range(args.K)],
        )
    all_proxies = np.full(args.K, float("inf"), dtype=np.float32)
    all_overlaps = np.zeros(args.K, dtype=np.int32)
    pos_K_legalized = pos_K.copy()
    for k, leg_hard, proxy, ovrlp in results_legalize:
        pos_K_legalized[k, :n_hard] = leg_hard
        all_proxies[k] = proxy
        all_overlaps[k] = ovrlp
    print(f"[gpu_run_one] legalize all: {time.time()-t_all:.1f}s "
          f"({n_workers} workers) [wall_elapsed={time.time()-t0:.1f}s]",
          flush=True)

    # ===== Distribution stats: видеть качество gradient'а, не только min =====
    valid_mask_for_stats = all_overlaps == 0
    n_valid = int(valid_mask_for_stats.sum())
    print(f"\n[gpu_run_one] === all-K stats (after legalize) ===", flush=True)
    print(f"[gpu_run_one] valid: {n_valid}/{args.K} "
          f"({100*n_valid/args.K:.1f}%)", flush=True)
    if n_valid > 0:
        valid_proxies = all_proxies[valid_mask_for_stats]
        stats_dict = {
            "min": float(valid_proxies.min()),
            "p05": float(np.percentile(valid_proxies, 5)),
            "p25": float(np.percentile(valid_proxies, 25)),
            "median": float(np.median(valid_proxies)),
            "mean": float(valid_proxies.mean()),
            "p75": float(np.percentile(valid_proxies, 75)),
            "p95": float(np.percentile(valid_proxies, 95)),
            "max": float(valid_proxies.max()),
            "std": float(valid_proxies.std()),
            "ci90_lo": float(np.percentile(valid_proxies, 5)),
            "ci90_hi": float(np.percentile(valid_proxies, 95)),
        }
        print(f"[gpu_run_one] proxy distribution (valid only):", flush=True)
        print(f"[gpu_run_one]   min     = {stats_dict['min']:.4f} "
              f"(seed k={int(np.argmin(np.where(valid_mask_for_stats, all_proxies, np.inf)))})",
              flush=True)
        print(f"[gpu_run_one]   p05/p25 = {stats_dict['p05']:.4f} / {stats_dict['p25']:.4f}",
              flush=True)
        print(f"[gpu_run_one]   median  = {stats_dict['median']:.4f}",
              flush=True)
        print(f"[gpu_run_one]   mean±σ  = {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}",
              flush=True)
        print(f"[gpu_run_one]   p75/p95 = {stats_dict['p75']:.4f} / {stats_dict['p95']:.4f}",
              flush=True)
        print(f"[gpu_run_one]   max     = {stats_dict['max']:.4f}",
              flush=True)
        print(f"[gpu_run_one]   90% CI  = [{stats_dict['ci90_lo']:.4f}, "
              f"{stats_dict['ci90_hi']:.4f}]", flush=True)
    else:
        stats_dict = None
        print(f"[gpu_run_one] no valid solutions in {args.K} seeds!", flush=True)

    # Update best from all-K (after legalize, more accurate than top-32 only)
    valid_mask = all_overlaps == 0
    if valid_mask.any():
        valid_idx = np.where(valid_mask)[0]
        leg_best_k = int(valid_idx[np.argmin(all_proxies[valid_idx])])
        leg_best_proxy = float(all_proxies[leg_best_k])
        if leg_best_proxy < best_proxy:
            print(f"[gpu_run_one] all-K legalize beat top-32: "
                  f"{leg_best_proxy:.4f} < {best_proxy:.4f} (k={leg_best_k})",
                  flush=True)
            best_proxy = leg_best_proxy
            best_idx = leg_best_k
            full_b = full_template.clone()
            full_b[:n_hard] = torch.tensor(pos_K_legalized[leg_best_k, :n_hard],
                                           dtype=torch.float32)
            if pos_K.shape[1] > n_hard:
                full_b[n_hard:n_hard + n_soft] = torch.tensor(
                    pos_K[leg_best_k, n_hard:n_hard + n_soft],
                    dtype=torch.float32)
            best_pos_full = full_b.cpu().numpy()

    best_orientations = [0] * n_hard

    orient_flip_enable = os.environ.get("STRAPLE_BATCH_ORIENT_FLIP", "0") == "1"
    if orient_flip_enable and valid_mask.any() and best_pos_full is not None:
        orient_topn = int(os.environ.get("STRAPLE_BATCH_ORIENT_TOPN", "16"))
        orient_rounds = int(os.environ.get("STRAPLE_BATCH_ORIENT_ROUNDS", "2"))

        sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple"))
        from orient_flip import (orient_flip_optimize,
                                 apply_orientations_to_plc,
                                 reset_orientations_to_n)

        valid_idx = np.where(valid_mask)[0]
        sorted_idx = valid_idx[np.argsort(all_proxies[valid_idx])]
        cands = [int(k) for k in sorted_idx[:orient_topn]]
        print(f"[gpu_run_one] orient_flip on top-{len(cands)} valid seeds "
              f"(rounds={orient_rounds})...", flush=True)
        t_orient = time.time()
        improved = 0
        best_oriented_proxy = best_proxy
        best_oriented_k = -1
        best_oriented_orient = None
        best_oriented_hard = None
        best_oriented_soft = None
        for k in cands:
            hard_pos_k = pos_K_legalized[k, :n_hard].astype(np.float64)
            if pos_K.shape[1] > n_hard:
                soft_pos_k = pos_K[k, n_hard:n_hard + n_soft].astype(np.float64)
            else:
                soft_pos_k = None
            orientations, _ = orient_flip_optimize(
                benchmark, hard_pos_k, soft_pos=soft_pos_k,
                rounds=orient_rounds, verbose=False,
            )
            n_changed = sum(1 for o in orientations if o != 0)
            if n_changed == 0:
                continue
            apply_orientations_to_plc(plc, benchmark, orientations)
            full_k = full_template.clone()
            full_k[:n_hard] = torch.tensor(hard_pos_k, dtype=torch.float32)
            if soft_pos_k is not None:
                full_k[n_hard:n_hard + n_soft] = torch.tensor(
                    soft_pos_k, dtype=torch.float32)
            c_k = compute_proxy_cost(full_k, benchmark, plc)
            reset_orientations_to_n(plc, benchmark)
            new_proxy = float(c_k["proxy_cost"])
            new_overlap = int(c_k["overlap_count"])
            if new_overlap == 0 and new_proxy < all_proxies[k] - 1e-6:
                improved += 1
                if new_proxy < best_oriented_proxy - 1e-6:
                    best_oriented_proxy = new_proxy
                    best_oriented_k = k
                    best_oriented_orient = list(orientations)
                    best_oriented_hard = hard_pos_k.copy()
                    best_oriented_soft = (soft_pos_k.copy()
                                           if soft_pos_k is not None else None)
        print(f"[gpu_run_one] orient_flip: {improved}/{len(cands)} improved, "
              f"{time.time()-t_orient:.1f}s", flush=True)
        if best_oriented_k >= 0:
            print(f"[gpu_run_one] orient_flip beat best: "
                  f"{best_oriented_proxy:.4f} < {best_proxy:.4f} "
                  f"(k={best_oriented_k})", flush=True)
            best_proxy = best_oriented_proxy
            best_idx = best_oriented_k
            best_orientations = best_oriented_orient
            full_b = full_template.clone()
            full_b[:n_hard] = torch.tensor(best_oriented_hard,
                                            dtype=torch.float32)
            if best_oriented_soft is not None:
                full_b[n_hard:n_hard + n_soft] = torch.tensor(
                    best_oriented_soft, dtype=torch.float32)
            best_pos_full = full_b.cpu().numpy()
            apply_orientations_to_plc(plc, benchmark, best_orientations)
        else:
            reset_orientations_to_n(plc, benchmark)

    cd_polish_enable = os.environ.get("STRAPLE_BATCH_CD_POLISH", "0") == "1"
    cd_gpu_enable = os.environ.get("STRAPLE_BATCH_CD_GPU_FILTER", "0") == "1"
    wall_tl = float(os.environ.get("STRAPLE_BATCH_WALL_TL", "0"))
    wall_reserve = float(os.environ.get("STRAPLE_BATCH_WALL_RESERVE", "30"))
    if cd_polish_enable and best_pos_full is not None and wall_tl > 0:
        elapsed_now = time.time() - t0
        wall_remaining = wall_tl - elapsed_now - wall_reserve
        if wall_remaining <= 0:
            print(f"[gpu_run_one] WALL_TL {wall_tl:.0f}s exhausted "
                  f"(elapsed {elapsed_now:.1f}s, reserve {wall_reserve:.0f}s) "
                  f"-- skipping CD polish", flush=True)
            cd_polish_enable = False
        else:
            print(f"[gpu_run_one] CD polish wall_remaining={wall_remaining:.0f}s "
                  f"(WALL_TL={wall_tl:.0f}s elapsed={elapsed_now:.1f}s)",
                  flush=True)
    if cd_polish_enable and best_pos_full is not None:
        sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple"))
        if cd_gpu_enable:
            from cd_polish import (cd_polish_gpu, cd_polish_gpu_with_restart,
                                    cluster_polish_gpu, pair_swap_polish_gpu)
            from gpu_proxy import (build_routing_edges_full,
                                    build_smooth_matrices,
                                    build_routing_consts,
                                    build_wl_pkg_full)
            cd_rounds = int(os.environ.get("STRAPLE_BATCH_CD_ROUNDS", "6"))
            cd_dirs = int(os.environ.get("STRAPLE_BATCH_CD_DIRS", "8"))
            cd_topk = int(os.environ.get("STRAPLE_BATCH_CD_GPU_TOPK", "3"))
            cd_macro_chunk = int(os.environ.get(
                "STRAPLE_BATCH_CD_GPU_MACRO_CHUNK", "64"))
            cd_time_budget = float(os.environ.get(
                "STRAPLE_BATCH_CD_GPU_TIME_BUDGET", "0"))
            if wall_tl > 0:
                wall_remaining = wall_tl - (time.time() - t0) - wall_reserve
                if cd_time_budget <= 0 or cd_time_budget > wall_remaining:
                    cd_time_budget = max(1.0, wall_remaining)
                    print(f"[gpu_run_one] cd_time_budget clamped to "
                          f"{cd_time_budget:.0f}s by WALL_TL", flush=True)
            cd_proxy_chunk_n = int(os.environ.get(
                "STRAPLE_BATCH_CD_GPU_PROXY_CHUNK_N", "32"))
            cd_approx_verify = os.environ.get(
                "STRAPLE_BATCH_CD_GPU_APPROX", "0") == "1"
            cd_approx_threshold = float(os.environ.get(
                "STRAPLE_BATCH_CD_GPU_APPROX_THRESHOLD", "1e-5"))
            cd_approx_refresh = os.environ.get(
                "STRAPLE_BATCH_CD_GPU_APPROX_REFRESH_PER_ACCEPT", "0") == "1"
            cd_top_n_seeds = int(os.environ.get(
                "STRAPLE_BATCH_CD_GPU_TOP_N_SEEDS", "1"))
            cd_restart_cycles = int(os.environ.get(
                "STRAPLE_BATCH_CD_GPU_RESTART_CYCLES", "0"))
            cd_jitter_frac = float(os.environ.get(
                "STRAPLE_BATCH_CD_GPU_JITTER_FRAC", "0.2"))
            cd_jitter_step = float(os.environ.get(
                "STRAPLE_BATCH_CD_GPU_JITTER_STEP", "0.5"))
            sf_str = os.environ.get(
                "STRAPLE_BATCH_CD_SF",
                "0.5,0.25,0.125,0.0625,0.03125,0.015625")
            cd_sf = tuple(float(x) for x in sf_str.split(",") if x.strip())
            print(f"[gpu_run_one] CD polish (GPU filter): "
                  f"rounds={cd_rounds} dirs={cd_dirs} topk={cd_topk} "
                  f"chunk={cd_macro_chunk} proxy_chunk_n={cd_proxy_chunk_n} "
                  f"approx_verify={cd_approx_verify} "
                  f"approx_threshold={cd_approx_threshold:g} "
                  f"top_n_seeds={cd_top_n_seeds} "
                  f"sf={cd_sf}", flush=True)
            name_to_global = {}
            for bidx, idx in enumerate(plc.hard_macro_indices):
                name_to_global[plc.modules_w_pins[idx].get_name()] = bidx
            for sidx, idx in enumerate(plc.soft_macro_indices):
                name_to_global[plc.modules_w_pins[idx].get_name()] = (
                    n_hard + sidx)
            n_total_full = benchmark.num_macros
            edges_pkg = build_routing_edges_full(
                plc, name_to_global, n_total_full)
            routing_consts = build_routing_consts(
                plc, float(benchmark.canvas_width),
                float(benchmark.canvas_height),
                int(benchmark.grid_rows), int(benchmark.grid_cols))
            smooth_matrices = build_smooth_matrices(
                int(benchmark.grid_rows), int(benchmark.grid_cols),
                routing_consts["smooth_range"])
            wl_pkg = build_wl_pkg_full(plc, name_to_global, n_total_full)
            proxy_pkgs_cd = {
                "edges_pkg": edges_pkg,
                "smooth_matrices": smooth_matrices,
                "routing_consts": routing_consts,
                "wl_pkg": wl_pkg,
            }
            t_cd = time.time()
            if cd_top_n_seeds > 1 and valid_mask.any():
                seed_top_idx = valid_idx[np.argsort(all_proxies[valid_idx])][:cd_top_n_seeds]
                print(f"[gpu_run_one] CD polish multi-seed top-{cd_top_n_seeds}: "
                      f"k={list(seed_top_idx)} "
                      f"proxies={[f'{all_proxies[k]:.4f}' for k in seed_top_idx]}",
                      flush=True)
                best_polished_proxy = best_proxy
                best_polished_pos = best_pos_full
                per_seed_budget = (cd_time_budget / cd_top_n_seeds
                                   if cd_time_budget > 0 else 0.0)
                for rank, k in enumerate(seed_top_idx):
                    full_seed = full_template.clone()
                    full_seed[:n_hard] = torch.tensor(
                        pos_K_legalized[int(k), :n_hard], dtype=torch.float32)
                    if pos_K.shape[1] > n_hard:
                        full_seed[n_hard:n_hard + n_soft] = torch.tensor(
                            pos_K[int(k), n_hard:n_hard + n_soft],
                            dtype=torch.float32)
                    seed_pos_full = full_seed.cpu().numpy()
                    seed_orig_proxy = float(all_proxies[int(k)])
                    print(f"[gpu_run_one] -- seed rank={rank+1}/{cd_top_n_seeds} "
                          f"k={int(k)} start_proxy={seed_orig_proxy:.4f}",
                          flush=True)
                    polished_seed_pos, polished_seed_proxy = cd_polish_gpu(
                        benchmark, plc, seed_pos_full,
                        proxy_pkgs=proxy_pkgs_cd,
                        rounds=cd_rounds, step_factors=cd_sf,
                        n_directions=cd_dirs, topk_verify=cd_topk,
                        macro_chunk=cd_macro_chunk,
                        time_budget=per_seed_budget,
                        proxy_chunk_n=cd_proxy_chunk_n,
                        approx_verify=cd_approx_verify,
                        approx_threshold=cd_approx_threshold,
                        approx_refresh_per_accept=cd_approx_refresh,
                        verbose=True)
                    if polished_seed_proxy < best_polished_proxy - 1e-6:
                        best_polished_proxy = polished_seed_proxy
                        best_polished_pos = polished_seed_pos.astype(np.float32)
                        print(f"[gpu_run_one] -- seed k={int(k)} NEW BEST: "
                              f"{polished_seed_proxy:.4f}", flush=True)
                polished_pos = best_polished_pos
                polished_proxy = best_polished_proxy
            else:
                polished_pos, polished_proxy = cd_polish_gpu_with_restart(
                    benchmark, plc, best_pos_full,
                    proxy_pkgs=proxy_pkgs_cd,
                    restart_cycles=cd_restart_cycles,
                    jitter_frac=cd_jitter_frac,
                    jitter_step=cd_jitter_step,
                    rounds=cd_rounds, step_factors=cd_sf,
                    n_directions=cd_dirs, topk_verify=cd_topk,
                    macro_chunk=cd_macro_chunk,
                    time_budget=cd_time_budget,
                    proxy_chunk_n=cd_proxy_chunk_n,
                    approx_verify=cd_approx_verify,
                    approx_threshold=cd_approx_threshold,
                    approx_refresh_per_accept=cd_approx_refresh,
                    verbose=True)
            cd_dt = time.time() - t_cd
        else:
            from cd_polish import cd_polish
            cd_rounds = int(os.environ.get("STRAPLE_BATCH_CD_ROUNDS", "3"))
            cd_dirs = int(os.environ.get("STRAPLE_BATCH_CD_DIRS", "8"))
            sf_str = os.environ.get(
                "STRAPLE_BATCH_CD_SF", "1.0,0.5,0.25,0.125")
            cd_sf = tuple(float(x) for x in sf_str.split(",") if x.strip())
            print(f"[gpu_run_one] CD polish on best seed: "
                  f"rounds={cd_rounds} dirs={cd_dirs} sf={cd_sf}", flush=True)
            t_cd = time.time()
            polished_pos, polished_proxy = cd_polish(
                benchmark, plc, best_pos_full,
                rounds=cd_rounds, step_factors=cd_sf,
                n_directions=cd_dirs, verbose=True)
            cd_dt = time.time() - t_cd
        if polished_proxy < best_proxy - 1e-6:
            print(f"[gpu_run_one] CD polish IMPROVED: {polished_proxy:.4f} "
                  f"< {best_proxy:.4f} ({cd_dt:.1f}s) "
                  f"[wall_elapsed={time.time()-t0:.1f}s]", flush=True)
            best_proxy = polished_proxy
            best_pos_full = polished_pos.astype(np.float32)
        else:
            print(f"[gpu_run_one] CD polish: {polished_proxy:.4f} "
                  f"(no improvement vs {best_proxy:.4f}, {cd_dt:.1f}s) "
                  f"[wall_elapsed={time.time()-t0:.1f}s]", flush=True)

        cluster_enable = (os.environ.get(
            "STRAPLE_BATCH_CLUSTER_POLISH", "0") == "1"
            and cd_gpu_enable and best_pos_full is not None)
        if cluster_enable:
            cluster_n = int(os.environ.get(
                "STRAPLE_BATCH_CLUSTER_N", "30"))
            cluster_rounds = int(os.environ.get(
                "STRAPLE_BATCH_CLUSTER_ROUNDS", "4"))
            cluster_grid = int(os.environ.get(
                "STRAPLE_BATCH_CLUSTER_GRID", "5"))
            sf_cluster_str = os.environ.get(
                "STRAPLE_BATCH_CLUSTER_SF",
                "0.5,0.25,0.125,0.0625")
            cluster_sf = tuple(float(x) for x in sf_cluster_str.split(",")
                                 if x.strip())
            cluster_tb = float(os.environ.get(
                "STRAPLE_BATCH_CLUSTER_TIME_BUDGET", "0"))
            if wall_tl > 0:
                wall_remaining = wall_tl - (time.time() - t0) - wall_reserve
                if cluster_tb <= 0 or cluster_tb > wall_remaining:
                    cluster_tb = max(1.0, wall_remaining)
            print(f"[gpu_run_one] CLUSTER polish: n_clusters={cluster_n} "
                  f"rounds={cluster_rounds} grid={cluster_grid} "
                  f"sf={cluster_sf} budget={cluster_tb:.0f}s", flush=True)
            t_cl = time.time()
            cl_pos, cl_proxy = cluster_polish_gpu(
                benchmark, plc, best_pos_full,
                proxy_pkgs=proxy_pkgs_cd,
                n_clusters=cluster_n,
                n_rounds=cluster_rounds,
                sf_list=cluster_sf,
                n_grid=cluster_grid,
                verbose=True,
                time_budget=cluster_tb,
                proxy_chunk_n=cd_proxy_chunk_n)
            cl_dt = time.time() - t_cl
            if cl_proxy < best_proxy - 1e-6:
                print(f"[gpu_run_one] CLUSTER polish IMPROVED: "
                      f"{cl_proxy:.4f} < {best_proxy:.4f} ({cl_dt:.1f}s) "
                      f"[wall_elapsed={time.time()-t0:.1f}s]", flush=True)
                best_proxy = cl_proxy
                best_pos_full = cl_pos.astype(np.float32)
            else:
                print(f"[gpu_run_one] CLUSTER polish: {cl_proxy:.4f} "
                      f"(no improvement vs {best_proxy:.4f}, {cl_dt:.1f}s)",
                      flush=True)

    pswap_enable = (os.environ.get("STRAPLE_BATCH_PAIR_SWAP", "0") == "1"
                     and cd_gpu_enable and best_pos_full is not None)
    if pswap_enable:
        pswap_neighbors = int(os.environ.get(
            "STRAPLE_BATCH_PAIR_SWAP_NEIGHBORS", "10"))
        pswap_rounds = int(os.environ.get(
            "STRAPLE_BATCH_PAIR_SWAP_ROUNDS", "5"))
        pswap_chunk = int(os.environ.get(
            "STRAPLE_BATCH_PAIR_SWAP_CHUNK", "256"))
        pswap_tb = float(os.environ.get(
            "STRAPLE_BATCH_PAIR_SWAP_TIME_BUDGET", "0"))
        if wall_tl > 0:
            wall_remaining = wall_tl - (time.time() - t0) - wall_reserve
            if pswap_tb <= 0 or pswap_tb > wall_remaining:
                pswap_tb = max(1.0, wall_remaining)
        print(f"[gpu_run_one] PAIR_SWAP: neighbors={pswap_neighbors} "
              f"rounds={pswap_rounds} chunk={pswap_chunk} "
              f"budget={pswap_tb:.0f}s", flush=True)
        t_ps = time.time()
        ps_pos, ps_proxy = pair_swap_polish_gpu(
            benchmark, plc, best_pos_full,
            proxy_pkgs=proxy_pkgs_cd,
            n_neighbors=pswap_neighbors,
            n_rounds=pswap_rounds,
            verbose=True,
            time_budget=pswap_tb,
            proxy_chunk_n=cd_proxy_chunk_n,
            chunk_pairs=pswap_chunk)
        ps_dt = time.time() - t_ps
        if ps_proxy < best_proxy - 1e-6:
            print(f"[gpu_run_one] PAIR_SWAP IMPROVED: "
                  f"{ps_proxy:.4f} < {best_proxy:.4f} ({ps_dt:.1f}s) "
                  f"[wall_elapsed={time.time()-t0:.1f}s]", flush=True)
            best_proxy = ps_proxy
            best_pos_full = ps_pos.astype(np.float32)
        else:
            print(f"[gpu_run_one] PAIR_SWAP: {ps_proxy:.4f} "
                  f"(no improvement vs {best_proxy:.4f}, {ps_dt:.1f}s)",
                  flush=True)

    pr_cycles = int(os.environ.get("STRAPLE_BATCH_PR_CYCLES", "0"))
    if (pr_cycles > 0 and best_pos_full is not None and cd_polish_enable
            and cd_gpu_enable):
        from perturb_relax import perturb_relax_cycles
        pr_K = int(os.environ.get("STRAPLE_BATCH_PR_K", "8"))
        pr_steps = int(os.environ.get("STRAPLE_BATCH_PR_STEPS", "100"))
        pr_time_budget = float(os.environ.get(
            "STRAPLE_BATCH_PR_TIME_BUDGET", "20"))
        pr_perturb_frac = float(os.environ.get(
            "STRAPLE_BATCH_PR_PERTURB_FRAC", "0.25"))
        pr_perturb_step = float(os.environ.get(
            "STRAPLE_BATCH_PR_PERTURB_STEP", "0.5"))
        pr_seed = int(os.environ.get("STRAPLE_BATCH_PR_SEED", "777"))
        wall_deadline = (t0 + wall_tl - wall_reserve) if wall_tl > 0 else 0.0
        from cd_polish import cd_polish_gpu as _cd_polish_gpu_fn
        from macro_place.objective import compute_proxy_cost as _cpc_fn

        def _grad_fn(bench, plc_arg, K, num_steps, time_budget,
                      seed, init_pos_override, verbose, **kw):
            return gradient_batch(
                bench, plc_arg, K=K, num_steps=num_steps,
                time_budget=time_budget, seed=seed,
                device="cuda" if torch.cuda.is_available() else "cpu",
                init_pos_override=init_pos_override,
                anchor_strategy=os.environ.get(
                    "STRAPLE_BATCH_ANCHOR_STRATEGY", "centroid"),
                spawn_radius_frac=0.05, spawn_adaptive=True,
                anchor_jitter_frac=0.05,
                anchor_loss_beta_start=anchor_beta_start,
                anchor_loss_beta_end=anchor_beta_end,
                cohesion_beta_start=cohesion_beta_start,
                cohesion_beta_end=cohesion_beta_end,
                use_eplace_density=use_eplace,
                eplace_grid_size=eplace_grid_size,
                cong_weight=cong_weight,
                per_k_diversity=per_k_diversity,
                lr=grad_lr, verbose=verbose,
            )

        def _cd_fn(bench, plc_arg, pos_full_in, **kw):
            return _cd_polish_gpu_fn(
                bench, plc_arg, pos_full_in,
                proxy_pkgs=proxy_pkgs_cd,
                rounds=cd_rounds, step_factors=cd_sf,
                n_directions=cd_dirs, topk_verify=cd_topk,
                macro_chunk=cd_macro_chunk,
                time_budget=cd_time_budget,
                proxy_chunk_n=cd_proxy_chunk_n,
                approx_verify=cd_approx_verify,
                approx_threshold=cd_approx_threshold,
                approx_refresh_per_accept=cd_approx_refresh,
                verbose=False)

        print(f"[gpu_run_one] PERTURB-RELAX: cycles={pr_cycles} K={pr_K} "
              f"steps={pr_steps} time_budget={pr_time_budget:.0f}s "
              f"perturb_frac={pr_perturb_frac:.2f} "
              f"perturb_step={pr_perturb_step:.2f} "
              f"[wall_elapsed={time.time()-t0:.1f}s "
              f"wall_remaining={(wall_deadline-time.time()):.0f}s]",
              flush=True)
        t_pr = time.time()
        pr_pos, pr_proxy = perturb_relax_cycles(
            benchmark, plc, best_pos_full,
            gradient_batch_fn=_grad_fn,
            cd_polish_fn=_cd_fn,
            legalize_fn=_legalize_only,
            compute_proxy_fn=_cpc_fn,
            n_cycles=pr_cycles,
            mini_K=pr_K,
            mini_steps=pr_steps,
            mini_time_budget=pr_time_budget,
            perturb_frac=pr_perturb_frac,
            perturb_step=pr_perturb_step,
            seed=pr_seed,
            wall_deadline=wall_deadline,
            verbose=True)
        pr_dt = time.time() - t_pr
        if pr_proxy < best_proxy - 1e-6:
            print(f"[gpu_run_one] PERTURB-RELAX IMPROVED: "
                  f"{pr_proxy:.4f} < {best_proxy:.4f} ({pr_dt:.1f}s) "
                  f"[wall_elapsed={time.time()-t0:.1f}s]", flush=True)
            best_proxy = pr_proxy
            best_pos_full = pr_pos.astype(np.float32)
        else:
            print(f"[gpu_run_one] PERTURB-RELAX: {pr_proxy:.4f} "
                  f"(no improvement vs {best_proxy:.4f}, {pr_dt:.1f}s)",
                  flush=True)

    grid_path = REPO_ROOT / "results" / f"gpu_pos_K_{args.bench}.npz"
    np.savez_compressed(
        grid_path,
        pos_K=pos_K_legalized.astype(np.float32),
        proxies=all_proxies,
        overlaps=all_overlaps,
        best_idx=np.array([best_idx], dtype=np.int32),
        best_orientations=np.array(best_orientations, dtype=np.int8),
    )
    print(f"[gpu_run_one] saved legalized pos_K to {grid_path} "
          f"({grid_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # ===== Per-step proxy_cost trajectory across ALL K seeds (GPU batch) =====
    use_gpu_proxy = os.environ.get("STRAPLE_GPU_PROXY", "1") == "1"
    if (snapshots_pos_all is not None and len(snapshots_step_all) > 0
            and use_gpu_proxy and torch.cuda.is_available()):
        sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple"))
        from gpu_proxy import gpu_proxy_batched
        from analytical_seed import (
            _build_net_pin_tensors_full, _build_padded_net_tensors)
        net_macro_idx, net_pin_offsets = _build_net_pin_tensors_full(
            benchmark, plc)
        padded = _build_padded_net_tensors(net_macro_idx, net_pin_offsets)
        macro_idx_p, offsets_p, mask_p = padded
        dev = torch.device("cuda")
        macro_idx_p = macro_idx_p.to(dev)
        offsets_p = offsets_p.to(dev)
        mask_p = mask_p.to(dev)
        sizes_t = benchmark.macro_sizes[:n_hard + n_soft].float().to(dev)
        num_nets_used = int(macro_idx_p.shape[0])

        n_snap = len(snapshots_step_all)
        proxy_max_snap = int(os.environ.get("STRAPLE_PROXY_TRAJ_SNAPS", "30"))
        sub_n = min(n_snap, proxy_max_snap)
        sub_indices = np.linspace(0, n_snap - 1, sub_n).astype(int)
        proxy_arr = np.full((sub_n, args.K), np.nan, dtype=np.float32)
        wl_arr = np.zeros((sub_n, args.K), dtype=np.float32)
        den_arr = np.zeros((sub_n, args.K), dtype=np.float32)
        cong_arr = np.zeros((sub_n, args.K), dtype=np.float32)
        print(f"[gpu_run_one] gpu_proxy traj: {sub_n} snapshots × "
              f"{args.K} seeds on GPU...", flush=True)
        t_proxy = time.time()
        for snap_i, idx in enumerate(sub_indices):
            pos_K_t = torch.from_numpy(
                snapshots_pos_all[idx]).to(dev).float()      # [K, n, 2]
            with torch.no_grad():
                p_K, comp = gpu_proxy_batched(
                    pos_K_t, sizes_t, macro_idx_p, offsets_p, mask_p,
                    float(benchmark.canvas_width),
                    float(benchmark.canvas_height),
                    int(benchmark.grid_rows), int(benchmark.grid_cols),
                    num_nets_used,
                )
            proxy_arr[snap_i] = p_K.cpu().numpy()
            wl_arr[snap_i] = comp["wl"].cpu().numpy()
            den_arr[snap_i] = comp["density"].cpu().numpy()
            cong_arr[snap_i] = comp["congestion"].cpu().numpy()
        proxy_traj = {
            "steps": np.array([int(snapshots_step_all[i])
                               for i in sub_indices]),
            "proxy_K": proxy_arr,
            "overlap_K": np.zeros_like(proxy_arr, dtype=np.int32),
            "wl_K": wl_arr,
            "density_K": den_arr,
            "congestion_K": cong_arr,
        }
        print(f"[gpu_run_one] gpu_proxy done in {time.time()-t_proxy:.2f}s",
              flush=True)

    # ===== Evolution plot: fitness + overlap_area + proxy_cost panels =====
    if fit_hist is not None and len(fit_hist) > 0:
        plot_path = REPO_ROOT / "vis" / f"{args.bench}_evolution.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            steps = np.arange(1, fit_hist.shape[0] + 1)
            f_min = fit_hist.min(axis=1)
            f_p25 = np.percentile(fit_hist, 25, axis=1)
            f_med = np.median(fit_hist, axis=1)
            f_p75 = np.percentile(fit_hist, 75, axis=1)
            f_max = fit_hist.max(axis=1)
            n_panels = 2 + (1 if proxy_traj is not None else 0)
            fig, axes = plt.subplots(n_panels, 1,
                                     figsize=(12, 4 * n_panels), sharex=True)
            if n_panels == 1:
                axes = [axes]
            ax = axes[0]
            ax.fill_between(steps, f_min, f_max, alpha=0.15,
                            color="C0", label="min..max")
            ax.fill_between(steps, f_p25, f_p75, alpha=0.30,
                            color="C0", label="p25..p75")
            ax.plot(steps, f_med, color="C0", lw=1.5, label="median")
            ax.plot(steps, f_min, color="C2", lw=1.0, label="min (best)")
            ax.set_yscale("log")
            ax.set_ylabel("fitness (loss)")
            ax.set_title(f"{args.bench}: per-step fitness across K={args.K} seeds")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            if ov_hist is not None and len(ov_hist) > 0:
                ax2 = axes[1]
                o_min = ov_hist.min(axis=1)
                o_med = np.median(ov_hist, axis=1)
                o_max = ov_hist.max(axis=1)
                ax2.fill_between(steps, o_min, o_max, alpha=0.20,
                                 color="C3", label="min..max")
                ax2.plot(steps, o_med, color="C3", lw=1.2, label="median")
                ax2.plot(steps, o_min, color="C2", lw=1.0, label="min")
                ax2.set_yscale("symlog")
                ax2.set_ylabel("overlap area")
                ax2.legend(loc="upper right", fontsize=8)
                ax2.grid(True, alpha=0.3)
            if proxy_traj is not None:
                ax3 = axes[-1]
                ps = proxy_traj["steps"]
                P = proxy_traj["proxy_K"]
                ovK = proxy_traj["overlap_K"]
                p_min = np.nanmin(P, axis=1)
                p_p25 = np.nanpercentile(P, 25, axis=1)
                p_med = np.nanmedian(P, axis=1)
                p_p75 = np.nanpercentile(P, 75, axis=1)
                p_max = np.nanmax(P, axis=1)
                p_min_valid = np.full(P.shape[0], np.nan, dtype=np.float32)
                for i in range(P.shape[0]):
                    valid_in_row = (ovK[i] == 0)
                    if valid_in_row.any():
                        p_min_valid[i] = float(P[i][valid_in_row].min())
                ax3.fill_between(ps, p_min, p_max, alpha=0.15,
                                 color="C4", label="all K min..max")
                ax3.fill_between(ps, p_p25, p_p75, alpha=0.30,
                                 color="C4", label="p25..p75")
                ax3.plot(ps, p_med, color="C4", lw=1.5, label="median")
                ax3.plot(ps, p_min, color="#cc6677", lw=1.0,
                         label="min (any seed, may be invalid)")
                if np.isfinite(p_min_valid).any():
                    ax3.plot(ps, p_min_valid, color="#117733", lw=1.5,
                             marker="o", ms=3,
                             label="min VALID (overlap=0)")
                ax3.set_ylabel("proxy_cost (raw, no legalize)")
                ax3.set_xlabel("step")
                ax3.legend(loc="upper right", fontsize=8)
                ax3.grid(True, alpha=0.3)
                ax3.set_title("proxy_cost across all K seeds (raw, no legalize)")
            else:
                axes[-1].set_xlabel("step")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=110)
            plt.close(fig)
            print(f"[gpu_run_one] evolution plot -> {plot_path}", flush=True)
        except Exception as e:
            print(f"[gpu_run_one] plot skipped: {e}", flush=True)
        hist_path = REPO_ROOT / "results" / f"gpu_history_{args.bench}.npz"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        npz_payload = {"fitness": fit_hist.astype(np.float32)}
        if ov_hist is not None:
            npz_payload["overlap_area"] = ov_hist.astype(np.float32)
        if proxy_traj is not None:
            npz_payload["proxy_traj_steps"] = proxy_traj["steps"]
            npz_payload["proxy_traj_K"] = proxy_traj["proxy_K"]
            npz_payload["proxy_traj_overlap"] = proxy_traj["overlap_K"]
        np.savez_compressed(hist_path, **npz_payload)
        print(f"[gpu_run_one] history -> {hist_path} "
              f"({hist_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # Save distribution stats JSON for offline config comparison
    if stats_dict is not None:
        import json
        stats_path = REPO_ROOT / "results" / f"gpu_stats_{args.bench}.json"
        cfg = {k: v for k, v in os.environ.items()
               if k.startswith("STRAPLE_BATCH_") or k.startswith("STRAPLE_DEMO_")}
        with open(stats_path, "w") as f:
            json.dump({
                "bench": args.bench,
                "K": args.K,
                "time_budget": args.time_budget,
                "n_valid": int(n_valid),
                "stats": stats_dict,
                "config": cfg,
                "best_idx": int(best_idx),
                "best_orientations": [int(o) for o in best_orientations],
            }, f, indent=2)
        print(f"[gpu_run_one] stats saved to {stats_path}", flush=True)

    if args.no_vis:
        return

    # Phase 3: simple JS canvas HTML viz using ACTUAL snapshots from
    # gradient_batch (best seed). Snapshot каждый step pos[best_idx] от GPU
    # batch run — это РЕАЛЬНАЯ траектория с ePlace+cong+anchor. Браузер
    # рендерит каждый кадр через canvas — никакого matplotlib.
    print(f"\n[gpu_run_one] simple HTML viz (k={best_idx})...", flush=True)
    vis_path = REPO_ROOT / "vis" / f"{args.bench}_gpu_best.html"
    vis_path.parent.mkdir(parents=True, exist_ok=True)

    from clustering import cluster_macros
    from make_simple_viz import render_simple_html

    snapshots_pos = stats.get("snapshots_pos", None)
    snapshots_step = stats.get("snapshots_step", [])
    cluster_target = max(15, benchmark.num_macros // 30)
    cluster_id, num_clusters, _ = cluster_macros(
        benchmark, method="louvain", seed=42,
        max_net_size=20, target_num_clusters=cluster_target,
    )

    if snapshots_pos is not None and len(snapshots_step) > 0:
        # Subsample frames for HTML viz (proxy compute is slow, ~2s per call).
        viz_max_frames = int(os.environ.get("STRAPLE_HTML_VIZ_FRAMES", "30"))
        n_snap_total = len(snapshots_step)
        sub_idx = (np.linspace(0, n_snap_total - 1,
                               min(n_snap_total, viz_max_frames))
                   .astype(int))
        traj = snapshots_pos[sub_idx, best_idx, :, :]   # [num_frames, n_total, 2]
        sub_steps = [int(snapshots_step[i]) for i in sub_idx]
        from macro_place.objective import compute_proxy_cost
        full_template = benchmark.macro_positions.clone()
        proxies = []
        labels = []
        t_proxy = time.time()
        # Intermediate snapshots are from gradient_batch which uses orientation N.
        # Make sure plc is in N state for these proxy calls. The final frame uses
        # the chosen best_orientations and is re-applied below.
        if any(o != 0 for o in best_orientations):
            sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple"))
            from orient_flip import (apply_orientations_to_plc,
                                     reset_orientations_to_n)
            reset_orientations_to_n(plc, benchmark)
        wl_costs = []
        density_costs = []
        cong_costs = []
        overlap_counts = []
        overlap_areas = []
        for i, step_n in enumerate(sub_steps):
            full = full_template.clone()
            full[:traj[i].shape[0]] = torch.tensor(traj[i], dtype=torch.float32)
            c = compute_proxy_cost(full, benchmark, plc)
            proxies.append(float(c["proxy_cost"]))
            wl_costs.append(float(c["wirelength_cost"]))
            density_costs.append(float(c["density_cost"]))
            cong_costs.append(float(c["congestion_cost"]))
            overlap_counts.append(int(c["overlap_count"]))
            overlap_areas.append(float(c["total_overlap_area"]))
            labels.append(f"step={step_n}")
        if any(o != 0 for o in best_orientations):
            apply_orientations_to_plc(plc, benchmark, best_orientations)
        full_f = torch.tensor(best_pos_full, dtype=torch.float32)
        cf = compute_proxy_cost(full_f, benchmark, plc)
        proxies.append(float(cf["proxy_cost"]))
        wl_costs.append(float(cf["wirelength_cost"]))
        density_costs.append(float(cf["density_cost"]))
        cong_costs.append(float(cf["congestion_cost"]))
        overlap_counts.append(int(cf["overlap_count"]))
        overlap_areas.append(float(cf["total_overlap_area"]))
        labels.append(f"FINAL legalized k={best_idx} proxy={best_proxy:.4f}")
        print(f"[gpu_run_one] proxy compute (HTML, "
              f"{len(sub_steps)} frames): {time.time()-t_proxy:.1f}s",
              flush=True)

        t_render = time.time()
        render_simple_html(
            str(vis_path), benchmark, traj, sub_steps,
            cluster_id, proxies, labels,
            final_pos=best_pos_full,
            final_proxy=best_proxy,
            final_label=labels[-1],
            wl_costs=wl_costs,
            density_costs=density_costs,
            cong_costs=cong_costs,
            overlap_counts=overlap_counts,
            overlap_areas=overlap_areas,
        )
        print(f"[gpu_run_one] HTML render: {time.time()-t_render:.1f}s",
              flush=True)
    else:
        print("[gpu_run_one] no snapshots, skipping HTML", flush=True)

    print(f"\n[gpu_run_one] === DONE === total={time.time()-t0:.1f}s "
          f"best_proxy={best_proxy:.4f}", flush=True)


if __name__ == "__main__":
    main()
