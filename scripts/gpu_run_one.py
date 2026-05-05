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
    pos_K, stats = gradient_batch(
        benchmark, plc, K=args.K,
        num_steps=20000,
        time_budget=args.time_budget,
        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",
        anchor_strategy="centroid",
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
    )
    grad_time = time.time() - t0
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"[gpu_run_one] GPU peak memory: {peak_mb:.0f} MB", flush=True)
    print(f"[gpu_run_one] gradient batch: {grad_time:.1f}s "
          f"({grad_time / args.K * 1000:.1f}ms per seed)", flush=True)

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
    print(f"[gpu_run_one] eval+legalize: {time.time()-t_eval:.1f}s", flush=True)
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
          f"({n_workers} workers)", flush=True)

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

    grid_path = REPO_ROOT / "results" / f"gpu_pos_K_{args.bench}.npz"
    np.savez_compressed(
        grid_path,
        pos_K=pos_K_legalized.astype(np.float32),
        proxies=all_proxies,
        overlaps=all_overlaps,
        best_idx=np.array([best_idx], dtype=np.int32),
    )
    print(f"[gpu_run_one] saved legalized pos_K to {grid_path} "
          f"({grid_path.stat().st_size / 1e6:.1f} MB)", flush=True)

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
        traj = snapshots_pos[:, best_idx, :, :]   # [num_snap, n_total, 2]
        # Compute proxy for each snapshot — это медленно (sequential
        # compute_proxy_cost) для 250 frames, но даёт точную метрику. Можем
        # ускорить если нужно.
        from macro_place.objective import compute_proxy_cost
        full_template = benchmark.macro_positions.clone()
        proxies = []
        labels = []
        t_proxy = time.time()
        for i, step_n in enumerate(snapshots_step):
            full = full_template.clone()
            full[:traj[i].shape[0]] = torch.tensor(traj[i], dtype=torch.float32)
            c = compute_proxy_cost(full, benchmark, plc)
            proxies.append(float(c["proxy_cost"]))
            labels.append(f"step={step_n}")
            if (i + 1) % 50 == 0:
                print(f"[gpu_run_one] proxy snapshots {i+1}/{len(snapshots_step)}",
                      flush=True)
        # Final legalized
        full_f = torch.tensor(best_pos_full, dtype=torch.float32)
        cf = compute_proxy_cost(full_f, benchmark, plc)
        proxies.append(float(cf["proxy_cost"]))
        labels.append(f"FINAL legalized k={best_idx} proxy={best_proxy:.4f}")
        print(f"[gpu_run_one] proxy compute: {time.time()-t_proxy:.1f}s",
              flush=True)

        t_render = time.time()
        render_simple_html(
            str(vis_path), benchmark, traj, snapshots_step,
            cluster_id, proxies, labels,
            final_pos=best_pos_full,
            final_proxy=best_proxy,
            final_label=labels[-1],
        )
        print(f"[gpu_run_one] HTML render: {time.time()-t_render:.1f}s",
              flush=True)
    else:
        print("[gpu_run_one] no snapshots, skipping HTML", flush=True)

    print(f"\n[gpu_run_one] === DONE === total={time.time()-t0:.1f}s "
          f"best_proxy={best_proxy:.4f}", flush=True)


if __name__ == "__main__":
    main()
