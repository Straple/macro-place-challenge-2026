"""GPU batch multi-start runner: K parallel gradient seeds в один torch tensor.

Использует T4 по максимуму — vectorized [K, n, 2] tensor, K seeds одновременно.
После backward'а — единый GPU kernel call. На T4 даёт реальный speedup vs
sequential gradient_demo.

Usage:
    uv run python scripts/gpu_batch_search.py --bench ibm01 --K 64 --steps 400
    uv run python scripts/gpu_batch_search.py --bench ibm17 --K 32 --steps 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple"))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--K", type=int, default=64, help="batch parallel runs")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--anchor-strategy", default="centroid")
    parser.add_argument("--spawn-radius", type=float, default=0.05)
    parser.add_argument("--anchor-jitter", type=float, default=0.05,
                        help="anchor jitter frac for K diversity")
    parser.add_argument("--no-spawn-adaptive", action="store_true")
    parser.add_argument("--legalize", action="store_true",
                        help="apply C++ legalize to top-N candidates")
    parser.add_argument("--legalize-topn", type=int, default=8)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    bdir = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / args.bench
    benchmark, plc = load_benchmark_from_dir(str(bdir))
    n_hard = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros
    print(f"[gpu_batch] {args.bench}: hard={n_hard} soft={n_soft} "
          f"nets={benchmark.num_nets}", flush=True)

    c0 = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
    print(f"[gpu_batch] INITIAL proxy={c0['proxy_cost']:.4f} "
          f"ovrlp={c0['overlap_count']}", flush=True)

    from gradient_batch import gradient_batch

    t_total = time.time()
    pos_K, _ = gradient_batch(
        benchmark, plc, K=args.K, num_steps=args.steps, seed=args.seed_base,
        device=args.device, anchor_strategy=args.anchor_strategy,
        spawn_radius_frac=args.spawn_radius,
        spawn_adaptive=not args.no_spawn_adaptive,
        anchor_jitter_frac=args.anchor_jitter,
    )
    grad_time = time.time() - t_total
    print(f"[gpu_batch] gradient batch: {grad_time:.1f}s "
          f"({grad_time / args.K * 1000:.1f}ms per seed)", flush=True)
    # pos_K: [K, n_total, 2]
    print(f"[gpu_batch] pos shape={pos_K.shape}", flush=True)

    # Evaluate proxy for each seed
    import torch
    results = []
    best_proxy = float("inf")
    best_idx = -1
    best_pos = None
    full_template = benchmark.macro_positions.clone()
    t_eval = time.time()
    for k in range(args.K):
        full = full_template.clone()
        full[:n_hard] = torch.tensor(pos_K[k, :n_hard], dtype=torch.float32)
        if pos_K.shape[1] > n_hard:
            full[n_hard:n_hard + n_soft] = torch.tensor(
                pos_K[k, n_hard:n_hard + n_soft], dtype=torch.float32)
        cost = compute_proxy_cost(full, benchmark, plc)
        valid = cost["overlap_count"] == 0
        results.append({
            "k": int(k),
            "proxy": float(cost["proxy_cost"]),
            "wl": float(cost["wirelength_cost"]),
            "den": float(cost["density_cost"]),
            "cong": float(cost["congestion_cost"]),
            "ovrlp": int(cost["overlap_count"]),
            "valid": bool(valid),
        })
        if valid and cost["proxy_cost"] < best_proxy:
            best_proxy = cost["proxy_cost"]
            best_idx = k
            best_pos = pos_K[k].copy()
    print(f"[gpu_batch] eval: {time.time()-t_eval:.1f}s", flush=True)

    proxies_valid = [r["proxy"] for r in results if r["valid"]]
    print(f"\n[gpu_batch] DONE total={time.time()-t_total:.1f}s")
    print(f"[gpu_batch] {len(proxies_valid)}/{args.K} valid")
    if proxies_valid:
        print(f"[gpu_batch] best={best_proxy:.4f} (k={best_idx})")
        print(f"[gpu_batch] mean={np.mean(proxies_valid):.4f} "
              f"median={np.median(proxies_valid):.4f} "
              f"std={np.std(proxies_valid):.4f} "
              f"worst={max(proxies_valid):.4f}")
        sorted_by_proxy = sorted(results, key=lambda r: r["proxy"] if r["valid"] else float("inf"))
        print(f"[gpu_batch] top10 proxies: "
              f"{[round(r['proxy'], 4) for r in sorted_by_proxy[:10]]}")

    # Optional: apply legalize on topN best, see if proxy improves
    if args.legalize and best_pos is not None:
        print(f"\n[gpu_batch] legalize top-{args.legalize_topn} candidates", flush=True)
        sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple" / "cpp"))
        import _placer_core
        sizes_np = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64)
        movable_np = benchmark.get_movable_mask()[:n_hard].cpu().numpy().astype(bool)
        sorted_idx = sorted(range(args.K),
                            key=lambda k: results[k]["proxy"] if results[k]["valid"] else float("inf"))
        leg_best_proxy = float("inf")
        leg_best_pos = None
        for rank, k in enumerate(sorted_idx[:args.legalize_topn]):
            cur_pos_hard = pos_K[k, :n_hard].astype(np.float64).copy()
            state = _placer_core.PlacerState()
            state.initialize(
                cur_pos_hard, sizes_np, movable_np,
                np.zeros((0, 2), dtype=np.int32),
                np.zeros(0, dtype=np.float64),
                float(benchmark.canvas_width), float(benchmark.canvas_height),
                int(args.seed_base + k),
            )
            state.legalize_min_displacement(500)
            state.legalize()
            leg_hard = state.current_positions()
            full = full_template.clone()
            full[:n_hard] = torch.tensor(leg_hard, dtype=torch.float32)
            if pos_K.shape[1] > n_hard:
                full[n_hard:n_hard + n_soft] = torch.tensor(
                    pos_K[k, n_hard:n_hard + n_soft], dtype=torch.float32)
            c = compute_proxy_cost(full, benchmark, plc)
            print(f"[gpu_batch] leg k={k}: proxy={c['proxy_cost']:.4f} "
                  f"ovrlp={c['overlap_count']} "
                  f"(was {results[k]['proxy']:.4f})", flush=True)
            if c["overlap_count"] == 0 and c["proxy_cost"] < leg_best_proxy:
                leg_best_proxy = c["proxy_cost"]
                leg_pos_full = full.clone().cpu().numpy()
                leg_best_pos = leg_pos_full
        if leg_best_pos is not None:
            print(f"\n[gpu_batch] LEG BEST proxy={leg_best_proxy:.4f}")
            best_proxy = min(best_proxy, leg_best_proxy)

    out = {
        "bench": args.bench,
        "K": int(args.K),
        "steps": int(args.steps),
        "device": args.device,
        "initial_proxy": float(c0["proxy_cost"]),
        "best_proxy": float(best_proxy) if proxies_valid else None,
        "best_idx": int(best_idx),
        "mean_valid": float(np.mean(proxies_valid)) if proxies_valid else None,
        "wall_time": float(time.time() - t_total),
        "grad_time": float(grad_time),
        "results": results,
    }
    out_path = Path(args.output_json) if args.output_json else (
        REPO_ROOT / "results" / f"gpu_batch_{args.bench}_{time.strftime('%Y%m%dT%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gpu_batch] saved {out_path}", flush=True)

    # Save best pos for ALNS polish
    if best_pos is not None:
        import pickle
        seed_path = REPO_ROOT / "results" / f"gpu_seed_{args.bench}.pkl"
        with open(seed_path, "wb") as f:
            pickle.dump({
                "hard": best_pos[:n_hard].astype(np.float64),
                "soft": best_pos[n_hard:n_hard + n_soft].astype(np.float64),
                "proxy": float(best_proxy),
                "K": int(args.K),
                "steps": int(args.steps),
            }, f)
        print(f"[gpu_batch] saved seed {seed_path} (proxy={best_proxy:.4f})",
              flush=True)


if __name__ == "__main__":
    main()
