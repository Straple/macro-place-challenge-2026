"""CPU-parallel multi-seed gradient search.

Запускает gradient_demo K раз через ProcessPoolExecutor (W workers).
Намного быстрее чем sequential GPU для маленьких benches: T4 launch overhead
~5-10ms per CUDA kernel, а тут много мелких ops -> GPU underutilized.

Usage:
    uv run python scripts/cpu_parallel_search.py --bench ibm01 --num-seeds 32 --workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple"))


def _worker(args):
    bench, seed, steps, env_overrides = args
    for k, v in env_overrides.items():
        os.environ[k] = v
    os.environ["STRAPLE_DEMO_DEVICE"] = "cpu"

    sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple"))
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from gradient_demo import gradient_demo
    import torch

    bdir = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04" / bench
    benchmark, plc = load_benchmark_from_dir(str(bdir))
    n_hard = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros

    t0 = time.time()
    result = gradient_demo(benchmark, plc, recorder=None,
                           num_steps=steps, seed=seed,
                           time_budget=0.0, score_png="")
    elapsed = time.time() - t0
    pos = result[0] if isinstance(result, tuple) else result
    pos = np.asarray(pos, dtype=np.float64)
    full = benchmark.macro_positions.clone()
    full[:n_hard] = torch.tensor(pos[:n_hard], dtype=torch.float32)
    if pos.shape[0] > n_hard:
        full[n_hard:n_hard + n_soft] = torch.tensor(
            pos[n_hard:n_hard + n_soft], dtype=torch.float32)
    cost = compute_proxy_cost(full, benchmark, plc)
    return {
        "seed": seed,
        "proxy": cost["proxy_cost"],
        "wl": cost["wirelength_cost"],
        "den": cost["density_cost"],
        "cong": cost["congestion_cost"],
        "ovrlp": cost["overlap_count"],
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--num-seeds", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--init-mode", default="anchor_soft")
    parser.add_argument("--anchor-strategy", default="centroid")
    parser.add_argument("--spawn-radius", type=float, default=0.05)
    parser.add_argument("--spawn-adaptive", default="1")
    parser.add_argument("--anchor-loss", default="0")
    parser.add_argument("--lambda-max", default="auto")
    parser.add_argument("--target-util", default="auto")
    parser.add_argument("--cluster-target", default="auto")
    parser.add_argument("--finish-legalize", default="1")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    print(f"[cpu_par] bench={args.bench} seeds={args.num_seeds} workers={args.workers} "
          f"steps={args.steps}", flush=True)

    env_overrides = {
        "STRAPLE_DEMO_INIT": args.init_mode,
        "STRAPLE_DEMO_ANCHOR_STRATEGY": args.anchor_strategy,
        "STRAPLE_DEMO_SPAWN_RADIUS_FRAC": str(args.spawn_radius),
        "STRAPLE_DEMO_SPAWN_ADAPTIVE": args.spawn_adaptive,
        "STRAPLE_DEMO_ANCHOR_LOSS": args.anchor_loss,
        "STRAPLE_DEMO_PLACE_ALL": "1",
        "STRAPLE_DEMO_FINISH_LEGALIZE": args.finish_legalize,
        "STRAPLE_DEMO_LAMBDA_MAX": args.lambda_max,
        "STRAPLE_DEMO_TARGET_UTIL": args.target_util,
        "STRAPLE_DEMO_CLUSTER_TARGET": args.cluster_target,
    }

    tasks = []
    for k in range(args.num_seeds):
        sd = args.seed_base + k * 1009
        tasks.append((args.bench, sd, args.steps, env_overrides))

    t_total = time.time()
    results = []
    best_proxy = float("inf")
    best_seed = -1
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_worker, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:
                print(f"[cpu_par] seed#{i} FAILED: {exc}", flush=True)
                continue
            valid = r["ovrlp"] == 0
            r["valid"] = valid
            r["seed_idx"] = i
            results.append(r)
            if valid and r["proxy"] < best_proxy:
                best_proxy = r["proxy"]
                best_seed = r["seed"]
            completed += 1
            marker = "★" if valid and r["proxy"] == best_proxy else " "
            print(f"[cpu_par] {completed}/{args.num_seeds} seed#{i:02d} sd={r['seed']:5d} "
                  f"{marker} proxy={r['proxy']:.4f} (wl={r['wl']:.3f} "
                  f"den={r['den']:.3f} cong={r['cong']:.3f}) "
                  f"ovrlp={r['ovrlp']} {r['elapsed']:.1f}s",
                  flush=True)

    proxies_valid = [r["proxy"] for r in results if r["valid"]]
    print(f"\n[cpu_par] DONE in {time.time()-t_total:.1f}s")
    print(f"[cpu_par] {len(proxies_valid)}/{args.num_seeds} valid")
    if proxies_valid:
        print(f"[cpu_par] best={best_proxy:.4f} (seed={best_seed})")
        print(f"[cpu_par] mean={np.mean(proxies_valid):.4f} "
              f"median={np.median(proxies_valid):.4f} "
              f"std={np.std(proxies_valid):.4f}")

    out = {
        "bench": args.bench,
        "num_seeds": args.num_seeds,
        "workers": args.workers,
        "steps": args.steps,
        "best_proxy": best_proxy if proxies_valid else None,
        "best_seed": best_seed,
        "mean_valid": float(np.mean(proxies_valid)) if proxies_valid else None,
        "results": sorted(results, key=lambda r: r["seed_idx"]),
        "wall_time": time.time() - t_total,
    }
    out_path = Path(args.output_json) if args.output_json else (
        REPO_ROOT / "results" / f"cpu_par_{args.bench}_{time.strftime('%Y%m%dT%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[cpu_par] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
