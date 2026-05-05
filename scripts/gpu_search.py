"""GPU-accelerated multi-seed gradient search for macro placement.

Запускает gradient_demo с N разных seeds на GPU, измеряет proxy_cost для каждого,
выбирает best. Опционально дополняет ALNS polish для best seed.

Usage:
    uv run python scripts/gpu_search.py --bench ibm01 --num-seeds 32
    uv run python scripts/gpu_search.py --bench ibm01 --num-seeds 64 --steps 600
    uv run python scripts/gpu_search.py --bench ibm04 --num-seeds 32 --polish-alns

Output:
    - results/gpu_search_<bench>_<timestamp>.json — все proxies
    - vis/gpu_search_<bench>_best.html — visualizer best seed (если задан --vis)
    - stdout: best/mean/std proxy + winner seed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "submissions" / "straple"))

from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost


def load_bench(name: str):
    iccad04_root = REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04"
    if (iccad04_root / name).exists():
        return load_benchmark_from_dir(str(iccad04_root / name))
    raise ValueError(f"unknown bench: {name}")


def run_single_seed(benchmark, plc, seed: int, steps: int, device: str,
                    init_mode: str, anchor_strategy: str, spawn_radius: float,
                    spawn_adaptive: bool, anchor_loss: bool, place_all: bool,
                    finish_legalize: bool, lambda_max: str, target_util: str,
                    cluster_target: str) -> tuple:
    from gradient_demo import gradient_demo

    snap = {}
    env_to_set = {
        "STRAPLE_DEMO_DEVICE": device,
        "STRAPLE_DEMO_INIT": init_mode,
        "STRAPLE_DEMO_PLACE_ALL": "1" if place_all else "0",
        "STRAPLE_DEMO_FINISH_LEGALIZE": "1" if finish_legalize else "0",
        "STRAPLE_DEMO_ANCHOR_STRATEGY": anchor_strategy,
        "STRAPLE_DEMO_SPAWN_RADIUS_FRAC": str(spawn_radius),
        "STRAPLE_DEMO_SPAWN_ADAPTIVE": "1" if spawn_adaptive else "0",
        "STRAPLE_DEMO_ANCHOR_LOSS": "1" if anchor_loss else "0",
        "STRAPLE_DEMO_LAMBDA_MAX": lambda_max,
        "STRAPLE_DEMO_TARGET_UTIL": target_util,
        "STRAPLE_DEMO_CLUSTER_TARGET": cluster_target,
    }
    for k, v in env_to_set.items():
        snap[k] = os.environ.get(k)
        os.environ[k] = v

    try:
        t0 = time.time()
        result = gradient_demo(benchmark, plc, recorder=None,
                               num_steps=steps, seed=seed,
                               time_budget=0.0, score_png="")
        elapsed = time.time() - t0
        pos = result[0] if isinstance(result, tuple) else result
        pos = np.asarray(pos, dtype=np.float64)
        n_hard = benchmark.num_hard_macros
        n_total = benchmark.num_macros
        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(pos[:n_hard], dtype=torch.float32)
        if pos.shape[0] > n_hard:
            n_soft = benchmark.num_soft_macros
            full[n_hard:n_hard + n_soft] = torch.tensor(
                pos[n_hard:n_hard + n_soft], dtype=torch.float32)
        cost = compute_proxy_cost(full, benchmark, plc)
        return (pos, cost, elapsed)
    finally:
        for k, prev in snap.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", required=True)
    parser.add_argument("--num-seeds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-mode", default="anchor_soft")
    parser.add_argument("--anchor-strategy", default="centroid",
                        choices=["grid", "centroid"])
    parser.add_argument("--spawn-radius", type=float, default=0.05)
    parser.add_argument("--spawn-adaptive", action="store_true", default=True)
    parser.add_argument("--anchor-loss", action="store_true", default=False)
    parser.add_argument("--place-all", action="store_true", default=True)
    parser.add_argument("--finish-legalize", action="store_true", default=True)
    parser.add_argument("--lambda-max", default="auto")
    parser.add_argument("--target-util", default="auto")
    parser.add_argument("--cluster-target", default="auto")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--polish-alns", action="store_true",
                        help="run ALNS on best gradient seed")
    parser.add_argument("--vis", action="store_true",
                        help="save HTML visualization for best seed")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    print(f"[gpu_search] bench={args.bench} seeds={args.num_seeds} steps={args.steps} "
          f"device={args.device}", flush=True)
    t_total = time.time()
    benchmark, plc = load_bench(args.bench)
    print(f"[gpu_search] loaded: hard={benchmark.num_hard_macros} "
          f"soft={benchmark.num_soft_macros} nets={benchmark.num_nets}", flush=True)

    if args.device == "cuda" and torch.cuda.is_available():
        print(f"[gpu_search] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print(f"[gpu_search] CPU mode", flush=True)

    # initial baseline
    c0 = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
    print(f"[gpu_search] INITIAL proxy={c0['proxy_cost']:.4f} "
          f"ovrlp={c0['overlap_count']}", flush=True)

    results = []
    best_proxy = float("inf")
    best_pos = None
    best_seed = -1
    best_idx = -1

    for k in range(args.num_seeds):
        sd = args.seed_base + k * 1009
        pos, cost, el = run_single_seed(
            benchmark, plc, seed=sd, steps=args.steps, device=args.device,
            init_mode=args.init_mode, anchor_strategy=args.anchor_strategy,
            spawn_radius=args.spawn_radius, spawn_adaptive=args.spawn_adaptive,
            anchor_loss=args.anchor_loss, place_all=args.place_all,
            finish_legalize=args.finish_legalize, lambda_max=args.lambda_max,
            target_util=args.target_util, cluster_target=args.cluster_target,
        )
        valid = cost["overlap_count"] == 0
        proxy = cost["proxy_cost"]
        marker = "★" if valid and proxy < best_proxy else " "
        print(f"[gpu_search] seed#{k:02d} sd={sd:5d} {marker} "
              f"proxy={proxy:.4f} (wl={cost['wirelength_cost']:.3f} "
              f"den={cost['density_cost']:.3f} cong={cost['congestion_cost']:.3f}) "
              f"ovrlp={cost['overlap_count']} {el:.1f}s", flush=True)
        results.append({
            "seed_idx": k,
            "seed": sd,
            "proxy": proxy,
            "wl": cost["wirelength_cost"],
            "den": cost["density_cost"],
            "cong": cost["congestion_cost"],
            "ovrlp": cost["overlap_count"],
            "valid": valid,
            "elapsed": el,
        })
        if valid and proxy < best_proxy:
            best_proxy = proxy
            best_pos = pos
            best_seed = sd
            best_idx = k

    proxies_valid = [r["proxy"] for r in results if r["valid"]]
    print(f"\n[gpu_search] DONE in {time.time()-t_total:.1f}s")
    print(f"[gpu_search] {len(proxies_valid)}/{args.num_seeds} valid")
    if proxies_valid:
        print(f"[gpu_search] best={best_proxy:.4f} (seed={best_seed} idx={best_idx})")
        print(f"[gpu_search] mean={np.mean(proxies_valid):.4f} "
              f"median={np.median(proxies_valid):.4f} "
              f"std={np.std(proxies_valid):.4f} worst={max(proxies_valid):.4f}")

    if args.polish_alns and best_pos is not None:
        print(f"\n[gpu_search] === ALNS polish on best gradient seed ===", flush=True)
        # Use placer.py with gradient seed pre-baked
        from placer import StraplePlacer
        # Can't directly inject best_pos — easier to env-flag use_gradient_seed
        # but we want it to use OUR best_pos. Hack: write to file, set env.
        # For simplicity here: just run placer with USE_GRADIENT_SEED=1 (it'll
        # rerun gradient internally with seed_base+0). To avoid that, set env
        # flag to use OUR pos. We use a temp file.
        import pickle
        tmp = REPO_ROOT / "results" / f"gpu_seed_{args.bench}.pkl"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            pickle.dump({"hard": best_pos[:benchmark.num_hard_macros],
                         "soft": best_pos[benchmark.num_hard_macros:]}, f)
        os.environ["STRAPLE_GRADIENT_SEED_FILE"] = str(tmp)
        os.environ["STRAPLE_USE_GRADIENT_SEED"] = "1"
        placer = StraplePlacer(seed=42)
        full = placer.place(benchmark)
        post = compute_proxy_cost(full, benchmark, plc)
        print(f"[gpu_search] AFTER ALNS polish: proxy={post['proxy_cost']:.4f} "
              f"ovrlp={post['overlap_count']}", flush=True)

    out = {
        "bench": args.bench,
        "num_seeds": args.num_seeds,
        "steps": args.steps,
        "initial_proxy": c0["proxy_cost"],
        "initial_ovrlp": c0["overlap_count"],
        "best_proxy": best_proxy if proxies_valid else None,
        "best_seed": best_seed,
        "mean_valid": float(np.mean(proxies_valid)) if proxies_valid else None,
        "results": results,
        "wall_time": time.time() - t_total,
    }
    if args.output_json:
        out_path = Path(args.output_json)
    else:
        out_path = REPO_ROOT / "results" / (
            f"gpu_search_{args.bench}_{time.strftime('%Y%m%dT%H%M%S')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gpu_search] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
