"""Action #5: Optuna TPE HPO over Round 23 best pipeline.

Runs sequential trials of gpu_run_one.py on ibm01 with sampled hyperparams.
Each trial = single run (~28 min). Persistent SQLite study at
.remote_runs/hpo_study.db so worker is restartable.

Usage:
  uv run python scripts/hpo_optuna.py --n-trials 25 --timeout-h 12

After run completes, top-N configs are listed; user re-tests them with
N=5 paired runs to verify.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import optuna

REPO = Path(__file__).resolve().parent.parent


TRIAL9_FIXED_ENV = {
    "STRAPLE_BATCH_EPLACE": "1",
    "STRAPLE_BATCH_EPLACE_GRID": "128",
    "STRAPLE_BATCH_COHESION_START": "5",
    "STRAPLE_BATCH_COHESION_END": "0.001",
    "STRAPLE_BATCH_DIVERSITY": "1",
    "STRAPLE_BATCH_OVERLAP_FORM": "rect_quad",
    "STRAPLE_BATCH_OVERFLOW_LAMBDA": "1",
    "STRAPLE_BATCH_OVERFLOW_COEF_HI": "1.5",
    "STRAPLE_BATCH_BLOCKAGE_W": "50",
    "STRAPLE_BATCH_CD_POLISH": "1",
    "STRAPLE_BATCH_CD_GPU_FILTER": "1",
    "STRAPLE_BATCH_CD_GPU_APPROX": "1",
    "STRAPLE_BATCH_CD_DIRS": "8",
    "STRAPLE_BATCH_CD_ROUNDS": "8",
    "STRAPLE_BATCH_PAIR_SWAP": "1",
    "STRAPLE_BATCH_PAIR_SWAP_NEIGHBORS": "12",
    "STRAPLE_BATCH_TRIPLE_CYCLE": "1",
    "STRAPLE_BATCH_TRIPLE_CYCLE_NEIGHBORS": "6",
    "STRAPLE_BATCH_TRIPLE_CYCLE_ROUNDS": "4",
    "STRAPLE_BATCH_WALL_TL": "1700",
    "STRAPLE_BATCH_WALL_RESERVE": "30",
    "STRAPLE_BATCH_COHESION_START": "5",
    "STRAPLE_BATCH_COHESION_END": "0.001",
    "STRAPLE_BATCH_BREAKDOWN_LOG": "1",
}


def _parse_proxy_from_line(line: str) -> Optional[float]:
    for tok in line.split():
        if tok.startswith("proxy="):
            v = tok.split("=", 1)[1].rstrip("] ,;")
            try:
                return float(v)
            except ValueError:
                return None
    return None


def _parse_pipeline_proxy(line: str) -> Optional[float]:
    """Extract proxy from end-of-stage 'IMPROVED'/no-improvement lines."""
    keys = ("CD polish IMPROVED:", "CD polish:",
            "CLUSTER polish IMPROVED:", "CLUSTER polish:",
            "PAIR_SWAP IMPROVED:", "PAIR_SWAP:",
            "PAIR_SWAP 2ND IMPROVED:", "PAIR_SWAP 2ND:",
            "CD POSTSWAP IMPROVED:", "CD POSTSWAP:",
            "TRIPLE_CYCLE IMPROVED:", "TRIPLE_CYCLE:")
    if any(k in line for k in keys):
        for tok in line.replace("<", " ").split():
            try:
                v = float(tok)
                if 0.5 < v < 5.0:
                    return v
            except ValueError:
                continue
    return None


def run_trial(trial_number: int, env_overrides: dict, time_budget: int,
              wall_tl: int, k_trial: int) -> dict:
    """Launch gpu_run_one.py once with given env, return parsed stats."""
    env = os.environ.copy()
    env.update(TRIAL9_FIXED_ENV)
    env.update(env_overrides)
    env["STRAPLE_BATCH_RUN_SEED_BASE"] = "42"
    env["STRAPLE_BATCH_WALL_TL"] = str(wall_tl)

    log_path = REPO / ".remote_runs" / f"hpo_trial_{trial_number:04d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stats_dst = REPO / ".remote_runs" / f"stats_hpo_trial_{trial_number:04d}.json"

    cmd = [
        "uv", "run", "python", str(REPO / "scripts/gpu_run_one.py"),
        "--bench", "ibm01",
        "--K", str(k_trial),
        "--time-budget", str(time_budget),
        "--no-vis",
    ]

    t0 = time.time()
    with open(log_path, "wb") as logf:
        logf.write(f"START {time.strftime('%Y-%m-%d %H:%M:%S')}\n".encode())
        logf.write(f"env_overrides={json.dumps(env_overrides)}\n".encode())
        logf.flush()
        ret = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                             cwd=REPO)
    dt = time.time() - t0

    stats_src = REPO / "results" / "gpu_stats_ibm01.json"
    if stats_src.exists():
        try:
            stats_dst.write_bytes(stats_src.read_bytes())
        except Exception:
            pass

    final_proxy = float("inf")
    pre_cd_min = float("inf")
    if log_path.exists():
        with open(log_path, "r", errors="replace") as f:
            lines = f.readlines()
        # Pass 1: prefer post-triple-cycle BREAKDOWN line (authoritative final).
        for line in lines:
            if "BREAKDOWN" in line and "stage=post-triple-cycle" in line:
                v = _parse_proxy_from_line(line)
                if v is not None:
                    final_proxy = v
            if "BREAKDOWN" in line and "stage=post-legalize" in line:
                v = _parse_proxy_from_line(line)
                if v is not None:
                    pre_cd_min = v
        # Pass 2 fallback: latest pipeline-stage line (CD/PAIR_SWAP/TRIPLE).
        if final_proxy == float("inf"):
            latest = None
            for line in lines:
                v = _parse_pipeline_proxy(line)
                if v is not None:
                    latest = v
            if latest is not None:
                final_proxy = latest

    return {
        "ret": ret.returncode,
        "dt": dt,
        "final_proxy": final_proxy,
        "pre_cd_min": pre_cd_min,
        "log": str(log_path),
    }


def objective(trial: optuna.Trial, time_budget: int, wall_tl: int) -> float:
    overlap_w_max = trial.suggest_float("OVERLAP_W_MAX", 20000, 100000, log=True)
    overlap_w_growth = trial.suggest_float("OVERLAP_W_GROWTH", 1.002, 1.010, log=True)
    overflow_target = trial.suggest_float("OVERFLOW_TARGET", 0.08, 0.20)
    overflow_exp = trial.suggest_float("OVERFLOW_EXP", 0.5, 1.0)
    cong_w = trial.suggest_float("CONG_W", 5.0, 25.0)
    pair_swap_rounds = trial.suggest_categorical("PAIR_SWAP_ROUNDS", [6, 8, 10, 12])
    k_trial = trial.suggest_categorical("K", [256, 384])

    env_overrides = {
        "STRAPLE_BATCH_OVERLAP_W_MAX": f"{overlap_w_max:.0f}",
        "STRAPLE_BATCH_OVERLAP_W_GROWTH": f"{overlap_w_growth:.6f}",
        "STRAPLE_BATCH_OVERFLOW_TARGET": f"{overflow_target:.4f}",
        "STRAPLE_BATCH_OVERFLOW_EXP": f"{overflow_exp:.4f}",
        "STRAPLE_BATCH_CONG_W": f"{cong_w:.3f}",
        "STRAPLE_BATCH_PAIR_SWAP_ROUNDS": str(pair_swap_rounds),
    }
    print(f"[hpo] trial {trial.number}: {env_overrides} K={k_trial}", flush=True)
    result = run_trial(trial.number, env_overrides, time_budget, wall_tl, k_trial)
    print(f"[hpo] trial {trial.number} -> proxy={result['final_proxy']:.4f} "
          f"pre_cd={result['pre_cd_min']:.4f} ret={result['ret']} "
          f"dt={result['dt']:.1f}s", flush=True)

    trial.set_user_attr("pre_cd_min", result["pre_cd_min"])
    trial.set_user_attr("dt", result["dt"])
    trial.set_user_attr("ret", result["ret"])
    trial.set_user_attr("k_trial", k_trial)

    if result["final_proxy"] == float("inf") or result["ret"] != 0:
        return 999.0
    return result["final_proxy"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--timeout-h", type=float, default=12.0)
    parser.add_argument("--storage", type=str,
                        default=str(REPO / ".remote_runs" / "hpo_study.db"))
    parser.add_argument("--study-name", type=str, default="ibm01_round23_hpo")
    parser.add_argument("--time-budget", type=int, default=1200)
    parser.add_argument("--wall-tl", type=int, default=1700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    storage_path = Path(args.storage)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_path}"

    sampler = optuna.samplers.TPESampler(seed=args.seed,
                                          n_startup_trials=5,
                                          multivariate=True)
    study = optuna.create_study(study_name=args.study_name,
                                  storage=storage_url,
                                  sampler=sampler,
                                  load_if_exists=True,
                                  direction="minimize")
    print(f"[hpo] study={args.study_name} storage={storage_url} "
          f"existing_trials={len(study.trials)}", flush=True)
    print(f"[hpo] target: n_trials={args.n_trials} timeout_h={args.timeout_h}",
          flush=True)

    timeout_s = int(args.timeout_h * 3600)
    study.optimize(
        lambda t: objective(t, args.time_budget, args.wall_tl),
        n_trials=args.n_trials,
        timeout=timeout_s,
        show_progress_bar=False,
    )

    print(f"\n[hpo] DONE. best_value={study.best_value:.4f}", flush=True)
    print(f"[hpo] best_params={study.best_params}", flush=True)
    print(f"\n[hpo] top-5 trials by value:", flush=True)
    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None and t.value < 998],
        key=lambda t: t.value,
    )
    for tr in sorted_trials[:5]:
        print(f"  #{tr.number:3d} value={tr.value:.4f} "
              f"pre_cd={tr.user_attrs.get('pre_cd_min', 'NA')} "
              f"k={tr.user_attrs.get('k_trial', 'NA')} "
              f"params={tr.params}", flush=True)


if __name__ == "__main__":
    main()
