"""Component breakdown logging for diagnostics (Action #1, H2).

Logs WL/density/congestion components at 5 pipeline stages
(post-gradient, post-legalize, post-CD, post-pair-swap, post-triple-cycle)
to identify which component dominates and where it stops moving.

Gated by env var STRAPLE_BATCH_BREAKDOWN_LOG=1 (default off).
TILOS overhead ~2s/call * 5 stages = ~10s total per run (negligible).
"""

import os
import time
from typing import Optional

import numpy as np
import torch


def breakdown_enabled() -> bool:
    return os.environ.get("STRAPLE_BATCH_BREAKDOWN_LOG", "0") == "1"


def breakdown_log(stage: str,
                  pos_full,
                  benchmark,
                  plc,
                  seed: Optional[int] = None,
                  run_id: str = "") -> Optional[dict]:
    """Compute and log proxy components at a pipeline stage.

    Args:
        stage: short stage name e.g. "post-gradient", "post-legalize",
            "post-cd", "post-pair-swap", "post-triple-cycle".
        pos_full: [n_total, 2] tensor or numpy array with positions
            (hard + soft, full template).
        benchmark: Benchmark object.
        plc: PlacementCost object (will be mutated by _set_placement).
        seed: optional seed index for context.
        run_id: optional run identifier (for log parsing).

    Returns:
        dict with proxy_cost, wirelength_cost, density_cost, congestion_cost,
        overlap_count, fractions; or None if disabled or failed.
    """
    if not breakdown_enabled():
        return None
    try:
        from macro_place.objective import compute_proxy_cost
    except ImportError:
        print("[BREAKDOWN ERR] compute_proxy_cost import failed", flush=True)
        return None

    if isinstance(pos_full, np.ndarray):
        pos_t = torch.tensor(pos_full, dtype=torch.float32)
    elif isinstance(pos_full, torch.Tensor):
        pos_t = pos_full.detach().cpu().to(torch.float32)
    else:
        print(f"[BREAKDOWN ERR] stage={stage} unknown pos_full type "
              f"{type(pos_full)}", flush=True)
        return None

    t_call = time.time()
    try:
        cost = compute_proxy_cost(pos_t, benchmark, plc)
    except Exception as exc:
        print(f"[BREAKDOWN ERR] stage={stage} compute_proxy_cost failed: {exc}",
              flush=True)
        return None
    elapsed = time.time() - t_call

    proxy = float(cost["proxy_cost"])
    wl = float(cost["wirelength_cost"])
    dens = float(cost["density_cost"])
    cong = float(cost["congestion_cost"])
    ovl = int(cost.get("overlap_count", -1))

    weighted_wl = wl
    weighted_dens = 0.5 * dens
    weighted_cong = 0.5 * cong
    if proxy > 1e-12:
        wl_frac = weighted_wl / proxy
        dens_frac = weighted_dens / proxy
        cong_frac = weighted_cong / proxy
    else:
        wl_frac = dens_frac = cong_frac = 0.0

    seed_str = f"seed={seed}" if seed is not None else "seed=best"
    run_str = f"run={run_id} " if run_id else ""
    print(
        f"[BREAKDOWN {run_str}stage={stage} {seed_str} "
        f"wl={wl:.4f} dens={dens:.4f} cong={cong:.4f} "
        f"proxy={proxy:.4f} ovl_n={ovl} "
        f"wl_frac={wl_frac:.3f} dens_frac={dens_frac:.3f} "
        f"cong_frac={cong_frac:.3f} t={elapsed:.2f}s]",
        flush=True,
    )
    return {
        "stage": stage,
        "seed": seed,
        "proxy": proxy,
        "wl": wl,
        "dens": dens,
        "cong": cong,
        "overlap_count": ovl,
        "wl_frac": wl_frac,
        "dens_frac": dens_frac,
        "cong_frac": cong_frac,
        "elapsed": elapsed,
    }
