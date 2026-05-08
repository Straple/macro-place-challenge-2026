"""Perturb-relax-CD cycles: escape CD floor via mini-gradient with spreading force.

Pipeline per cycle:
  1. Perturb fraction of macros by ±step×cell (introduces overlaps).
  2. Mini-gradient (K small, few steps) — overlap penalty + WL spreading
     resolves overlaps in physically meaningful direction (unlike random jitter).
  3. Legalize each of K via C++ legalizer.
  4. Pick best valid by proxy_cost.
  5. CD polish (approx mode) on best.
  6. Compare to previous best, accept if improved.

Goal: escape single-macro CD basin floor by combining gradient's spreading
force (cluster-level rearrangement) with CD's local refinement.
"""

from __future__ import annotations

import time

import numpy as np


def perturb_relax_cycles(benchmark, plc, pos_full,
                          gradient_batch_fn,
                          cd_polish_fn,
                          legalize_fn,
                          compute_proxy_fn,
                          n_cycles: int = 5,
                          mini_K: int = 8,
                          mini_steps: int = 100,
                          mini_time_budget: float = 20.0,
                          perturb_frac: float = 0.25,
                          perturb_step: float = 0.5,
                          seed: int = 777,
                          wall_deadline: float = 0.0,
                          gradient_kwargs: "dict | None" = None,
                          cd_kwargs: "dict | None" = None,
                          verbose: bool = True):
    """Run perturb-relax-CD cycles, return (best_pos, best_proxy).

    gradient_batch_fn: callable(benchmark, plc, K, num_steps, time_budget,
                                init_pos_override, **gradient_kwargs) -> (pos_K, stats)
    cd_polish_fn: callable(benchmark, plc, pos_full, **cd_kwargs) -> (pos, proxy)
    legalize_fn: callable(k, pos_hard, sizes_np, movable_np, canvas_w, canvas_h, seed)
                 -> (k, leg_hard)
    compute_proxy_fn: callable(pos_full_tensor, benchmark, plc) -> dict with
                     'proxy_cost', 'overlap_count'
    wall_deadline: absolute time (time.time() seconds) after which to stop.
                  0 = no deadline.
    """
    import torch
    from macro_place.objective import compute_proxy_cost

    if gradient_kwargs is None:
        gradient_kwargs = {}
    if cd_kwargs is None:
        cd_kwargs = {}

    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)
    grid_cols = int(benchmark.grid_cols)
    grid_rows = int(benchmark.grid_rows)
    cell_w = canvas_w / grid_cols
    cell_h = canvas_h / grid_rows

    sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
    fixed = benchmark.macro_fixed.cpu().numpy().astype(bool)
    half_w = sizes[:, 0] * 0.5
    half_h = sizes[:, 1] * 0.5
    n_hard = int(benchmark.num_hard_macros)
    n_soft = int(benchmark.num_soft_macros)
    movable_arr_hard = ~fixed[:n_hard]

    movable_hard_idx = np.where(movable_arr_hard)[0]
    n_perturb = max(1, int(perturb_frac * len(movable_hard_idx)))

    rng = np.random.default_rng(seed)
    best_pos = pos_full.astype(np.float64).copy()
    full_t = torch.tensor(best_pos, dtype=torch.float32)
    cost = compute_proxy_cost(full_t, benchmark, plc)
    best_proxy = float(cost["proxy_cost"])
    if verbose:
        print(f"[PR] start proxy={best_proxy:.4f} ovrlp={int(cost['overlap_count'])} "
              f"(cycles={n_cycles} mini_K={mini_K} mini_steps={mini_steps} "
              f"perturb_frac={perturb_frac} perturb_step={perturb_step})",
              flush=True)

    for cycle in range(n_cycles):
        if wall_deadline > 0 and time.time() >= wall_deadline:
            if verbose:
                print(f"[PR] cycle {cycle+1}: wall deadline reached, stop",
                      flush=True)
            break

        t_cycle = time.time()
        pos_perturbed = best_pos.copy()
        chosen = rng.choice(movable_hard_idx,
                             size=min(n_perturb, len(movable_hard_idx)),
                             replace=False)
        for i in chosen:
            dx = rng.uniform(-cell_w * perturb_step, cell_w * perturb_step)
            dy = rng.uniform(-cell_h * perturb_step, cell_h * perturb_step)
            pos_perturbed[i, 0] = float(np.clip(
                pos_perturbed[i, 0] + dx,
                half_w[i], canvas_w - half_w[i]))
            pos_perturbed[i, 1] = float(np.clip(
                pos_perturbed[i, 1] + dy,
                half_h[i], canvas_h - half_h[i]))

        if verbose:
            print(f"[PR] cycle {cycle+1}/{n_cycles}: perturbed {len(chosen)} "
                  f"macros (±{perturb_step:.2f} cell)", flush=True)

        t_grad = time.time()
        pos_K_relaxed, _grad_stats = gradient_batch_fn(
            benchmark, plc, K=mini_K, num_steps=mini_steps,
            time_budget=mini_time_budget,
            seed=seed + cycle * 13 + 1,
            init_pos_override=pos_perturbed,
            verbose=False,
            **gradient_kwargs,
        )
        grad_dt = time.time() - t_grad
        if verbose:
            print(f"[PR]   mini-gradient K={mini_K} {grad_dt:.1f}s",
                  flush=True)

        best_lk_proxy = float("inf")
        best_lk_pos = None
        for k in range(mini_K):
            pos_hard_k = pos_K_relaxed[k, :n_hard].astype(np.float64)
            _, leg_hard = legalize_fn(
                k, pos_hard_k, sizes[:n_hard],
                movable_arr_hard, canvas_w, canvas_h, seed + cycle + k)
            full_k = best_pos.copy()
            full_k[:n_hard] = leg_hard
            if pos_K_relaxed.shape[1] > n_hard:
                full_k[n_hard:n_hard + n_soft] = pos_K_relaxed[
                    k, n_hard:n_hard + n_soft].astype(np.float64)
            full_kt = torch.tensor(full_k, dtype=torch.float32)
            cost_k = compute_proxy_cost(full_kt, benchmark, plc)
            ov_k = int(cost_k["overlap_count"])
            pr_k = float(cost_k["proxy_cost"])
            if ov_k == 0 and pr_k < best_lk_proxy:
                best_lk_proxy = pr_k
                best_lk_pos = full_k

        if best_lk_pos is None:
            if verbose:
                print(f"[PR]   cycle {cycle+1}: all K had overlap after legalize, "
                      f"skip", flush=True)
            continue

        if verbose:
            print(f"[PR]   pre-CD best of K={mini_K}: {best_lk_proxy:.4f}",
                  flush=True)

        t_cd = time.time()
        polished_pos, polished_proxy = cd_polish_fn(
            benchmark, plc, best_lk_pos, **cd_kwargs)
        cd_dt = time.time() - t_cd

        cycle_dt = time.time() - t_cycle
        if polished_proxy < best_proxy - 1e-6:
            if verbose:
                print(f"[PR]   cycle {cycle+1} IMPROVED: "
                      f"{polished_proxy:.4f} < {best_proxy:.4f} "
                      f"(grad {grad_dt:.1f}s + cd {cd_dt:.1f}s = "
                      f"{cycle_dt:.1f}s)", flush=True)
            best_pos = polished_pos
            best_proxy = polished_proxy
        else:
            if verbose:
                print(f"[PR]   cycle {cycle+1} no improvement: "
                      f"{polished_proxy:.4f} >= {best_proxy:.4f} "
                      f"({cycle_dt:.1f}s)", flush=True)

    return best_pos, best_proxy
