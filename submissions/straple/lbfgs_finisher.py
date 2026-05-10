"""Batched L-BFGS finisher for late-stage gradient phase.

After Adam converges to a settled landscape (typically late P3), switch
to L-BFGS quasi-Newton steps for super-linear convergence on the locally
quadratic loss. K seeds run independently, each with its own m-history
of (s, y) pairs.

Standard Liu-Nocedal two-loop recursion (1989) with:
- Powell damping when s·y is small (avoids non-PD update)
- H_0 scaling = (s·y)/(y·y) for the most recent pair
- Fixed step size + magnitude clip (no Wolfe LS for v1 simplicity)
- History drop on phase boundaries (caller invokes reset())

Memory: m·K·n_active·2·4 bytes (e.g. m=10 K=384 n=1140 → 35 MB).

Activated via STRAPLE_BATCH_LBFGS_FROM_STEP=N (default 0 = always Adam).
"""

from __future__ import annotations

from typing import Optional

import torch


class BatchedLBFGS:
    def __init__(self, K: int, n_active: int, m: int = 10,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32):
        self.K = K
        self.n_active = n_active
        self.m = m
        self.dev = device
        self.dtype = dtype
        self.s = torch.zeros(m, K, n_active, 2, device=device, dtype=dtype)
        self.y = torch.zeros(m, K, n_active, 2, device=device, dtype=dtype)
        self.rho = torch.zeros(m, K, device=device, dtype=dtype)
        self.head = 0
        self.size = 0

    def reset(self) -> None:
        self.head = 0
        self.size = 0
        self.s.zero_()
        self.y.zero_()
        self.rho.zero_()

    def two_loop(self, grad: torch.Tensor) -> torch.Tensor:
        """Compute L-BFGS search direction p = -H · grad.

        grad shape: [K, n_active, 2].
        """
        q = grad.clone()
        alphas = []
        for i in range(self.size):
            idx = (self.head - 1 - i) % self.m
            sy_dot = (self.s[idx] * q).sum(dim=(1, 2))     # [K]
            a = self.rho[idx] * sy_dot                       # [K]
            q = q - a[:, None, None] * self.y[idx]
            alphas.append((idx, a))

        if self.size > 0:
            last = (self.head - 1) % self.m
            sy = (self.s[last] * self.y[last]).sum(dim=(1, 2))   # [K]
            yy = (self.y[last] * self.y[last]).sum(dim=(1, 2))   # [K]
            gamma = (sy / yy.clamp_min(1e-12)).clamp(min=1e-6, max=1e3)
            r = gamma[:, None, None] * q
        else:
            r = q

        for idx, a in reversed(alphas):
            yr_dot = (self.y[idx] * r).sum(dim=(1, 2))           # [K]
            b = self.rho[idx] * yr_dot
            r = r + (a - b)[:, None, None] * self.s[idx]

        return -r

    def push(self, s_new: torch.Tensor, y_new: torch.Tensor,
             powell_eps: float = 1e-10) -> torch.Tensor:
        """Add (s_new, y_new) pair with Powell damping if curvature too small.

        Returns: bool tensor [K] — True for seeds where new pair was accepted
        (sy > eps); pair stored damped or skipped per Powell.
        """
        sy = (s_new * y_new).sum(dim=(1, 2))                      # [K]
        ss = (s_new * s_new).sum(dim=(1, 2))
        yy = (y_new * y_new).sum(dim=(1, 2))
        # Powell damping: if sy < 0.2 * y_norm² scale it down
        powell_thr = 0.2 * yy.clamp_min(1e-12)
        bad = sy < powell_thr
        if bad.any():
            theta = torch.where(
                sy < powell_thr,
                (0.8 * yy) / (yy - sy).clamp_min(1e-12),
                torch.ones_like(sy),
            ).clamp(0.0, 1.0)
            y_damped = (theta[:, None, None] * y_new
                         + (1.0 - theta[:, None, None]) * s_new)
            sy = (s_new * y_damped).sum(dim=(1, 2))
            yy = (y_damped * y_damped).sum(dim=(1, 2))
            y_to_store = y_damped
        else:
            y_to_store = y_new

        valid = (sy > powell_eps) & (ss > powell_eps) & (yy > powell_eps)

        self.s[self.head] = s_new
        self.y[self.head] = y_to_store
        self.rho[self.head] = torch.where(
            valid, 1.0 / sy.clamp_min(1e-12),
            torch.zeros_like(sy),
        )
        self.head = (self.head + 1) % self.m
        self.size = min(self.size + 1, self.m)
        return valid

    def step(self, pos: torch.Tensor, grad: torch.Tensor,
             prev_pos: Optional[torch.Tensor] = None,
             prev_grad: Optional[torch.Tensor] = None,
             alpha: float = 1.0,
             max_step_norm: float = 0.5) -> torch.Tensor:
        """Compute new pos = pos + alpha * direction (clipped). Updates buffers.

        prev_pos / prev_grad: from previous L-BFGS step. If None, just store
        current as previous and return small Adam-like step (-alpha·grad).
        """
        if prev_pos is not None and prev_grad is not None:
            s_new = pos - prev_pos
            y_new = grad - prev_grad
            self.push(s_new, y_new)

        direction = self.two_loop(grad)

        # Per-seed step magnitude clip — protects against pathological
        # curvature explosions early.
        dir_norm = direction.norm(dim=(1, 2)).clamp_min(1e-12)
        clip_factor = (max_step_norm / dir_norm).clamp(max=1.0)
        direction = direction * clip_factor[:, None, None]

        return pos + alpha * direction
