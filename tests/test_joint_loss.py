"""Sanity check for STRAPLE_BATCH_JOINT_LOSS_P (Action #3, M1)."""

import math

import torch


def _joint_pen_norm(cong_total, dpen_total, cong_w, dens_w, p, eps=1e-9):
    cong_norm = cong_total / cong_total.detach().clamp_min(eps)
    dens_norm = dpen_total / dpen_total.detach().clamp_min(eps)
    return (
        (cong_w * cong_norm) ** p + (dens_w * dens_norm) ** p + eps
    ) ** (1.0 / p)


def test_joint_pen_value_matches_weights():
    cong = torch.tensor(1.07, requires_grad=False)
    dens = torch.tensor(0.58, requires_grad=False)
    cw, dw = 10.0, 5.0
    p = 4.0
    val = _joint_pen_norm(cong, dens, cw, dw, p)
    # Numerically: cong_norm = 1.0, dens_norm = 1.0
    # joint = (cw^p + dw^p)^(1/p) = (10000 + 625)^0.25
    expected = (cw ** p + dw ** p) ** (1.0 / p)
    assert abs(float(val) - expected) < 1e-3, (val, expected)


def test_joint_pen_gradient_flows_finite():
    cong = torch.tensor(1.07, requires_grad=True)
    dens = torch.tensor(0.58, requires_grad=True)
    cw, dw = 10.0, 5.0
    for p in (2.0, 4.0):
        val = _joint_pen_norm(cong, dens, cw, dw, p)
        cg, dg = torch.autograd.grad(val, [cong, dens])
        assert torch.isfinite(cg).all() and torch.isfinite(dg).all(), (p, cg, dg)
        # cong gradient should be larger because cw > dw (p=4 amplifies)
        assert float(cg) > float(dg), (p, cg, dg)


def test_joint_pen_handles_zero_components():
    eps = 1e-9
    for cong_v in (0.0, 1e-10):
        for dens_v in (0.0, 1e-10):
            cong = torch.tensor(cong_v, requires_grad=True)
            dens = torch.tensor(dens_v, requires_grad=True)
            val = _joint_pen_norm(cong, dens, 10.0, 5.0, 4.0)
            assert torch.isfinite(val), (cong_v, dens_v, val)


def test_per_k_joint_pen_per_seed():
    K, n_macros = 4, 8
    cong_K = torch.linspace(0.5, 1.5, K, requires_grad=True)
    dpen_K = torch.linspace(2.0, 1.0, K, requires_grad=True)
    cong_mul = torch.ones(K)
    dens_mul = torch.ones(K)
    cong_weight, density_weight, p = 10.0, 5.0, 4.0
    eps = 1e-9
    cong_norm = cong_K / cong_K.detach().clamp_min(eps)
    dens_norm = dpen_K / dpen_K.detach().clamp_min(eps)
    cw_per_K = cong_weight * cong_mul
    dw_per_K = density_weight * dens_mul
    joint_pen_K = (
        (cw_per_K * cong_norm) ** p
        + (dw_per_K * dens_norm) ** p
        + eps
    ) ** (1.0 / p)
    total = joint_pen_K.sum()
    cg, dg = torch.autograd.grad(total, [cong_K, dpen_K])
    assert torch.isfinite(cg).all() and torch.isfinite(dg).all()
    # Each per-seed joint_pen ~ (cw^p + dw^p)^(1/p) = constant numerically
    expected_per_seed = (cong_weight ** p + density_weight ** p) ** (1.0 / p)
    assert abs(float(total) - K * expected_per_seed) < 1e-2


if __name__ == "__main__":
    test_joint_pen_value_matches_weights()
    test_joint_pen_gradient_flows_finite()
    test_joint_pen_handles_zero_components()
    test_per_k_joint_pen_per_seed()
    print("OK")
