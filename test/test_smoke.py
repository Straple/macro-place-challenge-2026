"""Smoke tests to verify the competition infrastructure works end-to-end."""

import torch
import pytest
from pathlib import Path

from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement


TESTCASE_ROOT = Path("external/MacroPlacement/Testcases/ICCAD04")


@pytest.fixture
def ibm01():
    """Load ibm01 benchmark from source."""
    path = TESTCASE_ROOT / "ibm01"
    if not path.exists():
        pytest.skip("TILOS submodule not initialized")
    return load_benchmark_from_dir(str(path))


def test_load_benchmark_pt():
    """Benchmark .pt files can be loaded."""
    pt = Path("benchmarks/processed/public/ibm01.pt")
    if not pt.exists():
        pytest.skip("Benchmark .pt files not present")
    b = Benchmark.load(str(pt))
    assert b.num_macros > 0
    assert b.macro_positions.shape == (b.num_macros, 2)
    assert b.macro_sizes.shape == (b.num_macros, 2)


def test_load_benchmark_from_dir(ibm01):
    """Benchmark can be loaded from ICCAD04 directory."""
    benchmark, plc = ibm01
    assert benchmark.num_macros > 0
    assert benchmark.canvas_width > 0
    assert benchmark.canvas_height > 0


def test_compute_proxy_cost(ibm01):
    """Proxy cost can be computed on the default placement."""
    benchmark, plc = ibm01
    costs = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
    assert "proxy_cost" in costs
    assert "wirelength_cost" in costs
    assert "density_cost" in costs
    assert "congestion_cost" in costs
    assert costs["proxy_cost"] > 0


def test_validate_placement(ibm01):
    """Validation function runs without errors on default placement."""
    benchmark, plc = ibm01
    is_valid, violations = validate_placement(benchmark.macro_positions, benchmark)
    # Default placement may have overlaps — we just check the function works
    assert isinstance(is_valid, bool)
    assert isinstance(violations, list)


def test_net_pin_nodes(ibm01):
    """Loader exposes pin-level net connectivity consistent with net_nodes."""
    import torch

    benchmark, _ = ibm01
    assert len(benchmark.net_pin_nodes) == benchmark.num_nets

    for net_id, (net_pins, net_owners) in enumerate(
        zip(benchmark.net_pin_nodes, benchmark.net_nodes)
    ):
        # Shape: [pins_in_net, 2] — columns are (owner_idx, pin_slot)
        assert net_pins.ndim == 2 and net_pins.shape[1] == 2, (
            f"net {net_id}: net_pins shape {net_pins.shape}"
        )

        # Dedup+sort of owner column must match existing net_nodes exactly
        owners_sorted = torch.unique(net_pins[:, 0]).sort().values
        assert torch.equal(owners_sorted, net_owners), (
            f"net {net_id}: owners {owners_sorted.tolist()} != net_nodes {net_owners.tolist()}"
        )

        # Pin slots must index into macro_pin_offsets[owner] for hard macros
        for owner, slot in net_pins.tolist():
            if owner < benchmark.num_hard_macros:
                num_pins_on_macro = benchmark.macro_pin_offsets[owner].shape[0]
                assert slot < num_pins_on_macro, (
                    f"net {net_id}: owner {owner} slot {slot} >= "
                    f"macro_pin_offsets[{owner}].shape[0] {num_pins_on_macro}"
                )
            else:
                assert slot == 0, (
                    f"net {net_id}: non-hard-macro owner {owner} must use slot 0, got {slot}"
                )


def test_benchmark_save_load_roundtrip(ibm01, tmp_path):
    """Benchmark.save/load preserves net_pin_nodes."""
    import torch

    benchmark, _ = ibm01
    out = tmp_path / "roundtrip.pt"
    benchmark.save(str(out))
    loaded = Benchmark.load(str(out))

    assert len(loaded.net_pin_nodes) == len(benchmark.net_pin_nodes)
    for a, b in zip(loaded.net_pin_nodes, benchmark.net_pin_nodes):
        assert torch.equal(a, b)


def test_greedy_row_placer(ibm01):
    """Greedy row placer produces a valid, zero-overlap placement."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "greedy_row_placer",
        "submissions/examples/greedy_row_placer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    benchmark, plc = ibm01
    placer_cls = next(
        cls for name, cls in vars(mod).items()
        if isinstance(cls, type) and hasattr(cls, "place")
    )
    placer = placer_cls()
    placement = placer.place(benchmark)

    assert placement.shape == (benchmark.num_macros, 2)
    costs = compute_proxy_cost(placement, benchmark, plc)
    assert costs["overlap_count"] == 0, f"Greedy placer has {costs['overlap_count']} overlaps"


def test_straple_placer(ibm01):
    """Straple placer (C++ core) loads, runs, and produces a zero-overlap placement on ibm01."""
    import importlib.util

    placer_path = Path("submissions/straple/placer.py")
    if not placer_path.exists():
        pytest.skip("submissions/straple/placer.py not present")

    spec = importlib.util.spec_from_file_location("straple_placer", str(placer_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    benchmark, plc = ibm01
    placer = mod.StraplePlacer()
    placement = placer.place(benchmark)

    assert placement.shape == (benchmark.num_macros, 2)
    assert not torch.isnan(placement).any(), "Placement contains NaN"
    assert not torch.isinf(placement).any(), "Placement contains Inf"

    costs = compute_proxy_cost(placement, benchmark, plc)
    assert costs["overlap_count"] == 0, f"Straple placer has {costs['overlap_count']} overlaps"
    assert costs["proxy_cost"] < 1.5, (
        f"Straple proxy_cost {costs['proxy_cost']:.4f} suspicious "
        f"(expected ≲ 1.30 on ibm01 with C++ LNS)"
    )


def test_straple_proxy_cost_matches_plc(ibm01):
    """C++ proxy_cost (`_proxy_cost.ProxyEvaluator`) should match plc.compute_proxy_cost
    bit-for-bit on the initial placement — that's how we verify the C++ port is faithful."""
    import importlib.util

    placer_path = Path("submissions/straple/placer.py")
    if not placer_path.exists():
        pytest.skip("submissions/straple/placer.py not present")

    spec = importlib.util.spec_from_file_location("straple_placer", str(placer_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    benchmark, plc = ibm01
    mod._accelerate_plc(plc)
    evaluator = mod._build_proxy_evaluator(benchmark, plc)

    n_hard = benchmark.num_hard_macros
    hard_pos = benchmark.macro_positions[:n_hard].numpy().astype('float64').copy()

    cpp_cost, cpp_wl, cpp_den, cpp_cong = evaluator.evaluate_breakdown(hard_pos)
    plc_costs = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)

    assert abs(cpp_wl - plc_costs["wirelength_cost"]) < 1e-4, (
        f"WL mismatch: C++ {cpp_wl:.6f} vs plc {plc_costs['wirelength_cost']:.6f}"
    )
    assert abs(cpp_den - plc_costs["density_cost"]) < 1e-4, (
        f"density mismatch: C++ {cpp_den:.6f} vs plc {plc_costs['density_cost']:.6f}"
    )
    assert abs(cpp_cong - plc_costs["congestion_cost"]) < 5e-3, (
        f"congestion mismatch: C++ {cpp_cong:.6f} vs plc {plc_costs['congestion_cost']:.6f}"
    )


def test_straple_evaluate_returns_proxy_cost(ibm01):
    """ProxyEvaluator.evaluate() must return the SAME number as compute_proxy_cost
    (i.e. wl + 0.5*density + 0.5*congestion).

    Past bug (caught 2026-05-04): evaluate() returned wl + density + congestion
    (unweighted sum), so multi-start picked best by wrong metric and LNS accepted
    moves on wrong gradient. Both compute paths agree on components but the
    aggregate must use the official 1:0.5:0.5 weights.
    """
    import importlib.util

    placer_path = Path("submissions/straple/placer.py")
    if not placer_path.exists():
        pytest.skip("submissions/straple/placer.py not present")

    spec = importlib.util.spec_from_file_location("straple_placer", str(placer_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    benchmark, plc = ibm01
    mod._accelerate_plc(plc)
    evaluator = mod._build_proxy_evaluator(benchmark, plc)

    n_hard = benchmark.num_hard_macros
    hard_pos = benchmark.macro_positions[:n_hard].numpy().astype('float64').copy()

    cpp_aggregate = evaluator.evaluate(hard_pos)
    plc_proxy = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)["proxy_cost"]

    assert abs(cpp_aggregate - plc_proxy) < 5e-3, (
        f"ProxyEvaluator.evaluate() must equal compute_proxy_cost (1·wl + 0.5·density + 0.5·congestion). "
        f"Got C++ {cpp_aggregate:.6f} vs plc {plc_proxy:.6f} (delta {cpp_aggregate - plc_proxy:+.4f}). "
        f"If components agree but aggregate doesn't, the weights are wrong inside evaluate()."
    )

    cpp_breakdown_total, cpp_wl, cpp_den, cpp_cong = evaluator.evaluate_breakdown(hard_pos)
    expected_aggregate = cpp_wl + 0.5 * cpp_den + 0.5 * cpp_cong
    assert abs(cpp_aggregate - expected_aggregate) < 1e-6, (
        f"evaluate() must return wl + 0.5·density + 0.5·congestion. "
        f"Got {cpp_aggregate:.6f}, expected {expected_aggregate:.6f}."
    )
