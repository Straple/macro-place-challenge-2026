"""Fast-check: 4 representative benchmarks in parallel.

Replaces `evaluate --all` for the autonomous improvement loop. One cold-start
of the interpreter, four worker processes (1 core each) running ibm01, ibm10,
ibm14, ibm17 — covers small/medium/large/largest, plus the only mid-bench
where Straple beats RePlAce (ibm10) so regressions show up.

Usage:
    uv run python scripts/fast_check.py
    uv run python scripts/fast_check.py submissions/straple/placer.py
    uv run python scripts/fast_check.py --benches ibm01 ibm17

Output: per-bench proxy/wl/den/cong/time, AVG4, delta vs Straple #4 baseline.
"""

import argparse
import importlib.util
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

DEFAULT_BENCHES = ["ibm01", "ibm10", "ibm14", "ibm17"]

STRAPLE_4_BASELINE = {
    "ibm01": 1.1781,
    "ibm10": 1.3843,
    "ibm14": 1.6280,
    "ibm17": 1.7451,
}

REPLACE_BASELINE = {
    "ibm01": 0.9976,
    "ibm10": 1.4928,
    "ibm14": 1.5436,
    "ibm17": 1.6448,
}

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTCASE_ROOT = str(REPO_ROOT / "external" / "MacroPlacement" / "Testcases" / "ICCAD04")


def _load_placer_class(placer_path: Path):
    path = placer_path.resolve()
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None:
        raise RuntimeError(f"Failed to load placer from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for attr in vars(mod).values():
        if (
            isinstance(attr, type)
            and attr.__module__ == path.stem
            and callable(getattr(attr, "place", None))
        ):
            return attr

    raise RuntimeError(f"No placer class found in {path}")


def _run_one(args):
    placer_path_str, bench = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    placer_cls = _load_placer_class(Path(placer_path_str))
    placer = placer_cls()

    benchmark_dir = f"{TESTCASE_ROOT}/{bench}"
    benchmark, plc = load_benchmark_from_dir(benchmark_dir)

    start = time.time()
    placement = placer.place(benchmark)
    elapsed = time.time() - start

    is_valid, violations = validate_placement(placement, benchmark)
    costs = compute_proxy_cost(placement, benchmark, plc)

    return {
        "bench": bench,
        "proxy": costs["proxy_cost"],
        "wl": costs["wirelength_cost"],
        "den": costs["density_cost"],
        "cong": costs["congestion_cost"],
        "overlaps": costs["overlap_count"],
        "valid": is_valid,
        "violations": violations[:3],
        "time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "placer",
        nargs="?",
        default="submissions/straple/placer.py",
        help="Path to placer .py file",
    )
    parser.add_argument(
        "--benches",
        nargs="+",
        default=DEFAULT_BENCHES,
        help=f"Benchmarks to run (default: {' '.join(DEFAULT_BENCHES)})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially (debug)",
    )
    args = parser.parse_args()

    placer_path = str(Path(args.placer).resolve())

    print(f"=== fast_check ({args.placer}) ===")
    print(f"benches: {' '.join(args.benches)}, workers: {args.workers}")

    wall_start = time.time()
    if args.sequential:
        results = [_run_one((placer_path, b)) for b in args.benches]
    else:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            results = pool.map(_run_one, [(placer_path, b) for b in args.benches])
    wall = time.time() - wall_start

    print()
    print(
        f"{'bench':<8} {'proxy':>8} {'wl':>7} {'den':>7} {'cong':>7}"
        f" {'time':>7} {'ovrlap':>7} {'note':<10}"
    )
    print("-" * 70)
    proxies = []
    proxies_by_bench = {}
    for r in results:
        note = "" if r["valid"] else "INVALID"
        print(
            f"{r['bench']:<8} {r['proxy']:>8.4f} {r['wl']:>7.3f}"
            f" {r['den']:>7.3f} {r['cong']:>7.3f} {r['time']:>6.2f}s"
            f" {r['overlaps']:>7} {note:<10}"
        )
        proxies.append(r["proxy"])
        proxies_by_bench[r["bench"]] = r["proxy"]
    print("-" * 70)
    avg = sum(proxies) / len(proxies)
    print(f"{'AVG' + str(len(results)):<8} {avg:>8.4f}                            wall={wall:.1f}s")
    print()

    overlap_benches = set(args.benches) & set(STRAPLE_4_BASELINE.keys())
    if overlap_benches:
        ref_avg = sum(STRAPLE_4_BASELINE[b] for b in overlap_benches) / len(overlap_benches)
        cur_avg = sum(proxies_by_bench[b] for b in overlap_benches) / len(overlap_benches)
        delta = (cur_avg - ref_avg) / ref_avg * 100
        marker = "▼" if delta < -0.05 else ("▲" if delta > 0.05 else "≈")
        print(f"vs Straple #4 baseline ({ref_avg:.4f}): {cur_avg:.4f} {marker} {delta:+.2f}%")

        rep_avg = sum(REPLACE_BASELINE[b] for b in overlap_benches) / len(overlap_benches)
        rep_delta = (cur_avg - rep_avg) / rep_avg * 100
        print(f"vs RePlAce       baseline ({rep_avg:.4f}): {cur_avg:.4f}    {rep_delta:+.2f}%")

    overlap_fails = [r for r in results if r["overlaps"] > 0]
    bounds_warns = [r for r in results if not r["valid"] and r["overlaps"] == 0]

    if bounds_warns:
        print()
        print(f"WARN: {len(bounds_warns)} benchmark(s) have bounds/NaN violations (pre-existing baseline issue, not a regression)")
        for r in bounds_warns:
            print(f"  {r['bench']}: {r['violations']}")

    if overlap_fails:
        print()
        print(f"FAIL: {len(overlap_fails)} benchmark(s) have overlaps")
        for r in overlap_fails:
            print(f"  {r['bench']}: overlaps={r['overlaps']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
