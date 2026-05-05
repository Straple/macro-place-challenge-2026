"""
Straple Placer (C++ core) — Legalize + SA + LNS, fast native implementation.

Hot loops (legalize, SA refinement, LNS destroy/repair) live in
submissions/straple/cpp/placer_core.cpp and are loaded via pybind11.
Python here orchestrates: load the benchmark, build edges, run the C++
pipeline, and call the (Python) TILOS proxy_cost for LNS accept/reject.

Usage:
    uv run evaluate submissions/straple/placer.py
    uv run evaluate submissions/straple/placer.py --all

Build the native module once (or after editing the C++ source):
    submissions/straple/cpp/build.sh
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from macro_place.benchmark import Benchmark


_CPP_DIR = Path(__file__).resolve().parent / "cpp"
if str(_CPP_DIR) not in sys.path:
    sys.path.insert(0, str(_CPP_DIR))


def _import_native_or_build():
    try:
        import _placer_core
        import _proxy_cost
        return _placer_core, _proxy_cost
    except ImportError:
        build_script = _CPP_DIR / "build.sh"
        if not build_script.exists():
            raise
        os.system(f"bash {build_script}")
        import _placer_core
        import _proxy_cost
        return _placer_core, _proxy_cost


_placer_core, _proxy_cost = _import_native_or_build()


def _load_plc(name):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        _accelerate_plc(plc)
        return plc
    ng45 = {
        "ariane133_ng45": "ariane133",
        "ariane136_ng45": "ariane136",
        "nvdla_ng45": "nvdla",
        "mempool_tile_ng45": "mempool_tile",
    }
    d = ng45.get(name)
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            _accelerate_plc(plc)
            return plc
    return None


def _accelerate_plc(plc):
    if not isinstance(plc.soft_macro_pin_indices, set):
        plc.soft_macro_pin_indices = set(plc.soft_macro_pin_indices)
    if not isinstance(plc.hard_macro_pin_indices, set):
        plc.hard_macro_pin_indices = set(plc.hard_macro_pin_indices)


def _extract_edges(benchmark, plc):
    n_hard = benchmark.num_hard_macros
    name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx
    edge_dict = {}
    for driver, sinks in plc.nets.items():
        macros = set()
        for pin in [driver] + sinks:
            parent = pin.split("/")[0]
            if parent in name_to_bidx:
                macros.add(name_to_bidx[parent])
        if len(macros) >= 2:
            macro_list = sorted(macros)
            weight = 1.0 / (len(macro_list) - 1)
            for i in range(len(macro_list)):
                for j in range(i + 1, len(macro_list)):
                    pair = (macro_list[i], macro_list[j])
                    edge_dict[pair] = edge_dict.get(pair, 0) + weight
    if not edge_dict:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float64)
    edges = np.array(list(edge_dict.keys()), dtype=np.int32)
    weights = np.array([edge_dict[e] for e in edge_dict], dtype=np.float64)
    return edges, weights


_PIN_KIND_PORT = 0
_PIN_KIND_HARD = 1
_PIN_KIND_SOFT = 2


def _build_proxy_evaluator(benchmark, plc, soft_positions_override=None):
    hard_name_to_bidx = {}
    for bidx, idx in enumerate(plc.hard_macro_indices):
        hard_name_to_bidx[plc.modules_w_pins[idx].get_name()] = bidx
    soft_name_to_sidx = {}
    for sidx, idx in enumerate(plc.soft_macro_indices):
        soft_name_to_sidx[plc.modules_w_pins[idx].get_name()] = sidx
    port_name_to_pidx = {}
    for pidx, idx in enumerate(plc.port_indices):
        port_name_to_pidx[plc.modules_w_pins[idx].get_name()] = pidx

    pin_kinds = []
    pin_owners = []
    pin_offsets_x = []
    pin_offsets_y = []
    net_starts = [0]
    net_weights = []
    net_source_slots = []

    for driver, sinks in plc.nets.items():
        net_pins = [driver] + sinks
        net_pin_data = []
        source_slot = 0
        slot = 0
        for pin_name in net_pins:
            parent_name = pin_name.split("/")[0]
            pin_idx = plc.mod_name_to_indices.get(pin_name)
            if pin_idx is None:
                slot += 1
                continue
            pin_node = plc.modules_w_pins[pin_idx]
            pin_type = pin_node.get_type()
            if pin_type == "PORT":
                if parent_name not in port_name_to_pidx:
                    slot += 1
                    continue
                kind = _PIN_KIND_PORT
                owner = port_name_to_pidx[parent_name]
                offset_x = 0.0
                offset_y = 0.0
            elif pin_type == "MACRO_PIN":
                ox, oy = pin_node.get_offset()
                if parent_name in hard_name_to_bidx:
                    kind = _PIN_KIND_HARD
                    owner = hard_name_to_bidx[parent_name]
                elif parent_name in soft_name_to_sidx:
                    kind = _PIN_KIND_SOFT
                    owner = soft_name_to_sidx[parent_name]
                else:
                    slot += 1
                    continue
                offset_x = float(ox)
                offset_y = float(oy)
            else:
                slot += 1
                continue
            if pin_name == driver:
                source_slot = len(net_pin_data)
            net_pin_data.append((kind, owner, offset_x, offset_y))
            slot += 1
        if not net_pin_data:
            continue
        driver_node = plc.modules_w_pins[plc.mod_name_to_indices[driver]]
        weight = float(driver_node.get_weight())
        for kind, owner, ox, oy in net_pin_data:
            pin_kinds.append(kind)
            pin_owners.append(owner)
            pin_offsets_x.append(ox)
            pin_offsets_y.append(oy)
        net_starts.append(len(pin_kinds))
        net_weights.append(weight)
        net_source_slots.append(source_slot)

    n_hard = benchmark.num_hard_macros
    n_soft = benchmark.num_soft_macros

    hard_sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
    soft_sizes = benchmark.macro_sizes[n_hard:].numpy().astype(np.float64)
    if soft_positions_override is not None:
        soft_positions = np.asarray(soft_positions_override, dtype=np.float64)
        if soft_positions.shape != (n_soft, 2):
            raise ValueError(
                f"soft_positions_override shape {soft_positions.shape} != ({n_soft}, 2)"
            )
    else:
        soft_positions = benchmark.macro_positions[n_hard:].numpy().astype(np.float64)
    port_positions = benchmark.port_positions.numpy().astype(np.float64) if benchmark.port_positions.numel() > 0 else np.zeros((0, 2), dtype=np.float64)

    evaluator = _proxy_cost.ProxyEvaluator()
    evaluator.initialize(
        n_hard, n_soft,
        float(plc.width), float(plc.height),
        int(plc.grid_col), int(plc.grid_row),
        float(plc.hroutes_per_micron), float(plc.vroutes_per_micron),
        float(plc.hrouting_alloc), float(plc.vrouting_alloc),
        int(plc.smooth_range),
        float(plc.net_cnt) if plc.net_cnt > 0 else 1.0,
        hard_sizes, soft_sizes, soft_positions, port_positions,
        np.asarray(pin_kinds, dtype=np.int32),
        np.asarray(pin_owners, dtype=np.int32),
        np.asarray(pin_offsets_x, dtype=np.float64),
        np.asarray(pin_offsets_y, dtype=np.float64),
        np.asarray(net_starts, dtype=np.int32),
        np.asarray(net_weights, dtype=np.float64),
        np.asarray(net_source_slots, dtype=np.int32),
    )
    return evaluator


def _worker_run_start(start_idx, args, seed_base, refine_iters, lns_outer_iters,
                      lns_destroy_size, benchmark, plc):
    placer = StraplePlacer(
        seed=seed_base,
        refine_iters=refine_iters,
        lns_outer_iters=lns_outer_iters,
        lns_destroy_size=lns_destroy_size,
        verbose=0,
        analytical_steps=0,
    )
    evaluator = _build_proxy_evaluator(
        benchmark, plc,
        soft_positions_override=args.get("gradient_soft_pos"),
    )
    return placer._run_one_start(start_idx, args, evaluator, plc)


class StraplePlacer:
    def __init__(
        self,
        seed: int = 42,
        refine_iters: int = 3000,
        lns_outer_iters: int = 30,
        lns_destroy_size: int = 8,
        verbose: int = 0,
        analytical_steps: int = None,
        analytical_lr: float = 0.3,
        analytical_lambda_density: float = 50000.0,
        analytical_target_util: float = 0.2,
        analytical_lambda_schedule=None,
        analytical_gamma_schedule=None,
        analytical_gamma_frac: float = 0.05,
    ):
        if analytical_steps is None:
            analytical_steps = int(os.environ.get("STRAPLE_ANALYTICAL_STEPS", "0"))
        self.seed = seed
        self.refine_iters = refine_iters
        self.lns_outer_iters = lns_outer_iters
        self.lns_destroy_size = lns_destroy_size
        self.verbose = verbose if verbose else int(os.environ.get("STRAPLE_VERBOSE", "0"))
        self.analytical_steps = analytical_steps
        self.analytical_lr = analytical_lr
        self.analytical_lambda_density = analytical_lambda_density
        self.analytical_target_util = analytical_target_util
        self.analytical_lambda_schedule = analytical_lambda_schedule
        self.analytical_gamma_schedule = analytical_gamma_schedule
        self.analytical_gamma_frac = analytical_gamma_frac
        env_preset = os.environ.get("STRAPLE_ANALYTICAL_PRESET", "")
        if env_preset == "cold_start":
            self.analytical_lambda_schedule = [(0.0, 0.0), (0.2, 100.0),
                                               (0.5, 5000.0), (1.0, 50000.0)]
            self.analytical_gamma_schedule = [(0.0, 1.5), (1.0, 0.3)]

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _perturb_initial(self, args, start_idx):
        rng = np.random.default_rng(self.seed + start_idx * 1000 + 7)
        canvas_min = min(args["canvas_w"], args["canvas_h"])
        perturb_idx = start_idx - args["num_orig_starts"]
        env_scales = os.environ.get("STRAPLE_PERTURB_SCALES", "")
        if env_scales:
            scales = [float(s) for s in env_scales.split(",") if s.strip()]
            sigma_factor = scales[perturb_idx % len(scales)] if scales else 0.05
        else:
            default_scales = [0.05, 0.10, 0.15, 0.20]
            sigma_factor = default_scales[perturb_idx % len(default_scales)]
        sigma = sigma_factor * canvas_min
        movable = args["movable_mask"]
        perturb = rng.normal(0.0, sigma, size=args["initial_pos"].shape)
        perturb[~movable] = 0.0
        seed_pos = args["initial_pos"] + perturb
        sizes = args["sizes"]
        half_w = sizes[:, 0] / 2.0
        half_h = sizes[:, 1] / 2.0
        seed_pos[:, 0] = np.clip(seed_pos[:, 0], half_w, args["canvas_w"] - half_w)
        seed_pos[:, 1] = np.clip(seed_pos[:, 1], half_h, args["canvas_h"] - half_h)
        seed_pos[~movable] = args["initial_pos"][~movable]
        if self.verbose:
            self._log(f"[{args['bench_label']}] start#{start_idx} perturbed: "
                      f"sigma={sigma:.2f} (factor={sigma_factor:.3f})")
        return seed_pos

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        import time
        t_place_start = time.time()
        bench_label = getattr(benchmark, "name", "?")
        self._log(f"[{bench_label}] === StraplePlacer.place() ===")

        preset = os.environ.get("STRAPLE_PRESET", "")
        if preset == "high_effort":
            n_total_for_preset = benchmark.num_macros
            if n_total_for_preset < 1500:
                n_orig, n_pert = 4, 12
                lns_factor, lns_cap = 120, 80000
            elif n_total_for_preset < 2500:
                n_orig, n_pert = 3, 9
                lns_factor, lns_cap = 90, 65000
            else:
                n_orig, n_pert = 2, 6
                lns_factor, lns_cap = 70, 55000
            os.environ.setdefault("STRAPLE_NUM_STARTS", str(n_orig))
            os.environ.setdefault("STRAPLE_PERTURB_EXTRA_STARTS", str(n_pert))
            os.environ.setdefault(
                "STRAPLE_PERTURB_SCALES",
                "0.10,0.20,0.30,0.50,0.10,0.20,0.30,0.50,0.10,0.20,0.30,0.50",
            )
            os.environ.setdefault("STRAPLE_LNS_OUTER_FACTOR", str(lns_factor))
            os.environ.setdefault("STRAPLE_LNS_OUTER_CAP", str(lns_cap))
            os.environ.setdefault("STRAPLE_DESTROY_CAP", "32")
            self._log(f"[{bench_label}] PRESET=high_effort: "
                      f"starts={n_orig}+{n_pert} factor={lns_factor} cap={lns_cap}")

        n_hard = benchmark.num_hard_macros
        plc = _load_plc(benchmark.name)

        recorder = None
        vis_path = os.environ.get("STRAPLE_VIS_VIDEO", "")
        if vis_path and plc is not None:
            _STRAPLE_DIR = str(Path(__file__).resolve().parent)
            if _STRAPLE_DIR not in sys.path:
                sys.path.insert(0, _STRAPLE_DIR)
            from visualizer import PlacementRecorder
            interval = int(os.environ.get("STRAPLE_VIS_INTERVAL", "100"))
            max_frames = int(os.environ.get("STRAPLE_VIS_MAX_FRAMES", "60"))
            recorder = PlacementRecorder(benchmark, plc, vis_path,
                                         interval=interval, max_frames=max_frames)
            self._log(f"[{bench_label}] visualizer ENABLED → {vis_path} "
                      f"(interval={interval}, max_frames={max_frames})")
        t_after_plc = time.time()
        self._log(f"[{bench_label}] _load_plc: {t_after_plc - t_place_start:.2f}s "
                  f"(n_hard={n_hard})")

        if plc is not None:
            edges, edge_weights = _extract_edges(benchmark, plc)
        else:
            edges = np.zeros((0, 2), dtype=np.int32)
            edge_weights = np.zeros(0, dtype=np.float64)
        t_after_edges = time.time()
        self._log(f"[{bench_label}] _extract_edges: {t_after_edges - t_after_plc:.2f}s "
                  f"(edges={len(edges)})")

        demo_mode = os.environ.get("STRAPLE_DEMO", "")
        if demo_mode in ("force", "gradient") and plc is not None:
            _STRAPLE_DIR = str(Path(__file__).resolve().parent)
            if _STRAPLE_DIR not in sys.path:
                sys.path.insert(0, _STRAPLE_DIR)
            demo_iters = int(os.environ.get("STRAPLE_DEMO_ITERS", "300"))
            demo_seed = int(os.environ.get("STRAPLE_DEMO_SEED", "42"))
            time_budget = float(os.environ.get("STRAPLE_DEMO_TIME_BUDGET", "0"))
            score_png = os.environ.get("STRAPLE_DEMO_SCORE_PNG", "")
            score_sample_every = float(
                os.environ.get("STRAPLE_DEMO_SCORE_SAMPLE_S", "1.0"))
            self._log(f"[{bench_label}] DEMO mode={demo_mode} "
                      f"iters={demo_iters} time_budget={time_budget}s seed={demo_seed} "
                      f"score_png={score_png or 'none'}")
            t0 = time.time()
            if demo_mode == "force":
                from force_demo import force_directed_demo
                result = force_directed_demo(
                    benchmark, plc, edges, edge_weights,
                    recorder=recorder, num_iters=demo_iters, seed=demo_seed,
                    time_budget=time_budget, score_png=score_png,
                    score_sample_every_s=score_sample_every,
                )
            else:
                from gradient_demo import gradient_demo
                result = gradient_demo(
                    benchmark, plc,
                    recorder=recorder, num_steps=demo_iters, seed=demo_seed,
                    time_budget=time_budget, score_png=score_png,
                    score_sample_every_s=score_sample_every,
                )
            if isinstance(result, tuple):
                final_pos = result[0]
            else:
                final_pos = result
            self._log(f"[{bench_label}] {demo_mode} demo: {time.time()-t0:.2f}s")
            full = benchmark.macro_positions.clone()
            n_demo = final_pos.shape[0]
            full[:n_demo] = torch.tensor(final_pos, dtype=torch.float32)
            if recorder is not None:
                try:
                    recorder.render()
                except Exception as exc:
                    print(f"[visualizer] render failed: {exc}", file=sys.stderr)
            self._log(f"[{bench_label}] === DEMO DONE total={time.time()-t_place_start:.2f}s ===")
            return full

        initial_pos = benchmark.macro_positions[:n_hard].numpy().astype(np.float64).copy()
        sizes = benchmark.macro_sizes[:n_hard].numpy().astype(np.float64)
        movable_mask = benchmark.get_movable_mask()[:n_hard].numpy().astype(np.bool_)
        num_movable = int(movable_mask.sum())
        canvas_w = float(benchmark.canvas_width)
        canvas_h = float(benchmark.canvas_height)
        self._log(f"[{bench_label}] num_movable={num_movable} canvas={canvas_w:.0f}x{canvas_h:.0f}")

        analytical_pos = None
        if plc is not None and self.analytical_steps > 0:
            _STRAPLE_DIR = str(Path(__file__).resolve().parent)
            if _STRAPLE_DIR not in sys.path:
                sys.path.insert(0, _STRAPLE_DIR)
            from analytical_seed import analytical_seed
            t0 = time.time()
            analytical_pos = analytical_seed(
                benchmark, plc,
                num_steps=self.analytical_steps,
                lr=self.analytical_lr,
                lambda_density=self.analytical_lambda_density,
                gamma_frac=self.analytical_gamma_frac,
                target_util=self.analytical_target_util,
                lambda_schedule=self.analytical_lambda_schedule,
                gamma_schedule=self.analytical_gamma_schedule,
                log_every=(max(1, self.analytical_steps // 5) if self.verbose else 0),
                log_proxy=bool(self.verbose),
                label=bench_label,
            )
            self._log(f"[{bench_label}] analytical_seed({self.analytical_steps} steps): "
                      f"{time.time()-t0:.2f}s")

        gradient_hard_pos = None
        gradient_soft_pos = None

        # External gradient seed file (from scripts/gpu_batch_search.py)
        seed_file = os.environ.get("STRAPLE_GRADIENT_SEED_FILE", "")
        if seed_file and Path(seed_file).exists():
            try:
                import pickle
                with open(seed_file, "rb") as f:
                    seed_data = pickle.load(f)
                hp = np.asarray(seed_data["hard"], dtype=np.float64)
                sp = np.asarray(seed_data["soft"], dtype=np.float64)
                if hp.shape[0] == n_hard:
                    gradient_hard_pos = hp.copy()
                    self._log(f"[{bench_label}] loaded external gradient hard "
                              f"from {seed_file} (proxy={seed_data.get('proxy', '?')})")
                if sp.shape[0] == benchmark.num_soft_macros and \
                        os.environ.get("STRAPLE_GRADIENT_SEED_USE_SOFT", "0") == "1":
                    gradient_soft_pos = sp.copy()
                    self._log(f"[{bench_label}] loaded external gradient soft")
            except Exception as exc:
                print(f"[{bench_label}] failed to load external seed: {exc}",
                      file=sys.stderr)

        use_gradient_seed = (
            os.environ.get("STRAPLE_USE_GRADIENT_SEED", "0") == "1"
            and plc is not None
            and gradient_hard_pos is None
        )
        if use_gradient_seed:
            _STRAPLE_DIR = str(Path(__file__).resolve().parent)
            if _STRAPLE_DIR not in sys.path:
                sys.path.insert(0, _STRAPLE_DIR)
            try:
                from gradient_demo import gradient_demo
                env_snapshot = {}
                for k, v in (("STRAPLE_DEMO_INIT", "anchor_soft"),
                             ("STRAPLE_DEMO_PLACE_ALL", "1"),
                             ("STRAPLE_DEMO_FINISH_LEGALIZE", "1")):
                    env_snapshot[k] = os.environ.get(k)
                    if env_snapshot[k] is None:
                        os.environ[k] = v
                grad_steps = int(os.environ.get("STRAPLE_GRADIENT_SEED_STEPS", "500"))
                grad_seed = self.seed
                grad_n = max(1, int(
                    os.environ.get("STRAPLE_GRADIENT_SEED_NUM", "1")))
                self._log(f"[{bench_label}] gradient seed pre-pass "
                          f"({grad_n}x{grad_steps} steps)...")
                t0 = time.time()
                from macro_place.objective import compute_proxy_cost
                best_grad_pos = None
                best_grad_proxy = float("inf")
                gp = None
                for k in range(grad_n):
                    sd = grad_seed + k * 1009
                    grad_result = gradient_demo(
                        benchmark, plc, recorder=None,
                        num_steps=grad_steps, seed=sd,
                        time_budget=0.0, score_png="",
                        score_sample_every_s=1.0,
                    )
                    gp = grad_result[0] if isinstance(grad_result, tuple) else grad_result
                    gp = np.asarray(gp, dtype=np.float64)
                    full_test = benchmark.macro_positions.clone()
                    full_test[:n_hard] = torch.tensor(gp[:n_hard], dtype=torch.float32)
                    if gp.shape[0] > n_hard:
                        nsl = benchmark.num_soft_macros
                        full_test[n_hard:n_hard + nsl] = torch.tensor(
                            gp[n_hard:n_hard + nsl], dtype=torch.float32)
                    cost = compute_proxy_cost(full_test, benchmark, plc)
                    self._log(f"[{bench_label}] grad_seed#{k} sd={sd} "
                              f"proxy={cost['proxy_cost']:.4f} "
                              f"ovrlp={cost['overlap_count']}")
                    if (cost['overlap_count'] == 0
                            and cost['proxy_cost'] < best_grad_proxy):
                        best_grad_proxy = cost['proxy_cost']
                        best_grad_pos = gp
                grad_pos = best_grad_pos if best_grad_pos is not None else gp
                if grad_pos is None:
                    raise RuntimeError("all gradient seeds failed")
                if grad_pos.shape[0] >= n_hard:
                    gradient_hard_pos = grad_pos[:n_hard].copy()
                use_grad_soft = (
                    os.environ.get("STRAPLE_GRADIENT_SEED_USE_SOFT", "0") == "1"
                )
                if use_grad_soft and grad_pos.shape[0] > n_hard:
                    n_soft_local = benchmark.num_soft_macros
                    gradient_soft_pos = grad_pos[n_hard:n_hard + n_soft_local].copy()
                self._log(f"[{bench_label}] best grad proxy={best_grad_proxy:.4f}")
                for k, prev in env_snapshot.items():
                    if prev is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = prev
                self._log(f"[{bench_label}] gradient seed: {time.time()-t0:.2f}s "
                          f"(hard={gradient_hard_pos is not None}, "
                          f"soft={gradient_soft_pos is not None})")
            except Exception as exc:
                print(f"[{bench_label}] gradient seed FAILED: {exc}", file=sys.stderr)
                gradient_hard_pos = None
                gradient_soft_pos = None

        num_orig_starts = 3 if num_movable >= 300 else 1
        env_n = int(os.environ.get("STRAPLE_NUM_STARTS", "0"))
        if env_n > 0:
            num_orig_starts = env_n
        num_perturbed_starts = int(os.environ.get("STRAPLE_PERTURB_EXTRA_STARTS", "0"))
        if num_movable < 300:
            num_perturbed_starts = 0
        num_gradient_starts = 1 if gradient_hard_pos is not None else 0
        num_starts = (num_orig_starts + num_perturbed_starts
                      + num_gradient_starts
                      + (1 if analytical_pos is not None else 0))

        evaluator = None
        if plc is not None and self.lns_outer_iters > 0:
            t0 = time.time()
            evaluator = _build_proxy_evaluator(
                benchmark, plc, soft_positions_override=gradient_soft_pos)
            self._log(f"[{bench_label}] _build_proxy_evaluator: {time.time()-t0:.2f}s "
                      f"(soft_override={gradient_soft_pos is not None})")

        best_pos = None
        best_cost = float("inf")

        start_args = {
            "initial_pos": initial_pos,
            "sizes": sizes,
            "movable_mask": movable_mask,
            "edges": edges,
            "edge_weights": edge_weights,
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "num_movable": num_movable,
            "analytical_pos": analytical_pos,
            "gradient_hard_pos": gradient_hard_pos,
            "gradient_soft_pos": gradient_soft_pos,
            "num_starts": num_starts,
            "num_orig_starts": num_orig_starts,
            "num_perturbed_starts": num_perturbed_starts,
            "num_gradient_starts": num_gradient_starts,
            "bench_label": bench_label,
        }

        if recorder is not None:
            recorder.add(initial_pos, "initial")

        parallel_workers = int(os.environ.get("STRAPLE_PARALLEL_STARTS", "0"))
        if parallel_workers > 1 and evaluator is not None and num_starts > 1:
            self._log(f"[{bench_label}] === parallel multi-start ({parallel_workers} workers, "
                      f"{num_starts} starts) ===")
            results = self._run_starts_parallel(start_args, benchmark, plc,
                                                min(parallel_workers, num_starts))
            for trial_pos, trial_cost, start_idx in results:
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_pos = trial_pos
                    self._log(f"[{bench_label}] start#{start_idx} new best={trial_cost:.4f}")
                else:
                    self._log(f"[{bench_label}] start#{start_idx} cost={trial_cost:.4f} "
                              f"(not best, best={best_cost:.4f})")
        else:
            for start_idx in range(num_starts):
                trial_pos, trial_cost = self._run_one_start(start_idx, start_args, evaluator, plc,
                                                            recorder=recorder)
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_pos = trial_pos
                    self._log(f"[{bench_label}] start#{start_idx} new best={trial_cost:.4f}")
                else:
                    self._log(f"[{bench_label}] start#{start_idx} cost={trial_cost:.4f} "
                              f"(not best, best={best_cost:.4f})")

        if evaluator is not None and best_pos is not None:
            for refine_iter in range(3):
                t0 = time.time()
                state_r = _placer_core.PlacerState()
                state_r.initialize(
                    best_pos, sizes, movable_mask,
                    edges, edge_weights,
                    canvas_w, canvas_h, int(self.seed + 9999 + refine_iter * 100),
                )
                self._lns_loop(state_r, evaluator, plc, num_movable,
                               bench_label, 99 + refine_iter, recorder=recorder)
                refined_pos = state_r.current_positions()
                refined_cost = evaluator.evaluate(refined_pos)
                self._log(f"[{bench_label}] REFINE pass#{refine_iter}: "
                          f"{time.time()-t0:.2f}s "
                          f"cost {best_cost:.4f} -> {refined_cost:.4f} "
                          f"(delta {refined_cost-best_cost:+.4f})")
                if refined_cost < best_cost:
                    best_cost = refined_cost
                    best_pos = refined_pos
                    if recorder is not None:
                        recorder.add(best_pos, f"refine#{refine_iter} cost={best_cost:.4f}")
                else:
                    break

        full = benchmark.macro_positions.clone()
        full[:n_hard] = torch.tensor(best_pos, dtype=torch.float32)
        if gradient_soft_pos is not None:
            full[n_hard:n_hard + gradient_soft_pos.shape[0]] = torch.tensor(
                gradient_soft_pos, dtype=torch.float32)

        refine_soft = (
            os.environ.get("STRAPLE_REFINE_SOFT", "0") == "1"
            and plc is not None
            and benchmark.num_soft_macros > 0
        )
        if refine_soft and best_pos is not None:
            try:
                from copy import deepcopy
                _STRAPLE_DIR = str(Path(__file__).resolve().parent)
                if _STRAPLE_DIR not in sys.path:
                    sys.path.insert(0, _STRAPLE_DIR)
                from gradient_demo import gradient_demo
                refine_steps = int(
                    os.environ.get("STRAPLE_REFINE_SOFT_STEPS", "300"))
                bench_refine = deepcopy(benchmark)
                bench_refine.macro_positions = full.clone()
                bench_refine.macro_fixed = bench_refine.macro_fixed.clone()
                bench_refine.macro_fixed[:n_hard] = True
                env_snap = {}
                for k, v in (("STRAPLE_DEMO_INIT", "current"),
                             ("STRAPLE_DEMO_PLACE_ALL", "1"),
                             ("STRAPLE_DEMO_FINISH_LEGALIZE", "0"),
                             ("STRAPLE_DEMO_ANCHOR_LOSS", "0")):
                    env_snap[k] = os.environ.get(k)
                    os.environ[k] = v
                self._log(f"[{bench_label}] post-ALNS soft refine "
                          f"({refine_steps} steps)...")
                t0 = time.time()
                refined = gradient_demo(
                    bench_refine, plc, recorder=None,
                    num_steps=refine_steps, seed=self.seed + 31337,
                    time_budget=0.0, score_png="",
                )
                refined_pos = refined[0] if isinstance(refined, tuple) else refined
                refined_pos = np.asarray(refined_pos, dtype=np.float64)
                for k, prev in env_snap.items():
                    if prev is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = prev
                full_refined = full.clone()
                if refined_pos.shape[0] >= n_hard:
                    full_refined[n_hard:] = torch.tensor(
                        refined_pos[n_hard:], dtype=torch.float32)
                from macro_place.objective import compute_proxy_cost
                cost_before = compute_proxy_cost(full, benchmark, plc)
                cost_after = compute_proxy_cost(full_refined, benchmark, plc)
                self._log(f"[{bench_label}] soft refine: "
                          f"{time.time()-t0:.2f}s "
                          f"proxy {cost_before['proxy_cost']:.4f} -> "
                          f"{cost_after['proxy_cost']:.4f} "
                          f"(ovrlp {cost_before['overlap_count']} -> "
                          f"{cost_after['overlap_count']})")
                if (cost_after['proxy_cost'] < cost_before['proxy_cost']
                        and cost_after['overlap_count'] == 0):
                    full = full_refined
            except Exception as exc:
                print(f"[{bench_label}] soft refine FAILED: {exc}", file=sys.stderr)

        if recorder is not None and best_pos is not None:
            recorder.add(full[:n_hard].numpy().astype(np.float64),
                         f"FINAL best={best_cost:.4f}")
            try:
                recorder.render()
            except Exception as exc:
                print(f"[visualizer] render failed: {exc}", file=sys.stderr)

        self._log(f"[{bench_label}] === DONE total={time.time()-t_place_start:.2f}s "
                  f"final_cost={best_cost:.4f} ===")
        return full

    def _run_one_start(self, start_idx, args, evaluator, plc, recorder=None):
        import time
        is_analytical_start = (args["analytical_pos"] is not None
                               and start_idx == args["num_starts"] - 1)
        num_orig = args["num_orig_starts"]
        num_perturbed = args.get("num_perturbed_starts", 0)
        num_gradient = args.get("num_gradient_starts", 0)
        is_perturbed_start = (not is_analytical_start
                              and start_idx >= num_orig
                              and start_idx < num_orig + num_perturbed)
        is_gradient_start = (not is_analytical_start
                             and not is_perturbed_start
                             and num_gradient > 0
                             and start_idx >= num_orig + num_perturbed
                             and start_idx < num_orig + num_perturbed + num_gradient)
        seed = self.seed + start_idx
        t_start_iter = time.time()

        state = _placer_core.PlacerState()
        if is_analytical_start:
            seed_pos = args["analytical_pos"]
        elif is_gradient_start:
            seed_pos = args["gradient_hard_pos"]
        elif is_perturbed_start:
            seed_pos = self._perturb_initial(args, start_idx)
        else:
            seed_pos = args["initial_pos"]
        state.initialize(
            seed_pos, args["sizes"], args["movable_mask"],
            args["edges"], args["edge_weights"],
            args["canvas_w"], args["canvas_h"], int(seed),
        )

        if recorder is not None and (is_analytical_start or is_perturbed_start
                                     or is_gradient_start):
            recorder.add(state.current_positions(), f"start#{start_idx} pre-legalize")

        if is_analytical_start or is_gradient_start:
            state.legalize_min_displacement(500)
            state.legalize()
        else:
            state.legalize()
        cost_after_legal = evaluator.evaluate(state.current_positions()) if evaluator else 0.0
        if recorder is not None:
            recorder.add(state.current_positions(),
                         f"start#{start_idx} post-legalize cost={cost_after_legal:.4f}")

        sa_iters_to_run = self.refine_iters if (
            args["num_movable"] < 300
            and not is_analytical_start
            and not is_gradient_start) else 0
        if sa_iters_to_run > 0:
            state.sa_refine(sa_iters_to_run)
        cost_after_sa = evaluator.evaluate(state.current_positions()) if evaluator else 0.0
        if recorder is not None and sa_iters_to_run > 0:
            recorder.add(state.current_positions(),
                         f"start#{start_idx} post-SA cost={cost_after_sa:.4f}")

        self._log(f"[{args['bench_label']}] start#{start_idx} seed={seed}: "
                  f"legalize proxy={cost_after_legal:.4f} | "
                  f"sa_refine({sa_iters_to_run}) proxy={cost_after_sa:.4f}")

        if evaluator is not None:
            t0 = time.time()
            lns_log = self._lns_loop(state, evaluator, plc, args["num_movable"],
                                     args["bench_label"], start_idx, recorder=recorder)
            self._log(f"[{args['bench_label']}] start#{start_idx} LNS: {time.time()-t0:.2f}s "
                      f"cost {cost_after_sa:.4f} -> {lns_log['final_cost']:.4f}")

        trial_pos = state.current_positions()
        trial_cost = evaluator.evaluate(trial_pos) if evaluator else float(start_idx)
        self._log(f"[{args['bench_label']}] start#{start_idx} done in "
                  f"{time.time()-t_start_iter:.1f}s cost={trial_cost:.4f}")
        return trial_pos, trial_cost

    def _run_starts_parallel(self, args, benchmark, plc, num_workers):
        import multiprocessing as mp
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            ctx = mp.get_context("spawn")

        with ctx.Pool(num_workers) as pool:
            futures = []
            for start_idx in range(args["num_starts"]):
                futures.append(pool.apply_async(
                    _worker_run_start,
                    (start_idx, args, self.seed, self.refine_iters,
                     self.lns_outer_iters, self.lns_destroy_size, benchmark, plc),
                ))
            results = []
            for start_idx, fut in enumerate(futures):
                trial_pos, trial_cost = fut.get()
                results.append((trial_pos, trial_cost, start_idx))
        return results

    def _lns_loop(self, state, evaluator, plc, num_movable, bench_label="?", start_idx=0,
                  recorder=None):
        import time
        best_pos = state.current_positions()
        best_cost = evaluator.evaluate(best_pos)
        initial_cost = best_cost

        grid_rows = int(plc.grid_row)
        grid_cols = int(plc.grid_col)
        congested_percent = 0.05

        destroy_cap = int(os.environ.get("STRAPLE_DESTROY_CAP", "16"))
        destroy_pct = float(os.environ.get("STRAPLE_DESTROY_PCT", "0.025"))
        adaptive_destroy = max(self.lns_destroy_size,
                               min(destroy_cap, math.ceil(destroy_pct * num_movable)))
        outer_cap = int(os.environ.get("STRAPLE_LNS_OUTER_CAP", "50000"))
        outer_factor = float(os.environ.get("STRAPLE_LNS_OUTER_FACTOR", "60.0"))
        adaptive_outer = max(self.lns_outer_iters, min(outer_cap, math.ceil(outer_factor * num_movable)))

        accepted = 0
        accepted_random = 0
        accepted_congested = 0
        gain_random = 0.0
        gain_congested = 0.0
        log_step = max(1, adaptive_outer // 10) if self.verbose else 0

        accepted_swap = 0
        gain_swap = 0.0
        accepted_cluster = 0
        gain_cluster = 0.0
        no_improve_count = 0
        early_term_threshold = max(500, adaptive_outer // 5)

        ops = ["rand", "cong", "swap", "cluster"]
        op_weights = {o: 1.0 for o in ops}
        op_attempts = {o: 0 for o in ops}
        op_accepts = {o: 0 for o in ops}
        op_gains = {o: 0.0 for o in ops}
        rng = np.random.default_rng(self.seed + start_idx * 1000)
        warmup = min(40, adaptive_outer // 10)
        weight_decay = 0.95
        weight_reaction = 0.3

        shake_threshold = max(200, adaptive_outer // 20)
        shake_count = 0

        for iteration in range(adaptive_outer):
            saved = state.current_positions()
            t_iter = time.time()
            if iteration < warmup:
                op = ops[iteration % len(ops)]
            else:
                total_w = sum(op_weights.values())
                r = rng.random() * total_w
                acc = 0.0
                op = ops[-1]
                for o in ops:
                    acc += op_weights[o]
                    if r <= acc:
                        op = o
                        break
            op_attempts[op] += 1

            if op == "rand":
                trial = state.destroy_and_repair(adaptive_destroy)
            elif op == "cong":
                evaluator.evaluate(saved)
                hot_cells = evaluator.get_top_congested_cells(congested_percent)
                if hot_cells.shape[0] > 0:
                    cong_grid = evaluator.get_congestion_grid()
                    trial = state.destroy_congested_and_repair(
                        hot_cells, grid_rows, grid_cols, adaptive_destroy,
                        cong_grid, grid_rows, grid_cols,
                    )
                else:
                    trial = state.destroy_and_repair(adaptive_destroy)
            elif op == "swap":
                trial = state.swap_two_macros(max(2, adaptive_destroy // 2))
            else:
                trial = state.destroy_cluster_and_repair(adaptive_destroy)
            new_cost = evaluator.evaluate(trial)
            if recorder is not None:
                recorder.maybe_add_lns(trial, iteration, op,
                                       new_cost < best_cost, new_cost)
            if new_cost < best_cost:
                gain = best_cost - new_cost
                best_cost = new_cost
                best_pos = trial
                accepted += 1
                no_improve_count = 0
                op_accepts[op] += 1
                op_gains[op] += gain
                op_weights[op] = op_weights[op] * weight_decay + (
                    weight_reaction * (1.0 + 100.0 * gain))
                if op == "rand":
                    accepted_random += 1
                    gain_random += gain
                elif op == "swap":
                    accepted_swap += 1
                    gain_swap += gain
                elif op == "cluster":
                    accepted_cluster += 1
                    gain_cluster += gain
                else:
                    accepted_congested += 1
                    gain_congested += gain
                if log_step and (iteration % log_step == 0 or iteration == adaptive_outer - 1):
                    self._log(f"[{bench_label}] start#{start_idx} LNS iter={iteration:>3} "
                              f"op={op:<7} ACCEPT cost={new_cost:.4f} (-{gain:.4f}) "
                              f"k={adaptive_destroy} t={(time.time()-t_iter)*1000:.0f}ms")
            else:
                state.set_positions(saved)
                no_improve_count += 1
                op_weights[op] = op_weights[op] * weight_decay
                if op_weights[op] < 0.05:
                    op_weights[op] = 0.05
                if log_step and iteration % (log_step * 2) == 0:
                    self._log(f"[{bench_label}] start#{start_idx} LNS iter={iteration:>3} "
                              f"op={op:<7} reject (cost={new_cost:.4f} > {best_cost:.4f})")
                if no_improve_count >= shake_threshold and shake_count < 10:
                    state.set_positions(best_pos)
                    shake_k = min(num_movable // 4 + shake_count * 10, max(20, num_movable // 2))
                    state.swap_two_macros(shake_k)
                    state.destroy_and_repair(shake_k)
                    no_improve_count = 0
                    shake_count += 1
                    op_weights = {o: 1.0 for o in ops}
                    if self.verbose:
                        self._log(f"[{bench_label}] start#{start_idx} LNS SHAKE-UP @ iter={iteration} "
                                  f"k={shake_k} count={shake_count}")
                    continue
                if no_improve_count >= early_term_threshold:
                    if self.verbose:
                        self._log(f"[{bench_label}] start#{start_idx} LNS early term @ iter={iteration} "
                                  f"({no_improve_count} consec rejects)")
                    break

        state.set_positions(best_pos)
        if self.verbose:
            ops_summary = " | ".join(
                f"{o}: att={op_attempts[o]} acc={op_accepts[o]} "
                f"gain={op_gains[o]:.4f} w={op_weights[o]:.2f}"
                for o in ops
            )
            self._log(f"[{bench_label}] start#{start_idx} LNS ALNS: {ops_summary}")
            self._log(f"[{bench_label}] start#{start_idx} LNS summary: "
                      f"{accepted}/{adaptive_outer} accepted "
                      f"k={adaptive_destroy} initial={initial_cost:.4f} -> final={best_cost:.4f}")
        return {"iters": adaptive_outer, "accepted": accepted, "final_cost": best_cost}
