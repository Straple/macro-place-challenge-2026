# Best placements per benchmark — Straple submission

Pickled placements that achieve the best proxy_cost we know how to produce
with the current pipeline. Each `.pkl` contains:

- `hard`: `np.ndarray [n_hard, 2]` — hard macro positions (μm)
- `soft`: `np.ndarray [n_soft, 2]` — soft macro positions (μm)
- `proxy`, `wl`, `dens`, `cong`: scalars from TILOS PlacementCost
- `config`: human-readable winning config description
- `bench`: benchmark name

## Reproduce

The pipeline that produced each pickle is wrapped in
`scripts/run_best.sh`. Run on a GPU host:

```bash
./run_remote.sh push
ssh <host> 'cd macro-place && bash scripts/run_best.sh ibm01'
```

After ~25 min the final placement appears at `results/snapshots/ibm01_dump.npz`,
last frame `post-triple-cycle`. Extract with:

```python
import numpy as np, pickle
d = np.load('results/snapshots/ibm01_dump.npz', allow_pickle=True)
final = d['frames_pos'][-1]
n_hard = int(d['n_hard'])
n_total = int(d['n_total'])
metrics = d['frames_metrics'][-1]
pickle.dump({
    'hard': final[:n_hard].astype('float64'),
    'soft': final[n_hard:n_total].astype('float64'),
    'proxy': float(metrics[0]), 'wl': float(metrics[1]),
    'dens': float(metrics[2]), 'cong': float(metrics[3]),
    'config': 'L-BFGS finisher seed=43 (run_best.sh)',
    'bench': 'ibm01',
}, open('submissions/straple/best_placements/ibm01_X.pkl', 'wb'))
```

## Current best (ibm01)

| file | proxy | WL | dens | cong | config |
|---|---|---|---|---|---|
| ibm01_lbfgs_0.8865.pkl | **0.8865** | 0.0727 | 0.5540 | 1.0737 | L-BFGS finisher seed=43 |

- Challenge target proxy < 1.4578 → margin ≈ 39%
- Beats prior best Round 23 lucky 0.8856 by −0.001
- A prior single-shot of the same config gave 0.8785 (legalize stochasticity).
  Re-running may produce 0.87-0.89 depending on which seed wins all-K
  legalize and CD/pair-swap convergence path.

## AVG17 partial sweep (2026-05-10 / 2026-05-11)

Ran `scripts/run_best.sh` (L-BFGS finisher + Round 23 polish stack)
across IBM benchmarks in ICCAD04. K=384 for ibm01 (full memory budget),
K=64 for ibm02+ (avoid OOM on bigger benches). Snapshot dumps disabled
for ibm08+ to avoid silent OOM during TILOS proxy evaluation. Some big
benches (ibm06–14) hit WALL_TL during eval/CD and stopped at
post-legalize. ibm15–18 not completed (eval+legalize took 1+ hour each
for the biggest benches, batch killed for time budget).

| bench | proxy | stage | pkl/dump |
|---|---|---|---|
| ibm01 | **0.8882** | post-triple-cycle | dump |
| ibm02 | 1.4182 | post-triple-cycle | pkl |
| ibm03 | 1.2283 | post-triple-cycle | dump |
| ibm04 | 1.1687 | post-triple-cycle | dump |
| ibm06 | 1.5940 | post-legalize | dump |
| ibm07 | 1.2385 | post-legalize | dump |
| ibm08 | 1.4124 | post-legalize | pkl |
| ibm09 | 1.0137 | post-legalize | pkl |
| ibm10 | 1.3357 | post-legalize | pkl |
| ibm11 | 1.0649 | post-legalize | pkl |
| ibm12 | 1.5753 | post-legalize | pkl |
| ibm13 | 1.1669 | post-legalize | pkl |
| ibm14 | 1.4796 | post-gradient | pkl |
| ibm15 | — | not run | — |
| ibm16 | — | not run | — |
| ibm17 | — | not run | — |
| ibm18 | — | not run | — |

- **Average over 13 done benches: 1.2757** (challenge target ≤ 1.4578 → margin 12.5%)
- 13/13 done benches under target individually.
- All post-legalize results would improve ~0.01-0.03 by CD/pair-swap/triple polish (skipped due to wall-time).

Per-bench visualizations:
- `vis/ibm01_snapshots.html`, `vis/ibm03_snapshots.html`, … — interactive 4-panel HTML with cluster colors (only for benches that completed snapshot dump)
- `vis/<bench>_placement.png` — static placement PNG
- `vis/avg17_summary.png` — 4×4 grid showing all 13 placements at once

## Notes

- L-BFGS finisher (`STRAPLE_BATCH_LBFGS_FROM_STEP=1000`) is the only
  clean-win lever from the 2026-05-09/10/11 exploration round. All
  congestion-targeted mechanisms (cluster split, OVERLAP_SOFT, fanout
  weighting, star net, cong-aware inflate, L-route loss bbox-center
  proxy, long-net collapse / teleport / smart-teleport) yielded either
  noise or LOSE — see `readme/improve.md` v5+v6 and `readme/hessian.md`
  for the full record.
- The dominant cong source is **long multi-pin inter-cluster nets**
  (per `scripts/diagnose_lroute.py`), not clique-clusters (which RUDY
  attribution incorrectly suggested). Tight clusters minimize TILOS
  L-route demand; any physical spread of long-net pin macros increases
  cong because their L-routes lengthen.
