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
