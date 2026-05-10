# Improve.md v4 — FINAL action plan от 0.8856 → ≤0.85

> Прошёл 4 итерации с 12 агентами:
> - v1: 6 research agents (Hessian, Newton/quasi-Newton, Approximations, Critical, Synthesis, Web/papers)
> - v2: 3 reviewers (math, impl, missing ideas)
> - v3: 3 reviewers (rigor, execution, contrarian)
> - v4: 4 reviewers (verification, protocols, devil's advocate, final synthesis)
>
> **One-line pitch:** Diagnostics-first (Day 1-2) → DREAMPlace black-box paradigm shift (Day 3-4) → joint loss + L-BFGS (Day 5-7) → Bayesian HPO overnight. Realistic target **0.85-0.87**, stretch **0.79-0.82**.

---

## ⭐ Executive Summary — что и почему

**Текущий best:** 0.8856 (Round 23, lucky single-shot). **Median pipeline:** ~0.890-0.895.

**Кальибрированные probabilities (devil's advocate honest):**
- P(reach 0.85 in 1 week) ≈ **0.20-0.35**
- P(reach 0.80 in 2 weeks) ≈ **0.05-0.15**
- P(zero improvement) ≈ **0.20-0.25**

**5 Actions ranked by `expected_value × probability_success`:**

| # | Action | Confidence | Time | Realistic Δ | Stretch Δ |
|---|---|---|---|---|---|
| 1 | **H2 + S1 diagnostics** (Phase 0) | 9/10 | 5-7h | 0 (gates) | 0 |
| 2 | **DREAMPlace M3 black-box** | 7/10 | 12-18h | -0.02 | -0.06 |
| 3 | **M1 Joint p=4 cong+density loss** | 6/10 | 4-6h | -0.005 | -0.015 |
| 4 | **L-BFGS late-stage finisher** | 7/10 (down from 8/10) | 8-12h | -0.005 | -0.010 |
| 5 | **MOTPE Bayesian HPO overnight** | 7/10 (down from 8/10) | 8-12h+12h GPU | -0.008 | -0.015 |

**Cumulative pessimistic 0.881 / realistic 0.866 / stretch 0.795** (from 0.8856).

---

## 🔬 Empirical insights from execution (v5, 2026-05-09/10)

### Action #1 H2 result — cong is gradient-bound
Single-shot trajectory on ibm01 (trial9 + CD + pair-swap rounds=8 + triple-cycle):
- post-gradient `cong_frac = 0.609`, post-triple-cycle `cong_frac = 0.596`
- Cong drops 1.1116 → 1.0655 (−0.046 abs) across full pipeline; pair-swap+triple contribute only −0.005 (~10% of total cong reduction). **Polish operators are essentially noop on cong.**
- Cong-floor is set by gradient phase. WL is solved by gradient (~0.071 stable). Density spikes post-legalize (+0.024) then CD reverses partially.
- **→ Decision rule per v4 (`cong_fraction ≥ 0.55`): SKIP L-BFGS / Hessian / any pure gradient finisher.** They optimize WL/density convergence, not cong topology.

### Action #3 M1 joint p-norm — incompatible with dynamic λ_d (FAIL)
- Plan's `cong_norm = cong / cong.detach()` formula: loss numerically `(cw^p + dw^p)^(1/p)` (constant), gradient scales as `1/x.detach()`.
- For density (scale ~3000) the gradient gets divided by ~3000 → killed.
- Density not optimized → overflow stays high → λ_d explodes via overflow update → `dw^p` (e.g. `2000^4 = 1.6e13`) dominates joint_pen → cong gradient also collapses to ~0.
- Both p=4 and p=2 hit identical pathology at step 100–200 (dpen=260k vs baseline 17k = 15× explosion).
- **Don't reimplement plan's exact formula.** If knee-seeking is revisited:
  - fixed-weight joint (don't put dynamic λ_d inside p-norm); keep additive λ_d·dpen separately for overflow control;
  - target-relative ratios `(cong/cong_target)^p + (dens/dens_target)^p` (no detach);
  - apply joint_pen only late P3 after λ_d settled;
  - or constraint formulation (Lagrangian on cong < target ∧ dens < target).

### Cong-only ablation — density is for physical spread, not proxy weight
Disabled cohesion / anchor / ePlace density / overflow-λ / blockage; kept only WL + overlap (hard) + cong. ibm01 mini run (K=64, 60s gradient + light CD/pair/triple):

| metric           | baseline trial9 | cong-only | Δ |
|---               |---              |---        |---|
| proxy (final)    | **0.9219**      | 0.9691    | **+5.1% ⚠**  |
| WL               | 0.0749          | **0.0716**| −4.4% ✓ |
| density          | **0.5848**      | 0.7069    | **+20.9% ⚠** |
| congestion       | 1.1093          | **1.0881**| −1.9% ✓ |
| post-grad ovl    | 125             | 84        | −33% ✓ |

Surprises:
- **WL is hurt by cohesion** (−4% when removed). Cohesion makes clusters tight, but WL bbox-LSE finds its own optimum without help.
- **Cong loss alone DOES reduce cong** (−2%) — direct gradient signal works. Not the bottleneck.
- **Density penalty's job is physical spread, not proxy improvement.** Without it, layout drifts into denser packing under WL pull → density component (0.5·dens) of proxy alone adds +0.06 → proxy worse despite WL+cong wins.
- **Layout doesn't collapse** (overlap penalty for hard macros holds it). Earlier prediction was wrong.

→ **Knee idea: smarter density, not joint p-norm.**
- `STRAPLE_BATCH_DENSITY_TOPK_W>0` + `DENSITY_TOPK_PCT=0.05` already exists — penalizes only top-5% peak density cells. Allows dense zones where cong is fine, slams only true hotspots. Untried with cong-only style minimal loss.
- Soft-overlap penalty `STRAPLE_BATCH_OVERLAP_SOFT=1` is the documented spread-fix for soft-cluster pile-up (memory: −5% cong on ibm01 vs baseline). Should be default.

### Action #5 HPO partial result (killed before completion)
Trial 5 (trial9 + `OVERFLOW_TARGET=0.10` + `PAIR_SWAP_ROUNDS=12`, K=384) → 0.8910 single-shot. Within noise of Round 23 0.8856 lucky. TPE was learning around trial9 region but user killed for visualization detour. If revived: re-seed study with trial9 anchor + slightly tighter overflow + 12-round pair-swap as a productive starting basin.

### Visualization infrastructure
Added `submissions/straple/snapshot_dump.py` (env `STRAPLE_BATCH_DUMP_SNAPSHOTS=1`) — captures per-frame pos + TILOS metrics + density/cong grids + Louvain cluster_ids into `.npz`. Standalone renderer `scripts/render_snapshots.py` builds 4-panel HTML (placement + density turbo + congestion turbo + metrics) from raw arrays — re-rendering is free (no pipeline rerun).

---

## 🧪 Round of cong-targeted experiments (2026-05-10) — all LOSE/noise

After L-route diagnose (`scripts/diagnose_lroute.py`) revealed long inter-cluster nets dominate TILOS cong (NOT cluster cliques per RUDY), tried four cong-targeted mechanisms on ibm01 (K=64, 60s gradient):

| config | proxy | WL | dens | cong | verdict |
|---|---|---|---|---|---|
| baseline | **0.9219** | 0.0749 | 0.5848 | 1.1093 | reference |
| net alpha=0.5 (fanout up-weight) | 0.9288 | 0.0745 | 0.5863 | 1.1224 | noise (cong +1.2%) |
| star_net_thr=4 | 0.9227 | 0.0803 | 0.5914 | 1.0933 | noise (cong −1.4%, WL +7%) |
| stack alpha=0.5 + star_thr=4 | 1.020 | +10% | +18% | +6.8% | LOSE catastrophic |
| cong_inflate W=1.0 | overflow | — | — | — | crash |
| cong_inflate W=0.2 | 1.543 | +127% | +35% | +76% | LOSE catastrophic (positive feedback) |
| L-route loss W=2.0 (bbox-center proxy) | 0.9416 | +3.8% | −3.9% | +5.1% | LOSE (cong WORSE) |

**Mechanistic conclusions:**
- **Net-degree weighting (FastPlace 1/(k-1) или fanout up-weight)**: noise on bbox-HPWL because bbox already linear (not quadratic clique decomp). Effect cancels against density.
- **Star-net for k≥4**: trades WL up for cong down but noise net. Removes bbox-extreme pin domination but doesn't shorten long routes.
- **Cong inflate (RWCI-style)**: positive feedback loop → catastrophic dispersion. Cong source is **net topology**, not macro density; inflation fights wrong battle.
- **L-route bbox-center proxy**: fundamentally wrong because TILOS L-route requires per-edge **driver row + sink col**, not net-bbox center. Approximation pushes macros away from net midpoints → longer nets → worse cong.

**What's still on the table:**
- True per-edge L-route loss (~6h refactor to load `edges_pkg` into gradient_batch and scatter per-edge demand chunked over E~6000 edges).
- Adam → L-BFGS hybrid late stage (Action #4, was deprioritized after H2 cong-bound finding; now reconsider as the only remaining gradient-finisher angle).
- Multi-objective HPO over Round 23 best (Action #5 — partial trial 5 = 0.8910 found before HPO killed; can resume).

**Hard floor without paradigm change:** ~0.92 single-shot, ~0.89 best across luck.

---

## 🚫 KILL LIST (не делать — low ROI / already debunked)

1. **Tier 1.1 Per-macro 2×2 Newton CD** — paradigm risk (v1-v7 saddle history showed 768 escapes all in noise); density Hessian derivation 12-18h; gated by S2 which is itself inconclusive at -0.002 zone.
2. **Tier 1.3 SFN / Cubic Newton** — same paradigm mismatch; σ ≈ 0.01 noise dominates curvature signal.
3. **Tier 2.3 Hall spectral init** — Round 31 already failed; "schedule recalibration" hopeware.
4. **Tier 2.4 BB step** — NOT Adam drop-in (Round 27); separate optimizer track explodes (Round 11).
5. **A2 SA-from-scratch** — Round 23 full-stack 0.8856 vs random ≫ 0.95; gradient phase essential, no need to re-verify.

---

## ⚠️ Devil's advocate adjustments (Important!)

**v3 over-optimistic in:**
- L-BFGS 8/10 → **7/10:** legalize discontinuity (snap-to-grid jump) corrupts curvature pairs. Real use case = pre-legalize gradient phase only, не post-legalize.
- M5 Bayesian HPO 8/10 → **7/10:** N=17 paired per trial impossible at single T4 (would need 50+h per HPO trial). Adjusted protocol: N=1 per trial, top-3 verified with N=5.

**v3 under-rated:**
- Tier 2.5 SA-style CD acceptance 7/10 → **8/10:** cheapest, orthogonal, low risk. **Add as Day 1 sidecar.**

**S1 access problem (CRITICAL):**
- "Known-good external placement for ibm01" может быть **недоступен** в protobuf format. Catch-22: чтобы запустить S1 sanity check нужен Bookshelf converter, который сам = 1-2 дня (Action #2 dependency).
- **Fallback:** если нет access к vmallela's placement → запустить **DREAMPlace директно** (Action #2 first) → use ITS output как known-good для S1.
- → Order changes: **Action #1 H2 logging first (2h independent), then Action #2 partial (just install + run DREAMPlace), then S1 уsing DREAMPlace output.**

**Compute budget reality (devil's advocate):**
- v3 claimed 13h Phase 0 GPU + 56h Phase 1 (4 ideas × N=17 paired). Single T4 = 7h/day continuous.
- Realistic: **N=5-8 paired** (not 17), use Wilcoxon rank-sum test (more robust to non-normality). Detection threshold 0.005 abs.

---

## Action #1: H2 Component Breakdown Logging + S1 Loss-Floor Probe

### ЧТО
Залогировать WL/density/cong на 5 точках pipeline + прогнать наш proxy на known-good external IBM01 placement (DREAMPlace public output).

### ПОЧЕМУ — High confidence
- **Mechanism:** Round 18 showed cong=61% post-pipeline. H2 даёт **trajectory** через 5 stages → видно где cong "застывает". Если cong=0.95 уже после gradient → Newton/L-BFGS на gradient phase wasted (cong не двигается там). Если cong падает с 1.20→1.05 в pair-swap → есть hook для discrete moves.
- **Evidence:** Rounds 19-20 showed Pareto cong↔density (cong↓ ⇒ dens↑). Без trajectory не знаем **откуда** frontier.
- **Why others не addressэт:** все Tier 1 ideas (Newton/L-BFGS/AL) работают на gradient phase. Если cong dominates **post-CD** — Newton wasted.
- **Theoretical backing:** Bishop-style component analysis. Самая дешёвая, самая высокоinformative diagnostic.

### КАК
1. **Создать** `submissions/straple/breakdown_log.py` (~60 lines):
   ```python
   class BreakdownLogger:
       def __init__(self, benchmark, plc, run_id):
           self.b, self.p, self.run_id = benchmark, plc, run_id
       def log(self, pos, stage, seed=None):
           cost = compute_proxy_cost(pos, self.b, self.p)
           print(f"[BREAKDOWN stage={stage} seed={seed} "
                 f"wl={cost['wirelength_cost']:.4f} "
                 f"dens={cost['density_cost']:.4f} "
                 f"cong={cost['congestion_cost']:.4f} "
                 f"proxy={cost['proxy_cost']:.4f} "
                 f"ovl_n={cost['overlap_count']}]", flush=True)
   ```
2. **Wire в 5 sites:**
   - `gradient_batch.py` ~line 970 (post-gradient best seed)
   - `placer.py:861` (post-legalize)
   - `cd_polish.py:139` (post-CD)
   - `cd_polish.py:539` (post-pair-swap)
   - `cd_polish.py:872` (post-triple-cycle)
3. **Env:** `STRAPLE_BATCH_BREAKDOWN_LOG=1` (default off).
4. **S1 fallback (no external placement available):** если нет vmallela / DREAMPlace public reference, запустить DREAMPlace ourselves (overlap with Action #2 install) → use its output как S1 reference.
5. **Test:** 1 baseline trial9 run (~25 min) с logging + S1 на DREAMPlace output.

### Done criterion
- Breakdown log на all 5 stages, valid floats.
- Decision rule (refined):
  - `cong_fraction(post-gradient) ≥ 0.55` → **SKIP** Hessian/L-BFGS, GO Action #2 + #3 only.
  - `S1 proxy on reference ≈ 0.76` → continuous path viable, Action #4 (L-BFGS) worthwhile.
  - `S1 proxy ≥ 0.88` → proxy formulation **hard ceiling**; abort optimization track, GO Action #5 only.

### Risks
- **No external placement access** (high P) → run DREAMPlace ourselves for S1.
- **TILOS overhead 10s** → negligible.

### Time: 5-7h. **Confidence: 9/10** (diagnostics не fail).

---

## Action #2: DREAMPlace Black-Box Integration (M3)

### ЧТО
Конвертировать ibm01 protobuf → Bookshelf, запустить DREAMPlace, использовать его placement как input для нашего CD+pair-swap+triple stack.

### ПОЧЕМУ — High expected value (binary outcome)
- **Mechanism:** наш Adam batch K=384 — generic optimizer без macro-specific tricks (Lipschitz adaptive Nesterov, line search, multi-grid density). DREAMPlace = 8+ years placement-specific engineering. Round 27 показал что custom Nesterov у нас не работает без full DREAMPlace stack — нет смысла reinvent.
- **Evidence:** UT Austin paper claims DREAMPlace AVG17 ≈ 1.41 (мы на ~1.0 = +40% хуже). Vmallela (1st place) likely uses DREAMPlace + custom CD. Round 12 showed time_budget extension hurts без adaptive optimizer.
- **Why others не address:** Tier 1 Newton/SFN — micro-optimization within broken framework. M3 = **paradigm replacement**.
- **Theoretical backing:** Chen ICCAD 2023 explicit DREAMPlace + L-BFGS. State-of-art baseline.

### КАК
1. **Bookshelf converter:** `scripts/proto_to_bookshelf.py` (~150 lines): protobuf netlist + sizes → .nodes/.nets/.pl/.scl/.wts/.aux. Validate via WL round-trip test.
2. **DREAMPlace install:** `git clone limbo018/DREAMPlace`, build CUDA extension в Docker (avoid host CUDA conflicts on T4).
3. **Wrapper** `scripts/dreamplace_run.py`: protobuf → convert → run DREAMPlace → read .pl → convert back.
4. **Integration в pipeline:** `STRAPLE_BATCH_DREAMPLACE=1` в `gpu_run_one.py`. Если set → SKIP gradient batch, используем DREAMPlace pos как single best seed → applies legalize → CD → pair-swap → triple.
5. **Test:** ibm01 single-run 30 min (DREAMPlace ~10 min + наш pipeline 15 min + buffer).

### Done criterion
- DREAMPlace runs без crash на ibm01.
- Returns valid .pl (legalize succeeds, no overlaps).
- Final proxy after CD+pair-swap stack: **single run target ≤ 0.86, paired N=5 median ≤ 0.87.**
- If DREAMPlace alone gives < 0.88 без CD → NEW BEST, integrate as default.

### Risks
- **Bookshelf format mismatch** (soft macros, blockages) → ibm01 has 246 hard, 894 soft; treat soft as standard cells, validate WL preservation round-trip.
- **CUDA version conflicts** на T4 → use Docker DREAMPlace official image; CPU build fallback.

### Time: 12-18h (1.5-2 days). **Confidence: 7/10** (binary).

---

## Action #3: M1 Joint p=4 Cong+Density Loss

### ЧТО
Заменить additive `0.5·cong + 0.5·dens` на Chebyshev p-norm `((cw·cong)^p + (dw·dens)^p)^(1/p)` с p=4 в gradient loss.

### ПОЧЕМУ — Targets ROOT CAUSE
- **Mechanism:** Round 19 (cong_w=20) и Round 20 (cong_w=15+top_pct=0.05) обе хитнули **симметричный Pareto frontier** (cong↓ ⇒ dens↑ symmetrically). Additive loss `cong+dens` имеет линейные level-curves → optimizer движется ВДОЛЬ frontier (trade-off). p=4 имеет L_∞-like level-curves → **knee-seeking** → finds Pareto knee corner.
- **Evidence:** Round 18 breakdown: post-pair-swap cong=1.05, dens=0.58, WL=0.07. Knee target — где cong≈dens≈0.7 → proxy ≈ 0.07 + 0.5·0.7 + 0.5·0.7 = **0.77** (близко к vmallela 0.7644).
- **Why others не address:** все optimizer changes (Newton/L-BFGS/AL) работают на ту же loss → движутся по тому же frontier. M1 = **loss reformulation** → меняет geometry, не optimizer.
- **Theoretical backing:** Chebyshev scalarization (Miettinen 1999, Multiobjective Optimization). p=4 — empirically close to L_∞ smooth gradient.

### КАК
**Edit `submissions/straple/gradient_batch.py:867`:**
```python
joint_p = float(os.environ.get("STRAPLE_BATCH_JOINT_LOSS_P", "0"))
if joint_p > 0:
    eps = 1e-9
    cong_norm = cong_total / cong_total.detach().clamp_min(eps)
    dens_norm = dpen_total / dpen_total.detach().clamp_min(eps)
    joint_pen = ((cong_weight * cong_norm)**joint_p +
                 (density_weight * dens_norm)**joint_p)**(1.0/joint_p)
    loss = (wl_total + cur_overlap_w_phase * overlap_total
            + anchor_loss_total + cohesion_loss_total
            + topk_density_weight * density_topk_total
            + joint_pen)
else:
    loss = ... (original additive)
```

**Env:** `STRAPLE_BATCH_JOINT_LOSS_P=4` (default 0 = backwards compat).

**Unit test:** `tests/test_joint_loss.py` — gradcheck synthetic 4-macro.

**Test:** paired N=5 baseline vs p=4 (если v3 budget reality cuts N=17→5).

### Done criterion
- Unit test passes.
- Run без NaN through 500 steps.
- Pre-CD min ≤ 0.90 (vs typical 0.91) on single run.
- Paired N=5: median Δ ≤ -0.005.
- Breakdown: cong<1.00 AND dens<0.70 simultaneously (knee found).
- Если pre-CD не двигается ≤ 0.91 в 3 runs → drop p=4, try p=2 (smoother).

### Risks
- **Gradient instability при p=4** — clamp_min(1e-9), `torch.isfinite` early-abort.
- **Frontier shifts to different corner** — log breakdown per-step, abort если cong > 1.3 at step=200.

### Time: 4-6h. **Confidence: 6/10** (theoretically clean, empirical risk).

---

## Action #4: L-BFGS Late-Stage Finisher

### ЧТО
После Adam (step 1000), переключиться на batched L-BFGS (m=10, Wolfe LS) для super-linear convergence.

### ПОЧЕМУ — Highest math grounding
- **Mechanism:** Adam — 1st-order, scale-invariant per-coord. К концу gradient phase (P3 settling) landscape почти-quadratic. L-BFGS approximates inverse Hessian через secant updates → **локально quadratic convergence**. Round 12 анализ: late steps Adam wastes — gradient noise dominates.
- **Evidence:** Chen ICCAD 2023 Adam-only vs Adam→L-BFGS: -3-7% wirelength on standard benchmarks. DREAMPlace defaults к L-BFGS finisher. Round 30 showed extending pair-swap не пробивает -0.005 floor — final improvement должен идти от gradient phase.
- **Why others не address:** Per-macro Newton (1.1) overkill 12-18h. SFN (1.3) saddle-specific (paradigm risk). L-BFGS — proven, batched, no Hessian computation.
- **Theoretical backing:** Liu-Nocedal 1989 globally convergent с Wolfe LS, locally super-linear.

**⚠️ Devil's advocate concern (downgraded confidence):**
- Legalize discontinuity (snap-to-grid jump) corrupts curvature pairs `s_k = pos_k - pos_{k-1}`. L-BFGS works ONLY pre-legalize gradient phase, не post-legalize.
- Mitigation: drop history (`size=0`) at phase boundaries (overlap_w_phase transitions, schedule resets).

### КАК
**Новый файл** `submissions/straple/lbfgs_finisher.py` (~150 lines):
```python
class BatchedLBFGS:
    def __init__(self, K, n_active, m=10, device, dtype):
        self.s = torch.zeros(m, K, n_active, 2, device=device, dtype=dtype)
        self.y = torch.zeros_like(self.s)
        self.rho = torch.zeros(m, K, device=device, dtype=dtype)
        self.head = 0; self.size = 0; self.m = m

    def two_loop(self, g):  # standard Liu-Nocedal Algo 7.4
        q = g.clone()
        alphas = []
        for i in range(self.size):
            idx = (self.head - 1 - i) % self.m
            a = self.rho[idx] * (self.s[idx]*q).sum(dim=(1,2))
            q = q - a[:, None, None] * self.y[idx]
            alphas.append((idx, a))
        # H_0 scaling: γ_k = (s_k·y_k)/(y_k·y_k)
        if self.size > 0:
            last = (self.head - 1) % self.m
            sy = (self.s[last]*self.y[last]).sum(dim=(1,2))
            yy = (self.y[last]*self.y[last]).sum(dim=(1,2)).clamp_min(1e-12)
            gamma = (sy / yy).clamp_min(1e-6)
            r = gamma[:, None, None] * q
        else:
            r = q
        for idx, a in reversed(alphas):
            b = self.rho[idx] * (self.y[idx]*r).sum(dim=(1,2))
            r = r + (a - b)[:, None, None] * self.s[idx]
        return -r  # search direction p

    def push(self, s_new, y_new):
        sy = (s_new*y_new).sum(dim=(1,2))
        # Powell damping
        valid = sy > 1e-10
        # ... (apply θ blend where !valid)
        self.s[self.head] = s_new
        self.y[self.head] = y_new
        self.rho[self.head] = 1.0 / sy.clamp_min(1e-12)
        self.head = (self.head + 1) % self.m
        self.size = min(self.size + 1, self.m)
```

**Wire в gradient_batch.py ~line 700:**
```python
lbfgs_from_step = int(os.environ.get("STRAPLE_BATCH_LBFGS_FROM_STEP", "0"))
if lbfgs_from_step > 0 and step >= lbfgs_from_step:
    p = lbfgs.two_loop(grad)
    # Wolfe LS, max 4 evals
    alpha = wolfe_line_search(loss_fn, pos, p, grad, c1=1e-4, c2=0.9, max_evals=4)
    pos_new = pos + alpha * p
    grad_new = compute_grad(pos_new)
    lbfgs.push(pos_new - pos, grad_new - grad)
    pos = pos_new
else:
    optimizer.step()  # Adam
```

**Trigger:** `STRAPLE_BATCH_LBFGS_FROM_STEP=1000` (default 0).

### Done criterion
- No NaN/inf через 1200 steps.
- Memory peak < 11 GB.
- Pre-CD min ≤ 0.90 на 5/5 runs (paired).
- Paired N=5 median Δ ≤ -0.003.
- Если LS rejects > 50% steps → curvature corruption → revert.

### Risks
- **Curvature corruption near phase boundaries** → drop history at transitions.
- **LS wall-time blowup** → max 4 evals + parallel K amortizes.

### Time: 8-12h. **Confidence: 7/10** (down from 8/10 due to legalize concern).

---

## Action #5: MOTPE Bayesian HPO Overnight

### ЧТО
Optuna MOTPE sweep 25-30 trials over 7 hyperparams Round 23 best pipeline, optimizing (proxy_min, std) Pareto front overnight.

### ПОЧЕМУ — Auto-tuning works on any pipeline
- **Mechanism:** Rounds 14-30 showed individual hyperparam tweaks (cong_w, top_pct, time_budget, K) каждый меняет distribution на Pareto. Manual tuning slow и hits floors. MOTPE — Bayesian — exploits correlations, auto-prunes bad trials.
- **Evidence:** Round 28 replay 0.8955 showed σ ~0.005-0.01 between runs same config. HPO reduces variance + finds better median.
- **Why others не address:** Actions 1-4 — single-shot improvements. Action 5 — **stochastic exploitation**. Cheap (overnight GPU).
- **Theoretical backing:** MOTPE [Watanabe-Tsuruta 2024] dominates random search 5-10x.

**⚠️ Devil's advocate concern:**
- Single-trial → лучший по N=1 trial может быть lucky outlier. Mitigation: top-3 verified с N=5 paired.

### КАК
**Новый script** `scripts/hpo_motpe.py`:
```python
import optuna
sampler = optuna.samplers.MOTPESampler(seed=42)
study = optuna.create_study(directions=["minimize", "minimize"],
                             sampler=sampler,
                             pruner=optuna.pruners.MedianPruner())

def objective(trial):
    overlap_w_max = trial.suggest_float("OVERLAP_W_MAX", 20000, 100000, log=True)
    overlap_w_growth = trial.suggest_float("OVERLAP_W_GROWTH", 1.002, 1.010, log=True)
    overflow_target = trial.suggest_float("OVERFLOW_TARGET", 0.08, 0.20)
    overflow_exp = trial.suggest_float("OVERFLOW_EXP", 0.5, 1.0)
    cong_w = trial.suggest_float("CONG_W", 5, 25)
    pair_swap_rounds = trial.suggest_categorical("PAIR_SWAP_ROUNDS", [6, 8, 10, 12])
    K = trial.suggest_categorical("K", [256, 384, 512])
    
    proxy, std = run_gpu_run_one(
        OVERLAP_W_MAX=overlap_w_max, OVERLAP_W_GROWTH=overlap_w_growth,
        OVERFLOW_TARGET=overflow_target, OVERFLOW_EXP=overflow_exp,
        CONG_W=cong_w, PAIR_SWAP_ROUNDS=pair_swap_rounds, K=K,
        time_budget=1200, bench="ibm01"
    )
    return proxy, std  # multi-obj

study.optimize(objective, n_trials=25, timeout=12*3600)  # overnight 12h
```

**Output:** Pareto front config dump → pick best by `proxy + 0.5·std`.

### Done criterion
- 25-30 trials complete.
- Best HPO config `proxy_min ≤ 0.880` paired N=5 (vs Round 23 0.8856 ± 0.01 noise).
- Repeated best config 5x: std ≤ 0.005 (down from 0.01).
- Если best HPO config no improvement (paired N=5 median Δ < -0.003) → confirm Pareto floor, escalate to Action #2 retry.

### Risks
- **HPO finds overfit config** (lucky single trial) → top-3 retest N=5; pick by median.
- **GPU contention** → dedicated server window.

### Time: 8-12h coding + 12h overnight GPU. **Confidence: 7/10** (down from 8/10 due to single-trial verification concern).

---

## 🔀 Execution Flow Diagram

```
START: Day 0
  │
  ▼
[Action #1: H2 + S1 diagnostics]  (5-7h, Day 1-2)
  │
  ├─ S1 proxy ≈ 0.76 ──────────────► [continuous path viable]
  ├─ S1 proxy ≥ 0.85 ──────────────► [proxy ceiling, SKIP Action #2 + #4, GO Action #5 only]
  └─ cong_fraction ≥ 0.55 ─────────► [SKIP Hessian, GO Action #2 + #3]
  │
  ▼ (most likely branch)
[Action #2: DREAMPlace M3]  (12-18h, Day 3-4)
  │
  ├─ ibm01 < 0.88 (NEW BEST) ─────► [refine с our CD+pair-swap → run Action #5 over DREAMPlace baseline]
  └─ ibm01 ≥ 0.89 (no help) ──────► [continue to Action #3]
  │
  ▼
[Action #3: M1 joint loss p=4]  (4-6h, Day 5)
  │
  ├─ pre-CD min ≤ 0.90 paired N=5 ─► [stack with current pipeline → Action #4]
  └─ no improvement OR breakdown anomaly ─► [revert, GO Action #4 standalone]
  │
  ▼
[Action #4: L-BFGS finisher]  (8-12h, Day 6-7)
  │
  ├─ paired N=5 median Δ ≤ -0.003 ──► [stack with M1 if M1 won]
  └─ no improvement ──────────────► [Adam was optimal, GO Action #5]
  │
  ▼
[Action #5: MOTPE HPO overnight]  (8-12h coding + 12h GPU, Day 8-10)
  │
  ├─ best config ibm01 ≤ 0.84 ─────► STOP: GOAL REACHED
  ├─ ibm01 ∈ [0.85, 0.87] ─────────► [accept, submit, schedule Week 2 paradigm shifts]
  └─ ibm01 ≥ 0.88 ────────────────► PROVISIONAL DEFEAT: pivot to ML/RL (1-2 month effort)
```

### Decision triggers between actions
- **After Action #1:** breakdown reveals dominant component → routes to optimization (Adam/L-BFGS) vs formulation (M1/M3) vs paradigm (RL).
- **After Action #2:** binary works/doesn't. Если works — supersedes gradient phase, остальные actions stack on top.
- **After Action #3:** check breakdown again — knee-seeking moved cong&dens both down? Если yes — keep p=4, stack #4. Если no — revert.
- **After Action #4:** L-BFGS adds late-stage convergence; combine с M1 если won для cumulative.
- **After Action #5:** automated final tuning over best stack from #2-#4.

### Stop conditions
- **Success:** ibm01 ≤ 0.85 paired N≥5 median, std ≤ 0.005 → submit, lock config, move to AVG17 verification.
- **Provisional defeat:** после всех 5 actions ibm01 > 0.88 → escalate (M4 RL, 1-2 month) или accept current 0.8856 as session ceiling.
- **Mid-loop pivot:** если Action #1 S1 показывает proxy ≥ 0.90 на reference placement → ceiling fundamental, skip optimization actions, focus only HPO over Round 23 + submit.

---

## Cumulative expected gain (pessimistic / realistic / stretch)

| Action | Pessimistic Δ | Realistic Δ | Stretch Δ |
|---|---|---|---|
| #1 H2+S1 (diagnostic) | 0 | 0 (gates next) | 0 |
| #2 DREAMPlace M3 | 0 (fail) | -0.02 | -0.06 |
| #3 M1 joint p=4 | 0 | -0.005 | -0.015 |
| #4 L-BFGS | -0.001 | -0.005 | -0.010 |
| #5 MOTPE HPO | -0.003 | -0.008 | -0.015 |
| **Total from 0.8856** | **0.881** | **0.866** | **0.795** |

**Realistic target after Week 2: ibm01 ≈ 0.85-0.87.** **Stretch: 0.79-0.82** (Action #2 succeeds + #5 finds knee).

---

## Day-by-day execution (Week 1)

### Day 1 (Mon): H2 logging + Action #2 install kickoff
- Wire BreakdownLogger в 5 sites (~3h)
- Run baseline trial9 with logging (25 min)
- Start DREAMPlace install в Docker on T4 (~2h, possibly continues Day 2)
- **Add SA-CD acceptance** as cheap experiment (~2h, Tier 2.5 promoted from devil's advocate)
- Commit `diag-h2`

### Day 2 (Tue): S1 + Bookshelf converter
- Code Bookshelf converter (~3h)
- Run DREAMPlace on ibm01 → get `.pl` output (~2h)
- Run S1 (compute proxy on DREAMPlace output) (~30 min)
- **Decision point** based on H2+S1 results
- Commit `diag-h1, diag-s1, feat-bookshelf-converter`

### Day 3 (Wed): Action #2 wrapper integration
- Code dreamplace_run.py wrapper (~3h)
- Integration в gpu_run_one.py (~2h)
- Test full pipeline ibm01 with DREAMPlace input (~30 min)
- **If WIN (≤ 0.86):** commit, run paired N=5 verification (overnight)
- **If LOSE:** continue Action #3

### Day 4 (Thu): Action #3 M1 joint loss
- Edit gradient_batch.py:867 (~1h)
- Unit test gradcheck synthetic (~2h)
- Run paired N=5 baseline vs p=4 (overnight)
- Commit `feat-m1-joint-loss-p4`

### Day 5 (Fri): Read M1 results, start L-BFGS
- Analyze paired N=5 from M1
- If M1 wins → commit result, start L-BFGS
- If M1 fails → revert, full focus on L-BFGS
- Code BatchedLBFGS class (~6h, continues into Day 6)

### Day 6 (Sat): Finish L-BFGS
- Complete BatchedLBFGS (~4h)
- Wolfe LS + Powell damping (~2h)
- Unit test (~2h)
- Nightly: paired N=5 baseline vs L-BFGS

### Day 7 (Sun): Read L-BFGS, setup HPO
- Analyze L-BFGS results (commit if wins)
- Setup MOTPE Optuna sweep (~4h)
- Saturday night kicks off: MOTPE 25-30 trials (~12h overnight)
- Sunday morning: read HPO results

---

## Key sources

**Papers:**
- [Chen ICCAD 2023 — L-BFGS quasi-Newton for mixed-size placement](https://yibolin.com/publications/papers/PLACE_ICCAD2023_Chen.pdf)
- [Liu-Nocedal 1989 — L-BFGS](https://link.springer.com/article/10.1007/BF01589116)
- [Miettinen 1999 — Multiobjective Optimization (Chebyshev scalarization)]
- [Watanabe-Tsuruta 2024 — MOTPE (Optuna)]

**Repos:**
- [limbo018/DREAMPlace](https://github.com/limbo018/DREAMPlace)
- [NVlabs/AutoDMP](https://github.com/NVlabs/AutoDMP)
- [optuna/optuna](https://github.com/optuna/optuna)

---

*Document v4.0 — synthesized from 12 agents (6 research + 3 review v2 + 3 review v3 + 4 review v4) — 2026-05-09.*

## Changelog

**v5 (2026-05-09/10):**
- ⭐ Added "Empirical insights from execution" section: H2 result confirms cong-bound (`cong_frac=0.609`), M1 joint p-norm FAIL (incompatible with dynamic λ_d), cong-only ablation reveals density's job is physical spread (not proxy weight), HPO partial trial 5 = 0.8910 within noise.
- New direction proposed: top-k density (`DENSITY_TOPK_W`) + soft-overlap penalty (`OVERLAP_SOFT=1`) instead of joint p-norm reformulation.
- Visualization infra: snapshot_dump.py + render_snapshots.py (4-panel HTML, raw .npz cache).
- Action #2 (DREAMPlace) vetoed by user — don't revive without re-asking.

**v4 (2026-05-09):**
- ⭐ **Crystallized to 5 actions** with clear priority + cumulative expected gain table
- Devil's advocate adjustments: L-BFGS 8/10 → 7/10 (legalize discontinuity); HPO 8/10 → 7/10 (single-trial concern); SA-CD 7/10 → 8/10 (promoted to Day 1 sidecar)
- S1 access problem identified + DREAMPlace fallback solution
- Compute budget reality: N=17→5 paired (devil's advocate honest)
- Calibrated probabilities: P(0.85) = 0.20-0.35, P(0.80) = 0.05-0.15
- KILL list explicit (5 ideas not to do, with reasoning)
- Day-by-day Week 1 plan with concrete timestamps
- Stop conditions + decision triggers between actions
- Code skeletons (BreakdownLogger, BatchedLBFGS, MOTPE objective)

**v3 (2026-05-09):**
- Contrarian challenges (C1-C5)
- Phase 0 = mandatory diagnostics (S1, H1, H2, H3, S2, A2)
- Statistical rigor: Cohen's d, FDR Benjamini-Hochberg
- Convergence-class taxonomy with trust budget
- Density Hessian formula (Bell·Gaussian via FFT)
- Cubic Newton 2×2 closed form

**v2 (2026-05-09):**
- Math correction: ESGD-M Hutchinson (E[(Hv)²] not E[v⊙Hv])
- Cross-terms only zero for bbox WL (8% loss)
- SFN Tikhonov damping vs |λ|
- Adaptive Levenberg-Marquardt
- Effort estimates revised
- Added M1-M5, H1-H3, A2

**v1 (2026-05-09):** Initial synthesis from 6 research agents.
