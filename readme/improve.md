# Improve.md v3 — план прорыва от 0.8856 → < 0.8

> v1 (6 research agents) → v2 (3 reviewers: math, impl, missing ideas) → v3 (3 reviewers: experimental rigor, practical execution, contrarian challenge).
> Target ≤ 0.8 на ibm01 через **gradient + Hessian/2nd-order info** + diagnostics, без multi-month ML/RL.

## Контекст

- **Best 0.8856** (Round 23, single-shot lucky). Median pipeline ~0.890-0.895 (Round 28 replay = 0.8955).
- **Target ≤ 0.8.** Gap до vmallela 0.7644 = +16%.
- **Pipeline:** Adam batch K=384 (1200s) → C++ legalize → CD approx (8 rounds) → pair-swap (KNN=12, 8 rounds) → triple-cycle → CD postswap.
- **Wall:** 36+ rounds в этой сессии. История в `hessian.md` секция 7.5.

---

## ⚠️ v3 Critical reframing — contrarian challenges (must address FIRST)

**Все Tier 1-3 предполагают "continuous optimization paradigm правильна". Это не обязательно так.**

### Challenge C1: Wrong objective hypothesis
Proxy = WL_smooth + 0.5·density_RUDY + 0.5·cong_RUDY. Vmallela 0.7644 может оптимизировать **другую quantity** (routed HPWL, FLUTE Steiner) которая correlates но имеет лучший landscape. Все наши Tier 1 wasted если loss formulation сама — bottleneck.

### Challenge C2: ePlace overkill для 246 macros
ePlace machinery (FFT density 32×32 grid, gaussian mollification) designed для standard cells (миллионы tiny rects). Для IBM01 hard macros n=246 это **MILP-tractable**. Continuous relaxation сама вносит approximation errors которые constrain optimum.

### Challenge C3: Saddle obsession + noise floor
σ ≈ 0.01 (Round 28 replay). Vast majority differences между rounds — **noise**. Hessian methods cannot distinguish noise-equilibrium от saddle-equilibrium. Даже 90% saddles может означать landscape garbage, not solvable through 2nd-order.

### Challenge C4: K=384 cargo cult
384 = legacy. ROI per seed degrades после ~K=32 (within-batch correlation). 384×1200s = 460 GPU-min на **redundant exploration of one basin**. K=32 с 4× longer schedule + hyperparam-perturbed restarts likely сильнее.

### Challenge C5: Pipeline lock-in
"Adam → legalize → CD → pair-swap → triple-cycle" frozen as axiomatic. 8 dirs CD = "что было кодить удобно". 4? 16? 24 (orient-aware)? Not ablated. Same для KNN=12.

### → Phase 0 + S1+S2 sanity checks must run BEFORE Phase 1

**S1. Loss-floor probe** (2h): compute OUR proxy на externally-known good IBM01 placement (e.g., DREAMPlace public, or known reference). Three outcomes:
- proxy ≈ 0.76 → pipeline weaker, optimization problem (continue improve.md path)
- proxy ≈ 0.90 → **proxy formulation mismatch** — target unreachable через эту loss (pivot к M3 black-box)
- proxy ≈ 0.85 → partial mismatch

**S2. Diag-only Hessian gate** (3-5h, prev Tier 2.6 → now Phase 0): простейший Newton — `Δ = -g/diag(H+λI)`. If <0.001 improvement → abandon entire Newton track (Tier 1.1, 1.2, 1.3 share assumption).

**A2. SA-from-scratch on grid** (30 min CPU): random valid placement → SA с {single-move, pair-swap, triple-cycle} neighborhoods. Если final < 0.92 без gradient → pipeline simplification trumps Newton.

---

## ⚠️ v2 Math/impl corrections (preserved in v3)

### 1. ESGD-M Hutchinson — было wrong в v1
**Wrong:** `D ≈ √diag(H²)` через `E[v⊙Hv]` (это даёт `diag(H)`, signed!).
**Correct:** `D ≈ √E[(Hv)⊙(Hv)]` (per Dauphin 2015 §2-3).

### 2. Cross-terms x↔y нулевые ТОЛЬКО для bbox WL (8% loss)
density+cong (92% proxy) имеют cross-terms ≠ 0. Per-macro 2×2 Newton требует full analytical Hessian, не just WL.

### 3. SFN: Tikhonov damping вместо |λ|
**Correct:** `Δ = -∑_i (g·v_i · λ_i / (λ_i² + δ²)) · v_i`, δ = 0.01·|λ_max|. При λ→0 не взрывается.

### 4. Adaptive Levenberg-Marquardt вместо λ=1e-3
```
if det(H_i)>0 and trace(H_i)>0: Δ = -H_i⁻¹·g
else: H_i ← H_i + (|λ_min|+ε)·I  # explicit shift
backtracking line search: α=1; while L(x+αΔ)>L(x)+c·α·g·Δ: α*=0.5
```

### 5. Hutchinson noise floor
2-3 probes слишком noisy (~50% relative error). Need N=10-20 + EMA: `D_t = β·D_{t-1} + (1−β)·(Hv)⊙(Hv)`, β=0.999.

### 6. Effort estimates inflated
| Idea | Original | Real |
|---|---|---|
| Tier 1.1 Newton | 6-8h | **12-18h** |
| Tier 1.2 ESGD-M | 3-4h | **5-7h** (HVP refactor) |
| Tier 2.2 L-BFGS | 6-10h | **8-12h** |
| Tier 2.4 BB | 2-3h | **3-4h** (NOT drop-in for Adam) |

### 7. `_hvp_fd` не reusable — uses lightweight loss
Need refactor для full loss HVP (+4h).

---

## ⚠️ v3 Statistical rigor

### Power analysis for diagnostics
Round 28 replay measured `σ ≈ 0.005`. For honest detection of `Δ = 0.005 abs` (Cohen's d ≈ 1.0, 80% power, two-sided α=0.05):

**N ≥ 17 paired runs per arm.** Original plan N=5 only detects `d ≥ 1.85` (= 0.0093 abs, way too coarse).

### Multi-comparison correction
14 hypotheses tested → expected 0.7 false positives at α=0.05.

**Benjamini-Hochberg FDR-q=0.10:** rank p-values, accept idea k iff `p_(k) ≤ k·0.10/14`. Equivalent for σ=0.005:
```
require Δ ≥ 0.005·√(2·ln(14/k))
≈ 0.0115 для k=1, dropping to 0.005 при k=14
```

### Convergence-class taxonomy
| Method | Convergence | Trust |
|---|---|---|
| L-BFGS (Wolfe LS) | Globally to stationary, locally super-linear | high |
| Cubic Newton (Cartis-Gould) | Global O(ε⁻³/²) to 2nd-order points | high |
| ESGD-M | No global guarantee (stochastic precond.) | medium |
| Per-macro Newton (LM-damped) | Local quadratic, global w/ LS | medium |
| AL (Bertsekas) | Global if ω↑ unbounded | medium |
| SFN Tikhonov | Local first-order on saddles | low-medium |

### Failure modes (per Tier-1)
- **1.1:** density Hessian sign chaotic (`G''(x)·G'(y)` may be ≥0 with negative diag → indef 2×2). Detect via `det<0 ∧ trace<0` counter.
- **1.2:** EMA diverges if `(Hv)²` mean drifts → `D_t/D_{t-50}` ratio outside `[0.5,2]`.
- **1.4:** dual blow-up `μ_k → ∞` → clamp `μ_k ≤ 1e6`.
- **2.2:** L-BFGS curvature loss `s_k·y_k ≤ 1e-10·‖s_k‖‖y_k‖` → skip pair update via Powell damping.

---

## Tier 1 — Highest ROI (revised confidences)

### 1.1 Per-macro 2×2 Newton CD ★★★ (revised: 5/10)

**Method:** local 2×2 Hessian H_i для каждого макроса с **full loss components**:
- WL_smooth (LSE bbox): closed form softmax derivatives, b=0 для same-macro
- Density (ePlace eDensity): Bell convolution → `∂²E_d/∂x∂y = Σ_pq [φ_pq·b'_x·b'_y + (∂φ/∂y)·b'_x·b_y + (∂φ/∂x)·b_x·b'_y + (∂²φ/∂x∂y)·b_x·b_y]`. Reuses FFT'd `φ` from forward (+5%/step).
- Cong (RUDY top-5%): bilinear → b ≠ 0
- Overlap (rect_quad): pairwise quadratic, analytical

**Step formula (Modified Newton, per math reviewer):**
```
if det(H_i) > 0 and trace(H_i) > 0:
    Δ = -H_i⁻¹·g
else:
    H_i ← H_i + (|λ_min|+ε)·I
backtracking: α=1; while L(x+αΔ) > L(x) + c·α·g·Δ: α*=0.5
```

**Alternative: Cubic-Regularized 2×2 (closed form, math reviewer S4):**
```
Eigendecompose H = QΛQᵀ, g̃ = Qᵀg
s̃_i = -g̃_i / (λ_i + σ),  σ = M·‖s̃‖
Solve secular: σ² - M²·Σ g̃_i²/(λ_i+σ)² = 0  (1D Newton bracket)
s = Q·s̃
```
Globally convergent, escapes saddles. Preferred over Modified Newton.

**Implementation:**
- New file `submissions/straple/per_macro_newton.py`
- Integrate **в gradient_batch.py BEFORE legalize**
- Env `STRAPLE_BATCH_NEWTON_CD=1`, mutex с PLATEAU_OPS / SADDLE
- Trust-region clamp: `‖Δ‖ ≤ STRAPLE_BATCH_NEWTON_TR · cell_size`

**Expected gain:** -0.005..-0.010 (full loss). WL-only marginal -0.001.
**Effort:** **12-18h** (full loss Hessian).
**Confidence:** **5/10** (per critic — cong/density derivations non-trivial, paradigm risk).

### 1.2 ESGD-M (Hessian-diagonal preconditioner) ★★★★

**Method (corrected):**
```
D_t = β·D_{t-1} + (1−β)·(Hv)⊙(Hv), β=0.999, v∈{±1}
update: x ← x - η·g/√(D_t + ε)
```
N=5-10 probes accumulated, refresh каждые 50 steps. HVP via FD на FULL loss.

**Implementation:**
- Refactor `_hvp_fd` → full loss (+4h)
- Custom step within gradient loop, не replace `optimizer = torch.optim.Adam`
- Env `STRAPLE_BATCH_OPT=esgdm`

**Expected gain:** pre-CD min 0.91 → 0.89-0.90 → final 0.870-0.880.
**Effort:** 5-7h.
**Confidence:** 7/10. **Source:** [Dauphin 2015 arxiv 1502.04390](https://arxiv.org/abs/1502.04390).

### 1.3 Saddle-Free Newton (SFN) — fix v6 ★★★

**Method (corrected):**
```
Δ = -∑_i (g·v_i · λ_i / (λ_i² + δ²)) · v_i  # Tikhonov, не |λ|
δ = 0.01·|λ_max|
```
Lanczos top-k=10 eigenpairs.

**Bug fixes for v6:**
- σ via 10-15 Lanczos шагов с **selective re-orthogonalization** (Parlett-Scott)
- Eigvec normalization explicit ‖v_i‖=1
- HVP на **full loss**
- Tikhonov damping (no singular at λ→0)

**Alternative: Inexact Newton-CG (Steihaug truncation):**
- Solve `H·Δ = -g` via CG, stop when `‖r_k‖ ≤ η_k·‖g‖` with η_k=min(0.5, √‖g‖)
- If CG hits negative curvature direction d → return d (Steihaug)
- **No σ shift needed**, automatic saddle escape

**Effort:** 1.5-2 days. **Confidence:** 5-6/10.

### 1.4 Augmented Lagrangian (AL) для overlap ★★★★

**Method (corrected with projection):**
```
L = WL + μ·overlap + (ω/2)·overlap²
μ_{k+1} = max(0, μ_k + ω·overlap_k)  # projection ≥0
# Bertsekas §4.2: ω×=2 if ‖overlap_k‖ > η·‖overlap_{k-1}‖, η=0.25
```

**Mutex:** disable `cur_overlap_w_phase` schedule when `STRAPLE_BATCH_AUGLAG=1`.

**Expected gain:** distribution shift как Round 15 (mean -1.4%) plus best -0.5%.
**Effort:** 8-12h. **Confidence:** 6/10.

---

## Tier 2 — Medium effort

### 2.1 Cubic-Regularized Newton (Cartis-Gould-Toint 2011, closed form 2×2)
Применение в Tier 1.1. **Globally convergent O(ε⁻³/²)**. **+6h.** **Confidence:** 6/10.

### 2.2 L-BFGS late-stage finisher ★★★★ (highest confidence)

**Method:** m=10 history, batched two-loop. После step ~1000 switch с Adam.

**Scaling formula (verified by math reviewer):**
```
γ_k = (s_k·y_k) / (y_k·y_k)  # Shanno-Phua/Nocedal-Wright Eq 7.20
H_0 = γ_k·I  per-batch (γ_k ∈ R^K)
```
Powell damping: if `s_k·y_k < 1e-10·‖s_k‖·‖y_k‖` → `θ = 0.8·s·B·s/(s·B·s − s·y)`, `y_k ← θ·y_k + (1−θ)·B·s`.

**Implementation skeleton (per practical reviewer):**
```python
class BatchedLBFGS:
    def __init__(self, K, n_active, m=10, device, dtype):
        self.s = zeros(m, K, n_active, 2)
        self.y = zeros(m, K, n_active, 2)
        self.rho = zeros(m, K)
        self.head = 0; self.size = 0

    def two_loop(self, g):
        # ... (см. полный код в repo лог reviewer 2)
```
Full skeleton ~100 lines в `submissions/straple/lbfgs_finisher.py`.

**Trigger:** `STRAPLE_BATCH_LBFGS_FROM_STEP=1000`.
**Memory:** 10·384·1140·2·4 = 35 MB ✓ (use `n_active`, not `n_total`).
**Expected gain:** pre-CD min 0.904 → 0.895-0.90 → final 0.880-0.890.
**Effort:** 8-12h. **Confidence:** 8/10. **Source:** [Liu-Nocedal 1989](https://link.springer.com/article/10.1007/BF01589116), [Chen ICCAD 2023](https://yibolin.com/publications/papers/PLACE_ICCAD2023_Chen.pdf).

### 2.3 Hall placement init — re-investigate
INIT=spectral failed (Round 31). Maybe needs schedule recalibration. **4-6h.** **Confidence:** 4/10.

### 2.4 BB step (NOT Adam drop-in)
Separate optimizer track (`STRAPLE_BATCH_OPT=sgd_bb`). **3-4h.** **Confidence:** 4/10.

### 2.5 SA-style probabilistic CD acceptance ★★★★
P=exp(−Δ/T) accept worse moves. T cooling. ε-greedy random direction. **2-4h.** **Confidence:** 7/10.

---

## Tier 3 — Cheap experiments + missing ideas (v2)

### 3.1-3.5 (subsampled WL, K=1024, importance-sampled CD, path-relinking, multi-grid density) — see v2

### 🆕 M1. Joint cong+density regularizer ★★★★ [HIGH PRIORITY]

**Method (Chebyshev p=4 approximation of max):**
```python
joint_pen = ((cong_weight·cong_norm)^p + (density_weight·dens_norm)^p)^(1/p)
# p=4 → close to L_∞ norm, smooth gradient
```
Двигает к **knee point** Pareto frontier, не вдоль.

**Implementation (per practical reviewer):** edit `gradient_batch.py:867`:
```python
joint_p = float(os.environ.get("STRAPLE_BATCH_JOINT_LOSS_P", "0"))
if joint_p > 0:
    cong_norm = cong_total / cong_total.detach().clamp_min(1e-9)
    dens_norm = dpen_total / dpen_total.detach().clamp_min(1e-9)
    joint_pen = ((cong_weight·cong_norm)**joint_p + (density_weight·dens_norm)**joint_p)**(1.0/joint_p)
    loss = wl_total + cur_overlap_w_phase·overlap_total + ... + joint_pen
else:
    # original loss
```
Backwards compat: `JOINT_LOSS_P=0` (default) ≡ baseline.

**Effort:** 4-6h. **Confidence:** 6/10.

### 🆕 M3. Bookshelf → DREAMPlace black-box ★★★★ [PARALLEL TRACK]
Convert protobuf → Bookshelf, run DREAMPlace. **1-2 days.** Binary outcome (works or not). **Confidence:** 7/10.

### 🆕 M5. Bayesian HPO ★★★★ [HIGH PRIORITY]
MOTPE-Optuna 30-50 trials over 7 hyperparams. **1 day setup + overnight GPU.** **Confidence:** 8/10.

---

## Phase 0 — DIAGNOSTICS (must run BEFORE Phase 1)

### S1. Loss-floor probe ★★★★★ [contrarian]
Compute OUR proxy на externally-known good ibm01 placement.
- proxy ≈ 0.76: continue continuous path
- proxy ≈ 0.90: **proxy formulation issue** → pivot к M3
- proxy ≈ 0.85: partial mismatch

**Effort:** 2h. **Critical** — без этого все Tier 1 blind.

### H1. Saddle frac diagnostic ★★★★★
На сошедшейся точке посчитать sign(eig) распределения 2×2 локальных Hessian'ов.

**Decision rule (refined):** Newton justified iff:
```
saddle_frac ≥ 0.30 AND mean(|λ_min|/|λ_max|) ≥ 0.05
```
Иначе expected Newton gain `< 0.001 abs` (since correction `~ |λ_neg|/cond·g`).

**Effort:** 1 day. **Confidence:** 8/10.

### H2. Component breakdown logging ★★★★★
Log WL/density/cong в 5 точках pipeline.

**Code skeleton (BreakdownLogger class, ~50 lines):** see file `submissions/straple/breakdown_log.py` (to create). Wire into 5 sites:
- `placer.py:861` (post-legalize)
- `gradient_batch.py:~970` (post-gradient)
- `cd_polish.py:139,322,539,872` (CD/pair-swap/triple/postswap)

Format:
```
[BREAKDOWN stage=post_legalize seed=23 wl=0.2841 dens=1.0234 cong=1.0851 proxy=0.9237 ovl_n=0]
```

**Decision rule:** if cong fraction(post-gradient) ≥ 0.55 → **loss formulation root cause**, GO M1+M3, SKIP all Hessian work.

**Effort:** 2h. **Confidence:** 9/10.

### H3. Variance calibration (N≥17, не 5)
N≥17 trial9 baseline runs для honest σ_min, σ_median.

**Cost:** 17·25min = **7h GPU.**

**Decision rule:**
- σ ≤ 0.003 → 5-run pairs sufficient downstream
- σ 0.003-0.008 → bump K to 512, require N=8 paired runs
- σ ≥ 0.008 → critical: K=1024 + LHC mandatory; pin CUDA seed; cudnn deterministic

### S2. Diag-only Hessian gate ★★★★★ [contrarian, was Tier 2.6]
`Δ = -g/diag(H+λI)`. Gates Tier 1.1.

**Effort:** 3-5h. **Decision:** if <0.001 improvement → abandon entire Newton track.

### A2. SA-from-scratch ★★★ [contrarian sanity]
30 min CPU. Random init → SA с {move, swap, cycle}. Если final < 0.92 без gradient → pipeline simplification.

---

## Concrete roadmap v3 (Day-by-day Week 1)

### Day 1 (Mon): H2 logging
- Wire BreakdownLogger в 5 sites
- Commit `diag-h2`
- Run 1 baseline trial9 with logging on

### Day 2 (Tue): H1 + H3 launch
- Code H1 sign(eig) script (per-macro 2×2 from full loss FD-Hessian)
- Run on Day-1 final positions
- Commit `diag-h1`
- Start nightly: 17×trial9 baseline (H3, 7h GPU)

### Day 3 (Wed): S1 + S2 + A2
- S1: implement protobuf→Bookshelf converter (cheapest version, n=246 small)
- Run S1 against any known-good placement
- S2: Diag-only Hessian (3-5h)
- A2: SA-from-scratch (1h)
- **Decision point** based on H1+H2+H3+S1+S2+A2 results

### Day 4-5 (Thu-Fri): M1 + AL or pivot
**If diagnostics align (continuous path viable):**
- Day 4: Code M1 joint-loss + unit-test gradcheck. Commit `feat-m1`. Nightly: 17×base + 17×M1.
- Day 5: Read M1 result. Code AL.

**If S1 shows proxy mismatch:**
- Day 4: M3 DREAMPlace black-box (full integration)
- Day 5: Continue M3 + analyze

### Day 6-7 (Sat-Sun): L-BFGS + HPO overnight
- Code L-BFGS finisher (`lbfgs_finisher.py`, 100-150 lines)
- Setup MOTPE Optuna sweep (30-50 trials)
- Sat night: 17×base + 17×L-BFGS
- Sat night also: MOTPE 30-trial (12h overnight)
- Sun: read results, plan Week 2

---

## Decision tree (Phase 0 → Phase 1) — refined

```
S1 (loss-floor probe, OUR proxy on known-good placement)
  ├─ proxy ≈ 0.76: continue improve.md path
  ├─ proxy ≈ 0.85: partial mismatch — focus M1 + M3
  └─ proxy ≈ 0.90: PIVOT entirely to M3 + M4 + objective re-engineering

H3 (N=17 baseline runs)
  └─ measure σ_min, σ_median

H2 (component logging at 5 points)
  └─ if cong fraction(post-gradient) ≥ 0.55:
        → loss-formulation root cause; GO M1+M3, SKIP Hessian
     else:
        → optimization-side; continue

S2 (Diag-only Hessian, 3-5h)
  ├─ Δ ≥ 0.005 abs: full Tier 1.1 worth
  └─ Δ < 0.001 abs: ABANDON Newton track, focus L-BFGS+M1+AL

H1 (eig-sign over 246 hard macros)
  ├─ saddle_frac ≥ 0.30 AND |λ_min/λ_max|_med ≥ 0.05
  │     → GO Tier 1.1 (Cubic Newton) + Tier 2.6 (gate)
  ├─ 0.05 ≤ saddle_frac < 0.30
  │     → SFN/cubic marginal; GO L-BFGS (2.2) primary
  └─ saddle_frac < 0.05
        → no saddles; GO M1 + AL + L-BFGS finisher

A2 (SA-from-scratch)
  ├─ final < 0.92: simpler paradigm viable, refactor
  └─ final ≥ 0.95: gradient phase essential, continue

Backup if first deserialized idea fails:
  Tier 2.2 L-BFGS (highest confidence 8/10) → fallback always
  If L-BFGS flat → M3 DREAMPlace (paradigm shift)
```

---

## Compute budget breakdown (single T4, ≤30 min/run)

| Phase | Coding | GPU | Wall (calendar) |
|---|---|---|---|
| Phase 0 (S1+H1+H2+H3+S2+A2) | 8h | 13h (17×base + diags) | 1.5-2 days |
| Phase 1 (M1+AL+diagH gate) | 22h | 16h (4 paired N=17 + ablations) | 4-5 days |
| Phase 2 (L-BFGS+ESGD-M+HPO) | 25h | 20h+overnight HPO 12h | 5-6 days |
| Buffer | — | 8h | 2 days |
| **Total Week-1+2** | **55h coding** | **~50h GPU** | **12-14 calendar days** |

---

## Test design table (mandatory for any Phase-1 idea)

| Idea | Acceptance | Time/test | N pairs | Pass/Fail/Inconclusive |
|---|---|---|---|---|
| H2 logging | Diagnostic | 25 min | 1 | n/a |
| H3 variance | σ measurement | 25min × 17 = 7h | n/a | calibration |
| H1 saddle | %indef computed | 30min × 3 | n/a | n/a |
| M1 joint p=4 | median Δ ≤ −0.005 (FDR) | 8.5h paired (17×2) | 17 | <−0.003 inconclusive→re-run; ≥0 fail |
| AL overlap | median Δ ≤ −0.005 | 8.5h | 17 | same |
| L-BFGS finisher | min over 17 ≤ baseline_min−0.003 | 8.5h | 17 | <−0.001 fail |
| Diag-Hessian (S2) | median Δ ≤ −0.005 | 8.5h | 17 | gates Tier 1.1 |
| ESGD-M | median pre-CD ≤ 0.90 | 8.5h | 17 | else fall back to L-BFGS |
| Bayesian HPO | best ≤ baseline_best−0.005 | overnight 12h | n/a | always commit best config |

---

## Risk register

| Risk | P | Mitigation |
|---|---|---|
| OOM at K=384 with HVP probes | M | gate by env `STRAPLE_BATCH_HVP_PROBES`; chunk over K; first test K=128 |
| NaN (sqrt at indef λ, /0 in BB) | H | clamp_min(1e-12); `torch.isfinite` early-abort with skip |
| Integration conflict (AL + overlap_w + overflow_lambda) | H | mutex env-vars: AUGLAG=1 disables `cur_overlap_w_phase`; assert at startup |
| L-BFGS curvature corruption near legalize | M | drop history (`size=0`) on phase boundary; line-search reject if sy<1e-10 |
| Wall-time blowup (LBFGS LS 8x evals) | M | max 4 evals; parallel K → amortizes |
| Bug in joint-loss gradient (M1) | M | unit-test: backward(joint_p=1) ≡ baseline; gradcheck synthetic 4-macro |
| Density Hessian sign chaos (Tier 1.1) | H | derive from existing ePlace forward; cross-check via FD 1e-3 |

---

## Pessimistic floor (multi-reviewer alignment)

- **Без months of work:** 0.870-0.875 на ibm01 (-1% от 0.8856).
- **Чтобы пробить 0.85:** combo H1+H2+M1+L-BFGS → **0.85-0.86 likely** (1-2 weeks).
- **0.7644 vmallela:** требует paradigm shift, 1-2 месяца.

**Но contrarian C1:** если S1 покажет proxy formulation mismatch, наш ceiling может быть **0.85 fundamentally**, не 0.8.

---

## Pragmatic 1-2 week combo (recommended)

**Week 1:**
- Day 1-3: ALL Phase 0 diagnostics (S1, H1, H2, H3, S2, A2)
- Day 4-5: M1 joint loss + AL (если diagnostics good)
- Day 6: L-BFGS finisher
- Day 7: MOTPE HPO overnight + Sun analysis

**Week 2 (decision-tree based):**
- **Path A (saddles + continuous viable):** Per-macro Cubic Newton + ESGD-M
- **Path B (loss formulation issue):** M3 DREAMPlace + objective re-engineering
- **Path C (variance dominant):** K=1024 + LHC + multi-run averaging

**Expected after 2 weeks:** ibm01 best 0.85-0.87 high confidence; 0.83-0.85 stretch.

---

## Key sources (v3)

**Papers:**
- [Chen ICCAD 2023 — L-BFGS quasi-Newton mixed-size placement](https://yibolin.com/publications/papers/PLACE_ICCAD2023_Chen.pdf)
- [ePlace TODAES'15 — diagonal Hessian preconditioner](https://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf)
- [Dauphin 2014 — SFN (arxiv 1406.2572)](https://arxiv.org/abs/1406.2572)
- [Dauphin 2015 — Equilibrated learning rates (arxiv 1502.04390)](https://arxiv.org/abs/1502.04390)
- [Nesterov-Polyak 2006 — Cubic-Regularized Newton](https://link.springer.com/article/10.1007/s10107-006-0706-8)
- [Cartis-Gould-Toint 2011 — TRACE](https://link.springer.com/article/10.1007/s10107-009-0337-y)
- [Tripuraneni 2017 — Stochastic Cubic (arxiv 1711.02838)](https://arxiv.org/abs/1711.02838)
- [Steihaug 1983 — CG truncation for Newton](https://epubs.siam.org/doi/10.1137/0720042)
- [Liu-Nocedal 1989 — L-BFGS](https://link.springer.com/article/10.1007/BF01589116)
- [Curtis-Jiang-Robinson 2015 — Adaptive AugLag](https://coral.ise.lehigh.edu/frankecurtis/files/papers/CurtJianRobi15.pdf)
- Bertsekas 1996 — Constrained optimization

**Repos:**
- [limbo018/DREAMPlace](https://github.com/limbo018/DREAMPlace)
- [crowsonkb/esgd](https://github.com/crowsonkb/esgd)
- [NVlabs/AutoDMP](https://github.com/NVlabs/AutoDMP)

---

*Document v3.0 — synthesized from 9 agents (6 research + 3 review v2) + 3 review v3 — 2026-05-09.*

## Changelog

**v3 (2026-05-09):**
- ⚠️ **Contrarian challenges (C1-C5):** wrong objective hypothesis, ePlace overkill, saddle obsession, K=384 cargo cult, pipeline lock-in
- ⚠️ **Phase 0 expanded:** S1 (loss-floor probe), S2 (diag-Hessian gate), A2 (SA-from-scratch) — sanity checks BEFORE Phase 1
- Statistical rigor: H3 N≥17 (not 5), Cohen's d analysis, FDR Benjamini-Hochberg q=0.10
- Convergence-class taxonomy with trust budget
- Failure modes per Tier-1
- Density Hessian formula derived (Bell·Gaussian via FFT'd φ)
- L-BFGS scaling formula confirmed (Shanno-Phua/Nocedal-Wright Eq 7.20)
- Cubic Newton 2×2 closed form via secular equation
- Code skeletons: BreakdownLogger, joint loss patch, BatchedLBFGS class
- Test design table with FDR-corrected acceptance thresholds
- Risk register with mitigation
- Day-by-day Week 1 plan
- Decision tree refined для Phase 0 → Phase 1
- Compute budget: 55h coding + 50h GPU = 12-14 calendar days

**v2 (2026-05-09):**
- Math correction: ESGD-M Hutchinson formula (E[(Hv)²] not E[v⊙Hv])
- Math correction: Cross-terms only zero for bbox WL (8% loss)
- Math correction: SFN Tikhonov damping vs |λ|
- Math correction: Adaptive Levenberg-Marquardt (not fixed λ=1e-3)
- Implementation: Effort estimates revised
- Strategy: Added M1-M5, H1-H3, A2 (joint loss, orient flip, DREAMPlace, net weight, Bayesian HPO, diagnostics, cluster CD)
- Strategy: Phase 0 = diagnostics BEFORE Newton

**v1 (2026-05-09):** Initial synthesis from 6 research agents.
