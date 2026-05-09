# Final Action Plan — ibm01 0.8856 → ≤0.85 high confidence

> Synthesized from `improve.md` v3 + `hessian.md` Rounds 1-30 (28 attempts).
> 5 actions ranked by `expected_value × probability_success`.
> One-sentence pitch: **We do diagnostics-first (H2 breakdown logging), then attack the dominant 61% congestion via DREAMPlace black-box (M3) and joint p=4 loss (M1), then add L-BFGS finisher and Bayesian HPO over the working pipeline, expecting ibm01 best ≈ 0.83-0.85 in 1-2 weeks.**

---

## Action #1: H2 Component Breakdown Logging + S1 Loss-Floor Probe (Phase 0 diagnostics)

### ЧТО
Залогировать WL/density/cong на 5 точках pipeline (post-gradient, post-legalize, post-CD, post-pair-swap, post-triple) и одновременно прогнать наш proxy на known-good external IBM01 placement.

### ПОЧЕМУ — HIGH CONFIDENCE reasoning
- **Mechanism:** Round 18 уже показал cong=61%, dens=31%, WL=8% on TILOS final — это ОДНА точка. H2 даёт **trajectory** через всю pipeline → видно где caming cong "застывает". Если cong=0.95 уже после gradient и не двигается через CD/pair-swap → dens/WL CD не лечит cong. Если cong падает с 1.20→1.05 в pair-swap → есть hook.
- **Evidence:** Rounds 19-20 показали Pareto frontier dens↔cong (cong↓ ⇒ dens↑) — но без trajectory мы не знаем, **откуда** этот frontier появляется (gradient init? CD over-packing? оба?).
- **Why others не addressэт:** все Tier 1 Newton/L-BFGS/AL ideas работают на gradient phase, но если cong дominates **post-CD** (не gradient) — Newton wasted. S1 говорит — может ли наш proxy в принципе достичь 0.76, или мы упёрлись в формулировку.
- **Theoretical backing:** Bishop-style component analysis. v3 improve.md называет H2 "★★★★★ confidence 9/10". Это самая дешёвая, самая высокоinformative diagnostic.

### КАК — Implementation
1. **Создать** `submissions/straple/breakdown_log.py`: класс `BreakdownLogger` с методом `log(stage, seed, pos, plc)` который вызывает TILOS proxy_cost и парсит компоненты + outputs `[BREAKDOWN stage=X seed=Y wl=... dens=... cong=... proxy=... ovl_n=...]`.
2. **Wire в 5 sites:**
   - `gradient_batch.py` ~line 970 (post-gradient, перед legalize)
   - `placer.py:861` (post-legalize, best seed)
   - `cd_polish.py:139` (post-single-macro CD)
   - `cd_polish.py:539` (post-pair-swap)
   - `cd_polish.py:872` (post-triple-cycle)
3. **Env:** `STRAPLE_BATCH_BREAKDOWN_LOG=1` (default off).
4. **S1 — Bookshelf converter** (новый `scripts/proto_to_bookshelf.py`): protobuf → Bookshelf для ibm01 (n=246 hard, 5993 nets — small, 1-2h работы). Run DREAMPlace public binary on it → получить .pl файл. Прогнать через `compute_proxy_cost` → получить наш proxy на их placement.
5. **Test:** 1 run trial9+full pipeline с logging on (25 min) + S1 проверка (~1h).

### Когда сделано (Done criterion)
- Breakdown log printed на all 5 stages, valid float values, no NaN.
- S1 produces our_proxy на DREAMPlace placement.
- Decision rule applied:
  - `cong_fraction(post-gradient) ≥ 0.55` → SKIP всех Hessian ideas, GO Action #2 (M3) + Action #3 (M1).
  - `S1 proxy ≈ 0.76` → continuous path viable, can chase Action #4 (L-BFGS).
  - `S1 proxy ≥ 0.85` → proxy formulation hard ceiling, M3 не поможет → focus только Action #5 (HPO over current best).

### Risks
1. **TILOS proxy_cost slow (~2s/call × 5 stages × 1 run = 10s overhead)** — negligible, mitigate просто игнорируя.
2. **Bookshelf converter buggy** (rare format edge cases) — mitigate: cross-check WL только на нашем placement (round-trip test), если совпадает — converter ok.

### Time
**5-7h** (BreakdownLogger 2h + Bookshelf converter 3h + DREAMPlace install/run 2h).

### Confidence
**9/10.** Diagnostics не могут "fail to improve" — они либо unblock следующее decision, либо подтверждают, что continuous path viable. Самая дешёвая ROI/час из всех 5 actions.

---

## Action #2: DREAMPlace Black-Box Integration (M3)

### ЧТО
Запустить DREAMPlace (state-of-art ePlace++L-BFGS) поверх наших данных через protobuf↔Bookshelf конверсию, использовать его placement как **input** для нашей CD+pair-swap+triple stack.

### ПОЧЕМУ — HIGH CONFIDENCE reasoning
- **Mechanism:** наш Adam batch K=384 — генерик optimizer без macro-specific tricks (Lipschitz adaptive Nesterov, line search, multi-grid density). DREAMPlace = 8+ years of placement-specific engineering. Round 27 показал что custom Nesterov у нас не работает без full DREAMPlace stack — значит нет смысла повторять с нуля; легче использовать готовое.
- **Evidence:** improve.md M3 confidence 7/10. UT Austin paper claims DREAMPlace AVG17 ≈ 1.41 (мы на ~1.0 = +40% хуже на 17 benchmarks, на ibm01 specifically ~25% gap). vmallela (1st place) обычно использует DREAMPlace + custom CD. Round 12 показал что просто увеличение time_budget без custom optimizer вредит.
- **Why others не address:** Tier 1 Newton/SFN — micro-optimization gradient phase в нашем broken framework. M3 — **paradigm replacement**.
- **Theoretical backing:** [Lin Chen ICCAD 2023](https://yibolin.com/publications/papers/PLACE_ICCAD2023_Chen.pdf) explicit DREAMPlace + L-BFGS quasi-Newton — это и есть Action #4 fallback inside M3.

### КАК — Implementation
1. **Bookshelf converter (created в Action #1):** protobuf netlist → .nodes/.nets/.pl/.scl/.wts/.aux.
2. **DREAMPlace install** на server (`pip install dreamplace` or git clone limbo018/DREAMPlace + build CUDA extension).
3. **Wrapper** `scripts/dreamplace_run.py`: read protobuf → convert → run DREAMPlace → read output .pl → convert back → return pos tensor [n_macros, 2].
4. **Integration в pipeline:** новый env `STRAPLE_BATCH_DREAMPLACE=1` в `gpu_run_one.py`. Если set → SKIP gradient batch, instead use DREAMPlace pos as **single best seed** → applies legalize → CD → pair-swap → triple as obычно.
5. **Test:** ibm01 single-run wall budget 30 min (DREAMPlace ~10 min + our pipeline 15 min).

### Когда сделано (Done criterion)
- DREAMPlace runs без crash на ibm01.
- Returns valid .pl файл (no overlaps via legalize).
- Final proxy after our CD+pair-swap stack: **single run target ≤ 0.86**, paired N=5 median ≤ 0.87.
- If DREAMPlace alone gives < 0.88 без CD → **NEW BEST**, integrate как default.
- If DREAMPlace produces invalid (overlap, off-canvas) — debug Bookshelf format, allocate 4h extra.

### Risks
1. **Bookshelf format mismatch** (soft macros, blockages, cell sizes — DREAMPlace expects standard cells). Mitigate: ibm01 — 246 hard macros, treat soft cells как standard cells, validate WL preservation round-trip.
2. **GPU/CUDA версия conflicts** на T4. Mitigate: use Docker image dreamplace official; fallback к CPU build (slower но work).

### Time
**12-18h** (1.5-2 days). Bookshelf 4h + DREAMPlace install/debug 4h + wrapper 4h + integration test 4h + 1 buffer day.

### Confidence
**7/10.** Highest expected value (paradigm shift) but binary outcome (works/doesn't). Если works → ibm01 0.83-0.85 likely. Если не works → 1-2 lost days, но мы получим diagnostic info.

---

## Action #3: M1 Joint p=4 Cong+Density Loss

### ЧТО
Заменить additive `0.5·cong + 0.5·dens` на Chebyshev p-norm `((cw·cong)^p + (dw·dens)^p)^(1/p)` с p=4 в gradient loss, чтобы оптимизатор двигался к **knee point** Pareto frontier, не вдоль.

### ПОЧЕМУ — HIGH CONFIDENCE reasoning
- **Mechanism:** Round 19 (cong_w=20) и Round 20 (cong_w=15+top_pct=0.05) обе хитнули **симметричный Pareto frontier** — cong↓ ⇒ dens↑ симметрично. Additive loss `cong+dens` имеет линейные level-curves параллельные frontier → optimizer движется ВДОЛЬ → trade-off. P=4 имеет квадратные L_∞-like level-curves → "**knee-seeking**" → finds Pareto knee corner.
- **Evidence:** Round 18 breakdown показал post-pair-swap cong=1.05, dens=0.58 — мы в верхнем-левом углу frontier (cong-dominated). Knee — где cong≈dens≈0.7-0.8 → proxy ≈ WL(0.07) + 0.5·0.7 + 0.5·0.7 = 0.07 + 0.35 + 0.35 = **0.77** (близко к vmallela!).
- **Why others не address:** Tier 1.1 Per-macro Newton, AL overlap, L-BFGS — все **optimizer changes** на ту же loss → они движутся по тому же frontier. M1 — **loss reformulation** → меняет geometry, не optimizer.
- **Theoretical backing:** Chebyshev scalarization (Miettinen 1999, Multiobjective Optimization). p=4 — эмпирически близко к L_∞ (smooth gradient, без kinks). [Practical reviewer's exact code skeleton в improve.md M1 секции].

### КАК — Implementation
1. **Edit `submissions/straple/gradient_batch.py:867`** (точка где формируется total loss):
   ```python
   joint_p = float(os.environ.get("STRAPLE_BATCH_JOINT_LOSS_P", "0"))
   if joint_p > 0:
       cong_norm = cong_total / cong_total.detach().clamp_min(1e-9)
       dens_norm = dpen_total / dpen_total.detach().clamp_min(1e-9)
       joint_pen = ((cong_weight * cong_norm)**joint_p +
                    (density_weight * dens_norm)**joint_p)**(1.0/joint_p)
       loss = wl_total + cur_overlap_w_phase * overlap_total + joint_pen
   else:
       loss = wl_total + cur_overlap_w_phase * overlap_total + 0.5*cong_total + 0.5*dpen_total
   ```
2. **Env:** `STRAPLE_BATCH_JOINT_LOSS_P=4` (default 0 = backwards compat).
3. **Unit test** (новый `tests/test_joint_loss.py`): backward(p=1) ≡ baseline (analytical check, gradcheck synthetic 4-macro toy).
4. **Test:** 1 paired run (baseline vs p=4) on ibm01, 1200s gradient.

### Когда сделано (Done criterion)
- Unit test passes (gradcheck OK).
- Run `STRAPLE_BATCH_JOINT_LOSS_P=4` без NaN/inf через 500 steps.
- Pre-CD min ≤ 0.90 (vs typical 0.91) on single run.
- Paired N=5 runs: median Δ ≤ -0.005 (proxy lower).
- Breakdown: cong<1.00 AND dens<0.70 simultaneously (knee found).
- If pre-CD не двигается ≤ 0.91 в 3 runs → drop p=4, try p=2 (smoother).

### Risks
1. **Gradient instability** при p=4 (high-power gradient may explode). Mitigate: clamp_min(1e-9), test p=2 first, monitor `torch.isfinite` → early-abort.
2. **Frontier shifts to different corner** (e.g., dens=0.05 cong=1.5). Mitigate: log breakdown per-step, abort if cong > 1.3 at step=200.

### Time
**4-6h** (edit 1h + unit test 2h + run + analyze 2-3h).

### Confidence
**6/10.** Theoretically clean но empirical risk: p=4 mathematically correct, но frontier может быть non-convex → knee может не существовать. Backup: p=2.

---

## Action #4: L-BFGS Late-Stage Finisher

### ЧТО
После Adam (step 1000), переключиться на batched L-BFGS (m=10 history, Wolfe line search) для **super-linear convergence** на settled landscape.

### ПОЧЕМУ — HIGH CONFIDENCE reasoning
- **Mechanism:** Adam — 1st-order, scale-invariant per-coord. К концу gradient phase (P3 settling) landscape почти-quadratic. L-BFGS approximates inverse Hessian через secant updates → **локально quadratic convergence**. По Round 12 анализу: late steps Adam wastes — λ_o растёт без проп, gradient noise dominates.
- **Evidence:** Chen ICCAD 2023 сравнили Adam-only vs Adam→L-BFGS: -3-7% wirelength on standard benchmarks. DREAMPlace defaults к L-BFGS finisher. Round 30 показал что extending pair-swap не пробивает -0.005 floor — значит final improvement должен идти от **gradient phase**, а Adam там converged. L-BFGS — natural continuation.
- **Why others не address:** Per-macro Newton (1.1) — full Hessian — overkill 12-18h работы. SFN (1.3) — saddle-specific (Round 1-7 показали saddle hypothesis weak в нашем paradigm). L-BFGS — proven, batched, no Hessian computation.
- **Theoretical backing:** [Liu-Nocedal 1989](https://link.springer.com/article/10.1007/BF01589116) глобально сходящийся к stationary point (Wolfe LS) + локально super-linear. Trust = **high** (улучшая improve.md taxonomy).

### КАК — Implementation
1. **Новый файл** `submissions/straple/lbfgs_finisher.py` (~150 lines):
   - `class BatchedLBFGS(K, n_active, m=10, device, dtype)` с `s, y, rho` буферами shape (m, K, n_active, 2).
   - Method `two_loop(g)` — стандартный Liu-Nocedal алгоритм, batched по K.
   - Method `step(loss_fn, pos, g)` — Wolfe line search, max 4 evals.
   - **Powell damping** при `s·y < 1e-10·‖s‖·‖y‖`: `θ=0.8·s·B·s/(s·B·s − s·y)`, `y ← θy + (1−θ)·B·s`.
2. **Integration в `gradient_batch.py`** ~line 700 (gradient loop):
   ```python
   lbfgs_from_step = int(os.environ.get("STRAPLE_BATCH_LBFGS_FROM_STEP", "0"))
   if lbfgs_from_step > 0 and step >= lbfgs_from_step:
       lbfgs.step(loss_fn, pos, grad)
   else:
       optimizer.step()  # Adam
   ```
3. **Env:** `STRAPLE_BATCH_LBFGS_FROM_STEP=1000` (default 0 = no L-BFGS).
4. **Memory check:** 10·384·1140·2·4B = 35 MB → fits T4 16GB.
5. **Test:** paired N=5 baseline vs L-BFGS (1200s gradient).

### Когда сделано (Done criterion)
- No NaN/inf через 1200 steps.
- Memory peak < 11 GB (T4 has 16 GB, baseline peak 10.2 GB).
- Pre-CD min ≤ 0.90 (vs typical 0.91) на 5/5 runs.
- Paired N=5 median Δ ≤ -0.003.
- If LS rejects > 50% steps → curvature corruption → revert (drop history at step=1100).

### Risks
1. **Curvature corruption near phase boundaries** (P2→P3 при step=480) → s·y < 0. Mitigate: drop history (`size=0`) при `cur_overlap_w_phase` change; Powell damping inline.
2. **LS line search blowup wall-time** (8x evals × 1.5s/eval = 12s/step). Mitigate: max_evals=4 + parallel K amortizes; abort step if α<1e-6.

### Time
**8-12h** (BatchedLBFGS class 6h + integration 2h + Powell + tests 2-4h).

### Confidence
**8/10.** Highest confidence из всех Tier 1-2 ideas (per improve.md). Proven algorithm, clear integration, bounded risk, batched memory ok.

---

## Action #5: MOTPE Bayesian HPO Over Working Pipeline

### ЧТО
Optuna MOTPE (multi-objective Tree-Parzen Estimator) sweep 30-50 trials over 7 hyperparams Round 23 best pipeline, optimizing (proxy_min, std) Pareto front overnight.

### ПОЧЕМУ — HIGH CONFIDENCE reasoning
- **Mechanism:** Rounds 14-30 показали что individual hyperparam tweaks (cong_w, top_pct, time_budget, K) каждый меняет distribution на Pareto. **Manual tuning** в 1D — slow и hits floors. MOTPE — Bayesian — exploits correlations, auto-prunes bad trials. Round 23 best 0.8856 был **lucky single-shot**; HPO найдёт config где median ≤ 0.89.
- **Evidence:** Round 28 replay 0.8955 показал σ ~0.005-0.01 between runs same config. Если HPO reduces variance through better config (e.g., higher K, longer schedule, better λ_o cap) → median floor дipsi. AutoDMP (NVlabs) использует MOTPE внутри placement → published gain 5-10% over manual.
- **Why others не address:** все Actions 1-4 — single-shot improvements. Action 5 — **stochastic exploitation** дамаются. Cheap (overnight GPU), auto.
- **Theoretical backing:** MOTPE [Watanabe-Tsuruta 2024] dominates random search 5-10x на high-dim. improve.md M5 confidence 8/10.

### КАК — Implementation
1. **Новый скрипт** `scripts/hpo_motpe.py`:
   - `import optuna; sampler = optuna.samplers.MOTPESampler(seed=42)`.
   - Search space (7 params):
     - `STRAPLE_BATCH_OVERLAP_W_MAX` ∈ [20000, 100000] (log).
     - `STRAPLE_BATCH_OVERLAP_W_GROWTH` ∈ [1.002, 1.010] (log).
     - `STRAPLE_BATCH_OVERFLOW_TARGET` ∈ [0.08, 0.20].
     - `STRAPLE_BATCH_OVERFLOW_EXP` ∈ [0.5, 1.0].
     - `STRAPLE_BATCH_CONG_W` ∈ [5, 25].
     - `STRAPLE_BATCH_PAIR_SWAP_ROUNDS` ∈ {6, 8, 10, 12}.
     - `STRAPLE_BATCH_K` (batch size) ∈ {256, 384, 512}.
   - Objective: `(proxy, -reproducibility_count)` (multi-obj).
   - Trial = run `gpu_run_one.py` on ibm01, time-budget=1200s, single seed → ~25 min/trial.
2. **Pruning:** medianpruner — kill trials worse than median at step 600.
3. **Run:** overnight 12h → ~28 trials.
4. **Output:** Pareto front config dump → pick best by `proxy + 0.5·std`.

### Когда сделано (Done criterion)
- 25-30 trials complete (no crashes).
- Best HPO config `proxy_min ≤ 0.880` paired N=5 (vs Round 23 0.8856 ± 0.01 noise).
- Repeated best config 5x: std ≤ 0.005 (down from 0.01 baseline).
- Pareto front documented; commit best config as `trial10` baseline.
- If best HPO config no improvement over Round 23 (paired N=5 median Δ < -0.003) → confirm we're at Pareto floor, escalate to Action #2 (M3).

### Risks
1. **HPO finds overfit config** (lucky single trial). Mitigate: top-3 configs все retest N=5; pick by **median**, not best.
2. **GPU contention** if T4 shared. Mitigate: run на dedicated server window, kill других jobs первым.

### Time
**8-12h coding + overnight 12h GPU.** (HPO script 4h + Optuna setup 2h + analysis 2-4h).

### Confidence
**8/10.** Auto-tuning has lower variance than manual; works on **any** pipeline (orthogonal to other ideas). Может combinе с Action #3 (M1 joint loss as 8th hyperparam) после M1 implementation.

---

## Execution Flow Diagram

```
START: Day 0
  │
  ▼
[Action #1: H2 + S1 diagnostics]  (5-7h, Day 1-2)
  │
  ├─ S1 proxy ≈ 0.76 ───────────► [continuous path viable]
  ├─ S1 proxy ≥ 0.85 ───────────► [proxy formulation cap, SKIP M3, GO Action #5 only]
  └─ cong_fraction ≥ 0.55 ──────► [SKIP Hessian, GO M3 + M1]

  │ (most likely branch)
  ▼
[Action #2: DREAMPlace M3]  (12-18h, Day 3-4)
  │
  ├─ ibm01 < 0.88 ─────────────► NEW BEST → [refine with our CD+pair-swap]
  │                                       └─► run Action #5 over DREAMPlace baseline
  └─ ibm01 ≥ 0.89 (no help) ────► [continue to Action #3]
  │
  ▼
[Action #3: M1 joint loss p=4]  (4-6h, Day 5)
  │
  ├─ pre-CD min ≤ 0.90 paired N=5 ──► [stack with current pipeline → Action #4]
  └─ no improvement OR breakdown anomaly ──► [revert, GO Action #4 standalone]
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
  ├─ best config ibm01 ≤ 0.84 ────► STOP: GOAL REACHED
  ├─ ibm01 ∈ [0.85, 0.87] ────────► [accept, submit, schedule Week 2 for paradigm shifts]
  └─ ibm01 ≥ 0.88 ───────────────► PROVISIONAL DEFEAT: pivot to M4 (RL/ML), 1-2 month effort
```

### Decision triggers between actions
- **After Action #1:** breakdown reveals dominant component → routes to optimization (Adam/L-BFGS) или formulation (M1/M3) или paradigm (RL).
- **After Action #2:** binary works/doesn't. Если works — superseeds gradient phase, остальные actions stack on top.
- **After Action #3:** check breakdown again — knee-seeking moved cong&dens both down? If yes — keep p=4, stack #4. If no — revert.
- **After Action #4:** L-BFGS adds late-stage convergence; combine с M1 (если won) для cumulative.
- **After Action #5:** automated final tuning over best stack from #2-#4.

### Stop conditions
- **Success:** ibm01 ≤ 0.85 paired N≥5 median, std ≤ 0.005 → submit, lock config, move to AVG17 verification.
- **Provisional defeat:** после всех 5 actions ibm01 > 0.88 → escalate (M4 RL learning, 1-2 month timeline) или accept current best 0.8856 as session ceiling.
- **Mid-loop pivot:** если Action #1 S1 показывает proxy ≥ 0.90 на reference placement → ceiling fundamental, skip optimization actions, focus only HPO над Round 23 + submit.

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

Realistic target after Week 2: **ibm01 ≈ 0.85-0.87**. Stretch (Action #2 succeeds + #5 finds knee): **0.79-0.82**.
