# Идеи улучшения текущего pipeline

> Анализ от 2026-05-07 после прочтения всех проектных .md.
> Контекст: submitted ALNS AVG17=1.4445 (~16 место), pure-gradient GPU best ibm01=0.9453 (vs MTK 0.91, gap +3.9%).
> Цель — пробить топ-7 (AVG17 ≤ 1.3479) для квалификации на Гран-при.
>
> **Update 2026-05-07 (после Tier 1+2+3):** ibm01 best 0.9128 (паритет с MTK 0.91), median 1.06 (-16% vs original).
> Verified leader: **vmallela v7 = AVG17 1.0109** (wins all 17 benches), использует **Hessian negative-eigenvalue saddle escape**.
> Это сдвигает Tier 0 — без saddle escape выше топ-3 не пройти.

---

## 0. Tier 0 (TOP PRIORITY) — Hessian saddle escape

**Источник:** vmallela v7 leaderboard verification (AVG17=1.0109, beats all 17 benches verified). Сами слова: «Hessian negative-eigenvalue saddle escape (`vmallela_v7` branch). Previous Incremental CD+LNS verified 1.4152.»

**Идея.** В non-convex high-dim landscape (наши 2280 dims = 1140 макросов × 2) **седловые точки доминируют над локальными минимумами** (~2^n соотношение). Стандартный gradient descent видит `∇L ≈ 0` и останавливается, но в седле есть направления отрицательной кривизны — туда можно ещё спуститься. Random teleport / shake (наш `STRAPLE_BATCH_PLATEAU_OPS`) угадывает направление наугад → часто заходит в худший basin или возвращается. **Hessian eigenvector = математически точный escape direction.**

**Алгоритм (v7-style):**
1. Detector: для каждого K seed track loss за last 30 шагов. Если `(max-min)/|median| < ε` И `‖∇L‖` маленький → подозрение на седло.
2. Compute smallest eigenvalue λ_min Hessian'а через **Lanczos / power iteration on `-H`** (~15-20 HVPs).
3. Если λ_min < `-threshold` → это седло; v_min = соответствующий eigenvector.
4. Step `pos += step_size · v_min`, обнулить Adam moments для перетеленных pos.
5. Continue gradient descent.

**Compute primitives:**
- HVP (Hessian-vector product) через PyTorch double-backward:
  ```python
  grads = torch.autograd.grad(loss, pos, create_graph=True)[0]
  Hv = torch.autograd.grad((grads * v).sum(), pos)[0]
  ```
  Стоимость: ~3× обычного backward. Memory: ~2× (double-backward graph).
- Lanczos / power iteration: ~15-20 HVPs на seed для smallest eig+vector.

**Cost estimate (наш T4 K=384 budget=600s):**
- Trigger каждые ~50-100 steps в P2/P3 only.
- Только для seeds в плато (детектор) — типично 10-30% K в данный момент.
- ~15 HVPs × 3× backward × 0.2 × K_in_plateau / total_steps ≈ 5-15% overhead.
- Memory: chunked over K (по 64 seed за раз) чтобы вместить double-backward graph + ePlace + cong.

**Главные риски:**
1. **Memory OOM** на T4 16GB при K=384 + double-backward + ePlace 128 + cong. Решение — chunked compute.
2. **Lanczos numerical instability** на indefinite Hessian. Решение — restart / shift trick / fall back to random direction.
3. **Step size tuning** — too big разрушает layout, too small ничего не даёт. Tune через `STRAPLE_BATCH_SADDLE_STEP`.
4. **Compute time** — 5-15% overhead = меньше gradient steps в budget. Если saddle escape даёт +0.5% на step → break-even при 3% overhead.

**Confidence:** 9/10 что это работает (verified в leader's submission). 6/10 что мы аккуратно реализуем за 3-5 days — много sharp corners.

**Плюс — какие седла мы реально встречаем:**
- На ibm01 в P3 (settling) λ_d=2000, overlap уже почти 0, но WL не уменьшается → плато из-за «структурного баланса» between cluster (cohesion) и spreading (density). Это ровно седло — пара кластеров «зеркальны», push в любом направлении ломает баланс с одной стороны и улучшает с другой. Hessian eigenvector укажет именно нужный axis.

**Stage план реализации:**
| Stage | Что | Effort |
|---|---|---|
| 1 | Plateau detector (per-seed loss window) | 0.5 day |
| 2 | HVP primitive + sanity test | 1 day |
| 3 | Lanczos smallest-eig (single seed first, then batched) | 1.5 day |
| 4 | Saddle step integration (only when eig < -threshold) | 0.5 day |
| 5 | Memory tuning (chunked over K) + A/B test | 1 day |
| 6 | Confirmed WIN + submission run | 0.5 day |

Итого 5 days. Если работает — ожидаем -10 до -30% на ibm01 (сходимость к ~0.76 как у vmallela).

---

## 1. Что мы учитываем — и насколько правильно

### Учитываем корректно

- **Smooth HPWL через LSE по пинам** с cooling γ — стандарт DREAMPlace, эта часть в порядке.
- **ePlace FFT density** (Poisson на grid 128) — правильная electrostatic-аналогия, лучше bell-curve.
- **Overlap penalty `rect_quad` только между hard pairs** — корректно (soft могут пересекаться).
- **Cluster cohesion с dynamic centroid** + β decay 5→0.001 — даёт structural prior без жёсткого anchor-loss (с которым он конфликтовал).
- **Multi-start K=384 + per-K diversity** + parallel C++ legalize всех 384 — хорошая инфраструктура.

### Учитываем слабо или неправильно

**Congestion surrogate vs TILOS reality.** Самое уязвимое место. У нас «smooth net-bbox demand → top-10% mean». TILOS считает совсем иначе:
- 2-pin nets: **L-shape routing** (одно колено), а не bbox
- 3-pin: Steiner-like
- N>3: bbox decomposition
- **Macro blockage** — hard макросы съедают routing tracks в своей области
- **Smoothing kernel** (5×5 Gaussian-like) перед top-k
- **Top-5%**, а не top-10%
- Раздельные H/V capacities через `hroutes_per_micron` / `vroutes_per_micron`

В readme прямо написано: «эксперимент с congestion-aware loss не сходился». Скорее всего, потому что наш surrogate далёк от TILOS-метрики, и оптимизатор тянет в локальный минимум surrogate'а, который ортогонален истине. Confidence: 8/10. **Это объясняет основной gap до MTK.**

**Density: ePlace FFT vs top-10%.** ePlace минимизирует total electrostatic potential. TILOS считает **top-10% densest cells**. Это разные функционалы — наш меньше штрафует «равномерно плохое» распределение, чем TILOS. Hybrid (ePlace для spreading + soft-top-k для refinement) может дать выигрыш.

**Net weights.** Нужно проверить, попадает ли `benchmark.net_weights` в `build_wl_pkg_full` и в congestion surrogate. Если все nets с весом 1.0 — теряем сигнал, что критические net'ы важнее.

**Pin offsets.** В TILOS `pin_pos = macro_center + pin_offset`. Если в нашем smooth WL мы считаем `macro_center` без offset (что грубее), HPWL-bbox systematically отличается. Особенно для крупных макросов с пинами по краям — это десятки μm разницы.

**Overlap penalty не учитывает congestion.** Overlap rect_quad одинаковый везде. Реально overlap в congested зоне «дороже», т.к. блокирует ещё и routing. Можно weighted: `overlap_w(cell) = base + α × congestion(cell)`.

**Soft в density bell.** Все макросы (hard+soft) одинаково контрибьютят в density penalty. Но soft — абстракция кластеров стандартных ячеек, которые в реальности «разливаются». Возможно их надо weighted с меньшим charge или разнести в отдельный density grid.

---

## 2. Что не учитываем вообще

| Что | Почему важно | Trade-off |
|---|---|---|
| **Klein-4 orientations** (N/FN/FS/S) | Sidecar `orientations.pt` опционален; +1-3% на ORFS pin-access. На Tier 1 влияет через pin offset → HPWL/congestion. | Дешёвый эксперимент: после legalize попробовать 4 ориентации каждого hard, оставить лучшую. |
| **Boundary affinity** | Макросы с большим количеством IO-пинов лучше у краёв canvas. | Loss term `Σ pin_io_count_i × distance_to_nearest_boundary(macro_i)`. |
| **Net criticality** | High-weight net'ы должны быть короче. | Просто умножить slot в smooth WL на `net_weight`. |
| **Adaptive overflow-based schedule** (DREAMPlace 4.0) | У нас линейный λ growth. У DRP — `coef = 10^((overflow - stop_overflow) × β)`. Это и есть «smoothly steer out of local traps» из MTK-комментария. | Заменить `LAMBDA_GROWTH` на overflow-driven update. |
| **PDN ≥12 μm clearance** | На Tier 2 evaluator раздвигает за нас, но это меняет proxy. Лучше сразу резервировать. | Пост-обработка: после legalize iteratively push apart до 12 μm. |
| **Symmetry constraints** | Парные SRAM banks часто лучше зеркально. | Detection через name regex и size matching. |
| **Macro blockage в congestion** | Hard macro = 0 routing capacity внутри. Наш surrogate этого не видит. | `routing_capacity[cell] -= sum(macro_area_in_cell)`. |
| **Pin-aware HPWL** | См. п.1. | Поправка `pin_pos = center + pin_offset` в LSE. |

---

## 3. Научные статьи и что из них взять

### Прямые апгрейды gradient pipeline (высокий ROI)

**AutoDMP (Agnesina et al., NVIDIA, ISPD 2023)** — https://github.com/NVlabs/AutoDMP
Bayesian multi-objective optimization над hyperparams DREAMPlace. Pareto front (WL, density, congestion) из ~30 точек. На leaderboard это #7 «Archgen» = 1.3479. Применимо к нам напрямую: запустить MOBO над нашими `STRAPLE_BATCH_*` env vars (target_util, gamma_frac, lambda_max, cong_w, cohesion_start). Confidence: 9/10, expected gain: 2-5% AVG17.

**Spindler & Johannes (DAC 2007) "Fast and Accurate Routing Demand Estimation"** — RUDY metric.
RUDY = `Σ_nets (bbox_perimeter / bbox_area) × indicator(cell ∈ bbox)`. Коррелирует с реальной congestion лучше bbox demand, дифференцируема через smooth indicator. Это **прямая замена** нашему `cong_w` term'у. Confidence: 8/10.

**Lin et al. "DREAMPlace 4.0" (TCAD 2021)** — adaptive subgradient ascent для density_weight. Псевдокод доступен. Скорее всего, MTK ровно это и использует (его коммент про «smoothly steer out of local traps» — буквальная цитата DREAMPlace overflow algorithm). Confidence: 9/10.

**Chu & Wong "FLUTE" (TCAD 2008)** — fast lookup-table Steiner trees. Для нетов с >2 пинами FLUTE даёт **истинную Steiner length**, а HPWL — overestimate на 10-20%. Использовать FLUTE как target в loss даст более точное направление gradient'а. Однако FLUTE недифференцируема — можно использовать как **importance weight per net**: `w_net *= flute_length / hpwl_estimate`. Confidence: 6/10.

### Better init / structural constraints

**Hu & Marek-Sadowska "FastPlace" (ISPD 2005)** — quadratic placement: solve `Lx = b` где L — Laplacian netlist. Даёт WL-optimal placement без overlap-aware фазы за O(N) через CG. Один K seed из spectral init = бесплатная WL-baseline.

**Karypis et al. "hMETIS" (DAC 1999)** — multi-level hypergraph partitioning. Лучше Louvain для placement, потому что Louvain на clique expansion теряет hypergraph структуру. На больших benches (n>2000) разница заметна.

**Boyd et al. "Optimal Transport for placement"** — Hungarian / Sinkhorn для matching clusters to anchor positions (вместо random grid). Минимизирует inter-cluster WL bottom-up. Confidence: 7/10.

### Better congestion (наш самый слабый компонент)

**Pan et al. "GeniusRoute" (ICCAD 2019)** — neural net предсказывает routing congestion из placement features. **Идея**: train small CNN/GNN на `(placement, true_TILOS_congestion)` парах из наших K=384 runs, использовать как differentiable surrogate в loss. По сути learned drop-in replacement нашего `cong_w` term'а. Дорого по dev time, но потенциально mid-game changer. Confidence: 6/10.

**Cheng et al. "RePlAce" (TCAD 2018)** — bivariate density penalty + Nesterov. Их Nesterov формулировка с правильной step size — отдельная статья, не «просто Nesterov». В readme мы пробовали Nesterov с lr=0.3 → catastrophic. Стоит попробовать с **их** rule: `lr_k = 1 / L_k` где L — backtracking Lipschitz estimate.

### ML approaches (long-term, after submission)

**DG-RePlAce / Wireplanner** — GNN добавляет residual displacement к gradient: `pos_next = pos + lr·grad + α·GNN(pos, netlist)`. Train на парах `(pos_t, displacement_to_best_seed)`. Plateau escape без random teleports. В readme есть секция 13 с этой идеей.

**Mirhoseini et al. (Google Nature 2021)** + Cheng et al. re-evaluation (CACM 2023). RL не воспроизводится, но идея «agent предлагает destroy operator» в LNS — рабочая (как TAISPlAce с Thompson Sampling).

---

## 4. Идеи, которых нет в плане

### Quick wins (день каждая)

1. **Top-k smooth density вместо ePlace-only**: `density_loss = soft_topk(cell_density, k=top_10pct)`. Прямо матчит TILOS. Hybrid с ePlace: `λ_eplace × eplace + λ_topk × topk`.

2. **L-shape routing surrogate** для 2-pin nets. Для каждого 2-pin net: вместо bbox считаем sum demand вдоль L-shape (right-then-up или up-then-right, soft choose через sigmoid). Больших N-pin нетов мало, mostly это и нужно.

3. **Macro-as-blockage в congestion**: `cell_capacity[c] -= overlap_area(macro, c) × routes_per_micron`. Differentiable через smooth overlap.

4. **Net-weight propagation**: проверить и пробросить `benchmark.net_weights` во все loss terms. Если сейчас все 1.0 — это бесплатный сигнал.

5. **Pin offsets в smooth WL**: `pin_pos = pos[macro_id] + pin_offset` вместо `pos[macro_id]`. Просто переписать `_build_net_pin_tensors_full`.

### Mid (2-4 дня)

6. **AutoDMP-style MOBO** над нашими env vars. Готовый Ax/BoTorch + 30-50 trials = expected -2-4% AVG17.

7. **Overflow-based λ schedule** (DREAMPlace 4.0). Простая замена growth: `coef = 10^((overflow - 0.07) × 2.2)`, `λ *= max(min(coef, 2.0), 0.5)`.

8. **Hybrid spectral+constructive init** (это есть в `improve.md` старой версии как промпт для агента — не реализовано). 1/4 K = spectral, 1/4 K = constructive Boltzmann, 1/2 K = current Louvain. Diverse basins без потери качества.

9. **Joint hard+soft с weighted density**: soft в density grid с весом 0.3-0.5 (потому что они спокойно overlap). Может вернуть density penalty в правильную область.

10. **Orientation flip optimizer**: после legalize — для каждого hard попробовать 4 Klein-4 ориентации, выбрать с минимальным local HPWL (только смежные net'ы пересчитывать). O(n_hard × 4 × avg_net_recompute), очень дёшево.

### Структурно более амбициозное

11. **Replace ePlace + cong_w + cohesion одним trained surrogate**: small NN `(macro_positions, sizes, netlist_features) → proxy_cost`. Train на `(K=384 placements, true_TILOS_proxy)` парах. Уже сейчас имеем тысячи таких пар из runs. Дифференцируемый, точно матчит TILOS.

12. **Real DREAMPlace integration через subprocess** (план C). 1.5-2 дня работы, потенциал — топ-10 проверен (UT Austin AS = plain DREAMPlace = 1.4076). С нашим LNS polish сверху может дать топ-7.

---

## 5. Гипотезы про gap MTK 0.91 vs наш 0.95 на ibm01

В убывающей confidence:

1. **Adaptive overflow-based schedule** (то самое «smoothly steer out» MTK-комментария). Confidence: 8/10. Дешёвый эксперимент.
2. **Pin-aware HPWL + net weights** правильно проброшены. Confidence: 7/10.
3. **Better congestion model** (RUDY или L-shape). Confidence: 6/10.
4. **Boundary affinity** для IO-rich макросов. Confidence: 5/10.
5. **Orientation flips после legalize**. Confidence: 5/10.

Если набрать 2-3 из этого, gap до MTK закрывается. Confidence overall: 7/10.

---

## 6. Что не делать (антипаттерны)

- **Не пытаться писать свой DREAMPlace с нуля** — месяцы работы. Бери готовый.
- **Не оптимизировать pure-gradient на ibm01 до бесконечности** — variance ±0.02 уже больше шага улучшений. Перейти к ablation на ibm14/17 для генерализации.
- **Не игнорировать verify в чистом env** — все top-3 кейса DQ (Mike Gao, BakaBobo, vmallela) случились из-за silent fail на eval-машине.
- **Не делать ансамбль gradient + ALNS как «basin escape»** — в readme явно отмечено что hybrid даёт basin trap (попадание в basin gradient'а из которого ALNS не может выйти).

---

## 7. Рекомендация в порядке приоритета

| # | Идея | Effort | Expected gain |
|---|---|---|---|
| 1 | Pin offsets + net weights проверка/фикс в smooth WL | 0.5 дня | 1-3% |
| 2 | DREAMPlace 4.0 overflow schedule для λ_d / γ | 1 день | 2-4% |
| 3 | Top-k smooth density поверх ePlace | 1 день | 1-2% |
| 4 | RUDY congestion вместо bbox demand | 2 дня | 2-4% |
| 5 | AutoDMP-style MOBO sweep | 2-3 дня | 2-5% |
| 6 | Orientation flip optimizer post-legalize | 1 день | 1-2% |
| 7 | Hybrid spectral+constructive+Louvain init | 2 дня | 1-3% |
| 8 | Real DREAMPlace integration | 3-5 дней | 5-10% |

Сумма потенциалов перекрывает gap до топ-7 (≤1.3479) с большим запасом, но они не аддитивны (часть перекрывается). Realistic: 5-8% выигрыша если сделать пп.1-5.
