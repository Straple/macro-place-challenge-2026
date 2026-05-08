# Hessian Saddle Escape — лог попыток

> Цель: реализовать у нас то, что использует **vmallela v7** (1-е место в лидерборде, AVG17=1.0109): **Hessian negative-eigenvalue saddle escape**. На ibm01 у них **0.7644**, у нас **~0.91** — gap ~16%.
>
> Этот документ — единственная точка истины по этой теме. Перед запуском нового раунда coder/reviewer должны его прочесть полностью.

---

## 1. Контекст задачи

Macro placement: разместить ~1140 макросов (1140 × 2 = 2280 переменных) на 2D-канвасе минимизируя `proxy_cost = WL + 0.5×density + 0.5×congestion`. Метрика — AVG17 на 17 IBM benchmark'ах.

Наш pipeline:
- GPU batch K=384 параллельных копий placement'а
- 500 шагов gradient descent на дифференцируемой loss за 600s
- C++ legalize (parallel mp.Pool 16 workers) для всех 384 → выбираем best

Текущий best на ibm01: **0.9040** (после Tier 3 + CD polish 4 rounds), целевой: **≤0.8**.

Лучший конфиг (trial9 baseline для всех экспериментов):
```
STRAPLE_BATCH_EPLACE=1 STRAPLE_BATCH_EPLACE_GRID=128 STRAPLE_BATCH_CONG_W=10
STRAPLE_BATCH_COHESION_START=5 STRAPLE_BATCH_COHESION_END=0.001
STRAPLE_BATCH_DIVERSITY=1 STRAPLE_BATCH_OVERLAP_FORM=rect_quad
STRAPLE_BATCH_OVERFLOW_LAMBDA=1 STRAPLE_BATCH_OVERFLOW_TARGET=0.13
STRAPLE_BATCH_OVERFLOW_EXP=0.7 STRAPLE_BATCH_OVERFLOW_COEF_HI=1.5
STRAPLE_BATCH_BLOCKAGE_W=50
```

---

## 2. Математика — почему это работает

### Седловые точки доминируют

Loss `L(pos)` в нашей задаче — non-convex в 2280 размерностях. Критические точки (∇L=0) бывают трёх типов:
1. Локальный минимум — все направления выгнуты вверх (все λ_i > 0)
2. Локальный максимум — все λ_i < 0
3. **Седло** — есть и положительные, и отрицательные λ

В высоких размерностях **седла встречаются экспоненциально чаще** локальных минимумов. Большинство «застреваний» в gradient descent — это седла.

### Hessian eigenvector = математически точный escape

`H = ∂²L/∂x∂x` (матрица 2280×2280). Её собственные значения λ_i и собственные векторы v_i описывают локальную геометрию. Если в текущей точке `λ_min(H) < 0`, то соответствующий `v_min` — direction убывания (loss уменьшается, если двигаться вдоль v_min).

Этот direction — **точный**, а не угаданный. Random teleport / shake (наш `STRAPLE_BATCH_PLATEAU_OPS`) угадывает направление наугад → часто заходит в худший basin.

### Power iteration — не считая полный Hessian

Полный Hessian — 5M чисел. Не строим явно. Используем итерационные методы которые требуют только **Hessian-vector product** (HVP) — `H·v` для произвольного v:

**FD-HVP (finite differences):**
```
H·v ≈ (∇L(pos + ε·v) − ∇L(pos − ε·v)) / (2ε)
```
Нужно 2 forward+backward. Без double-backward → меньше памяти.

**Power iteration на M:** v_new = M·v / ‖M·v‖, повторить k раз → сходится к eigenvector с **largest |eigenvalue|** of M.

**Чтобы найти smallest eig of H — нужен сдвиг:** power на `(σI − H)` где σ > λ_max(H). Largest eig of (σI − H) = σ − λ_min(H), соответствующий eigvec — это v_min(H).

**Lanczos** — точнее: строит k-мерную трёхдиагональную матрицу T через k HVP, потом её eigendecompose'им (k×k batched через `torch.linalg.eigh`). v_min(H) ≈ V[0..k-1] @ u_min(T) где V — Krylov basis.

---

## 3. Что я попытался — 6 версий saddle escape

Файл: `submissions/straple/gradient_batch.py`. Гейт: `STRAPLE_BATCH_SADDLE=1`.

### Архитектура

```python
# 1. plateau detector (per-seed loss spread tracking)
# 2. _saddle_loss closure (lightweight WL + density + overlap, без cong/anchor/cohesion)
# 3. _hvp_fd(pos, v) — FD HVP
# 4. Power iter / Lanczos для smallest eig + vector
# 5. Apply step pos += step_size * v_min для seeds с eig < threshold
# 6. Adam moments reset для перетеленных seeds
```

### Результаты всех версий (ibm01 K=384 600s, baseline trial9 = 0.9065)

| версия | алгоритм | step | threshold | плато-гейт | escaped | min | median | вывод |
|---|---|---|---|---|---|---|---|---|
| trial9 | (no saddle) | — | — | — | — | 0.9065 | 1.0616 | baseline |
| v1 | naive `-Hv` | 0.05 | -1e-6 | yes | 0 | 0.9069 | 1.0645 | плато != седло (positive eig) |
| v2 | naive | 0.05 | -1000 | no | 2 | 0.9084 | 1.0669 | very few escapes |
| v3 | **shifted** `(σI−H)v` | 0.05 | -1 | no | 57 | 0.9088 | 1.0659 | works но gain noise |
| v4 | shifted | 0.5 | -1 | gate p>0.65 | 0 (gate too late) | — | — | progress gate killed it |
| v5 | shifted | 0.5 | -1 | no | OOM | — | — | step too large fragmented memory |
| v6 | **Lanczos** k=20 | 0.1 | -1 | yes | 768 (all 384 ×2) | 0.9164 | 1.0588 | Lanczos точнее, но distribution = noise |
| v7 | Lanczos | 0.3 | -100 | yes | 768 | 0.9182 | 1.0687 | бóльший step не помог |

**Все рабочие версии (v1-v3, v6, v7) дают min/median в пределах ±0.005 от trial9** = noise level.

### Найденные баги

**Bug в v1, v2:** writing `v_new = -Hv` думая, что найдёт smallest eig of H. На самом деле power iteration on `-H` сходится к direction с largest |eig| of H (not smallest). Если λ_max=1e6, λ_min=-1e3, то power на `-H` найдёт direction для λ=-1e6 (это λ_max(H), не v_min).

**Fix (v3):** shifted iteration `v_new = σ·v − H·v` где σ оценивается через probe HVP: `σ = 2·‖H·v_random‖`.

### Почему не дало улучшения — гипотезы

1. **Paradigm mismatch.** vmallela использует **Coordinate Descent** — двигают **по одному макросу** за раз, локальный 2×2 Hessian для (x_i, y_i), eigvec — точно для **одного** макроса. Наш batch GP сдвигает все 1140 макросов одним 2280-мерным eigvec → каждый макрос движется чуть-чуть в «среднем направлении» — это **smudge**, не уход в новый basin.

2. **Adam перебивает saddle step.** После saddle step мы reset'им Adam moments. Следующий gradient step возвращает в старый basin (ту же direction что привела к saddle).

3. **Trigger timing.** Plateau detection (rel_spread < 0.005) триггерит в transition P1→P2 (steps 240-300, progress 0.4-0.5), когда landscape ещё активно меняется — не «реальное» седло, а transient «тонкое место». В P3 settling plateau появляется редко.

4. **Step size mismatch.** step=0.05·canvas_min ≈ 1 μm на 23μm canvas — слишком мало для escape. step=0.5 → OOM. step=0.3 → marginal.

5. **Hessian eigenvalues очень разнокалиберные.** В P3 у нас λ_o (overlap penalty weight) = 12000-26000, λ_d = 6 (низкий), cong_w = 10. Hessian eigenvalues могут варьироваться от -3e6 до +1e7. Power iter / Lanczos выходит в диапазон с поточной численной ошибкой.

---

## 4. Pivot на CD post-polish

Файл: `submissions/straple/cd_polish.py`. Гейт: `STRAPLE_BATCH_CD_POLISH=1`.

### Идея

После legalize, для каждого hard макроса i пробуем 8 окрестных позиций. Accept ту, что снижает proxy_cost без новых overlap'ов. Round-robin несколько проходов с уменьшающимся step factor.

Это **локальный аналог Hessian-проверки** — тестирует «является ли pos[i] минимумом по координатным направлениям?». Если нет → двигаемся.

### Что получилось на ibm01

Старт: proxy=0.9172 (после legalize одного run'а)

| round | step factor | улучшено макросов | proxy | время round'а |
|---|---|---|---|---|
| 1 | 0.50 | 29/246 | 0.9138 | 660s |
| 2 | 0.25 | 77/246 | 0.9092 | 1249s |
| 3 | 0.125 | 86/246 | 0.9059 | 1648s |
| 4 | 0.0625 | 111/246 | **0.9040** | 2918s |

**Финальное улучшение: 0.9172 → 0.9040 (−1.4%).**

**Critical observation:** улучшения **росли** с каждым раундом (29 → 77 → 86 → 111). Не сошлись. Меньший step → больше valid moves → больше improvements. То есть **convergence далеко** — потенциал ещё на много раундов.

### Что мешает

CD polish ОЧЕНЬ медленный:
- Round 4 = 49 минут (1602 проверки proxy_cost × 2с/вызов)
- Полный run (gradient + 4 rounds CD): 14 + 108 = **122 минуты**
- В 1ч судейского лимита не помещается

Источник медлительности: `compute_proxy_cost` вызывает TILOS plc client — single-threaded Python, ~2с/вызов. На GPU не уходит.

---

## 5. Открытые пути для атаки

### A) Ускорить CD polish 10-100× через GPU-batched proxy

`submissions/straple/gpu_proxy.py` имеет `gpu_proxy_batched` — приближённый proxy_cost для всех K seeds параллельно. Использовать его как **фильтр** — все 1968 кандидатов (246 макросов × 8 направлений) проверять за секунды на GPU, потом верифицировать только top-3 кандидата каждого макроса через точный TILOS proxy_cost.

Ожидание: per-round 30 min → 3 min. **30+ rounds в 1 час.** Возможно достаточно для пробития 0.85.

**Effort:** 4-6 часов кода. Главные точки:
- gpu_proxy_batched требует pos_K [K, n, 2] — построить batch [n_hard×8, n, 2]
- Memory: 1968 × 1140 × 2 × 4 = 18 MB tensor. На T4 OK
- Mismatch GPU vs TILOS: GPU proxy — наш approximation. Может ranking кандидатов не совпадать с точным. Решение: top-K по GPU, верификация TILOS.

### B) Per-macro 2×2 Hessian (вместо batch eigvec)

Vmallela paradigm. Для каждого hard макроса i:
1. Compute 2×2 Hessian на (x_i, y_i): через 4 directional grad differences
2. Solve 2×2 eigendecomposition (closed form)
3. Если λ_min < 0: pos[i] += step · v_min (только для этого макроса)

Cost: 4 grad calls per macro × 246 macros = 984 grad calls per trigger. На batched K=384 GPU: 984 forward+backward = ~25 минут per trigger. Слишком медленно.

Ускорение: использовать analytical second derivatives WL (только bbox-LSE term has closed form Hessian). Без cong/density (которые медленнее). Это approximation but fast.

**Effort:** 2-3 дня. Ключевая часть — корректная analytical 2×2 Hessian для WL_smooth.

### C) Batched Lanczos с restarts + line search

Расширить v6 Lanczos:
- Найти **top-3** smallest eigenvalues (не только smallest) — может v_min не главный
- **Line search** along v_min: тестировать step ∈ {0.05, 0.1, 0.3, 0.5} canvas_min, выбрать лучший по loss
- **Per-K restart** — если step не улучшил, попробовать v_2 (2-й smallest) или random

**Effort:** 1 день. Risk: всё ещё может страдать от paradigm mismatch.

### D) Pair-swap CD (2-opt)

В дополнение к single-macro CD, добавить операцию обмена pair (i, j). Может сильно reduce WL для зеркально-плохо размещённых пар. Cost: O(n_hard^2) pairs, реалистично — только spatial neighbors.

**Effort:** 1 день. Independently от Hessian / CD.

### E) Real DREAMPlace (Plan C)

Внешний state-of-art placer, верифицированно даёт ~1.41 на AVG17 (UT Austin AS) и 1.35 (AutoDMP/Archgen). С нашим CD polish сверху может дать топ-7. **Effort:** 3-5 дней. Risk silent fail.

---

## 6. Constants — измеренные характеристики

- ibm01: 246 hard macros + 894 soft, canvas 22.9×23.0 μm, 5993 nets
- T4 GPU: 16 GB VRAM, peak gradient batch ~10.2 GB (K=384, ePlace 128, cong 10)
- Saddle escape overhead: K=384, Lanczos k=20 → ~25s per trigger
- compute_proxy_cost (TILOS): ~2с/вызов CPU
- gpu_proxy_batched: ~50ms на K=384 placements
- Parallel C++ legalize all-K: 16 workers ~135s

---

## 7. Iteration workflow

Каждый раунд работы — три участника:

### 7.1. Coder

- Читает `readme/hessian.md` (этот файл) полностью
- Реализует/правит код в `submissions/straple/gradient_batch.py` или `submissions/straple/cd_polish.py` (или новый модуль)
- Pushes diff на review
- Defends/discusses with reviewer until consensus

### 7.2. Reviewer

- Читает `readme/hessian.md` полностью + coder's diff
- Критикует: math correctness, memory budget T4, runtime budget (≤30 min run), edge cases (OOM, NaN), backwards compatibility (trial9 baseline должен работать без флагов)
- Может **спорить** с coder и **оспаривать** — это поощряется
- Approves только если убедился что нет regressions и идея математически обоснована

### 7.3. Test runner (orchestrator)

- Получает approved diff
- Pushes на server (`./run_remote.sh push`)
- Запускает test: ibm01, K=384, time-budget=600s (10 мин), `--no-vis`, с experimental env-vars
- **Total wall time ≤ 30 минут** (gradient 10 + legalize 3 + saddle/CD overhead ~10 + buffer)
- Если CD polish — 1 round only (sf=0.5, dirs=4, ~7 мин)
- Если saddle escape — full Lanczos OK (overhead ~25s × несколько триггеров)
- Pulls stats, обновляет hessian.md секцию «Лог раундов»

### 7.4. Тест-команда (template)

```bash
ssh evyukhnevich@103.76.52.240 'cd macro-place && export PATH="$HOME/.local/bin:$PATH" && nohup bash -lc "
source .venv/bin/activate
export STRAPLE_BATCH_EPLACE=1 STRAPLE_BATCH_EPLACE_GRID=128 STRAPLE_BATCH_CONG_W=10 \
  STRAPLE_BATCH_COHESION_START=5 STRAPLE_BATCH_COHESION_END=0.001 \
  STRAPLE_BATCH_DIVERSITY=1 STRAPLE_BATCH_OVERLAP_FORM=rect_quad \
  STRAPLE_BATCH_OVERFLOW_LAMBDA=1 STRAPLE_BATCH_OVERFLOW_TARGET=0.13 \
  STRAPLE_BATCH_OVERFLOW_EXP=0.7 STRAPLE_BATCH_OVERFLOW_COEF_HI=1.5 \
  STRAPLE_BATCH_BLOCKAGE_W=50 \
  # ↓ experimental env-vars здесь ↓
  STRAPLE_BATCH_SADDLE=1 STRAPLE_BATCH_SADDLE_ITERS=20 \
  STRAPLE_BATCH_SADDLE_STEP=0.1 STRAPLE_BATCH_SADDLE_THRESHOLD=-1.0
uv run python scripts/gpu_run_one.py --bench ibm01 --K 384 --time-budget 600 --no-vis > .remote_runs/ROUND_NAME.log 2>&1
cp results/gpu_stats_ibm01.json .remote_runs/stats_ROUND_NAME.json
echo ROUND_NAME_DONE >> .remote_runs/ROUND_NAME.log
" > /dev/null 2>&1 < /dev/null & disown'
```

Pull результат:
```bash
scp evyukhnevich@103.76.52.240:macro-place/.remote_runs/stats_ROUND_NAME.json /tmp/
```

### 7.5. Лог раундов

Здесь обновляется после каждого test'а. Format:

```
### Round N (YYYY-MM-DD)
- **Hypothesis:** что и почему
- **Code change:** короткое описание (полный diff в git)
- **Run config:** experimental env-vars
- **Result:** min/p25/median/mean/std из stats. Saddle events / CD improvements counters.
- **Verdict:** WIN / LOSE / NOISE / OOM
- **Next:** что попробовать после
```

#### Пример (Round 0 — baseline trial9):
- Hypothesis: trial9 — отправная точка (best Tier 3 config)
- Code change: none
- Config: только trial9 базовые env-vars
- Result: min=0.9065 p25=0.9765 median=1.0616 mean=1.0485 std=0.0745
- Verdict: BASELINE
- Next: см. Round 1

#### Round 1 — Lanczos saddle escape (ранее v6)
- Hypothesis: точнее eigenvector → лучший escape direction
- Code: replaced power iteration with batched Lanczos в saddle handler
- Config: SADDLE=1 ITERS=20 STEP=0.1 THRESHOLD=-1.0
- Result: min=0.9164 median=1.0588 — 768 escapes но distribution в пределах noise vs trial9
- Verdict: NOISE (paradigm mismatch — все 384 seeds сдвигаются «в среднем направлении»)
- Next: переходить к per-macro 2×2 Hessian (path B) или GPU-accelerated CD polish (path A)

#### Round 2 — GPU-batched CD polish (path A) — 2026-05-08
- **Hypothesis:** CD polish работает (Round 0→0.9040 demo), но 1 round в 30 min budget. GPU-batched ranker через `gpu_proxy_batched` фильтрует все 1968 кандидатов (246 macros × 8 dirs) за ~50ms, потом TILOS verify только top-K → 5-10× rounds в budget.
- **Code:** новая функция `cd_polish_gpu` в `submissions/straple/cd_polish.py`. Двухстадийный фильтр: GPU rank всего chunk_size×n_dirs кандидатов (vectorized `cand_pos_full` patching через scatter), затем TILOS top-K verify. Гейт `STRAPLE_BATCH_CD_GPU_FILTER=1` в `scripts/gpu_run_one.py`. Защита от перерасхода: per-chunk и per-macro time check + global `STRAPLE_BATCH_WALL_TL` который клампит cd_time_budget по wall_remaining. Подробное логирование skip_border/skip_overlap/skip_no_improve/accept + avg/max ms per TILOS call.
- **Reviewer fixes:** sf default 1.0→0.5 (orig CD начинался с 0.5; sf=1.0 регрессия); topk_verify 2→3 (больше робастности при больших sf); proxy_chunk_n=32 (внутренний chunk gpu_proxy_batched, было 96 → OOM risk на T4).
- **Run config (Round 2v2 — после kill v1, добавлен per-chunk TL и логирование):**
  ```
  STRAPLE_BATCH_CD_POLISH=1 STRAPLE_BATCH_CD_GPU_FILTER=1
  STRAPLE_BATCH_CD_ROUNDS=4 STRAPLE_BATCH_WALL_TL=1500
  STRAPLE_BATCH_WALL_RESERVE=30
  ```
  (defaults: TOPK=3, MACRO_CHUNK=64, PROXY_CHUNK_N=32, sf=0.5,0.25,…)
- **Result:** distribution (pre-CD после legalize) min=0.9124 p25=0.9776 median=1.0634 mean=1.0483 std=0.0747. CD polish best seed 0.9124 → 0.9098 (R1 sf=0.5, 23 acc, 245 calls × 1.82s) → **0.9082** (R2 sf=0.25, 19 acc, 93 calls × 1.82s). Total CD time 622s, wall_elapsed 1475.9s (24.6 min). Только 2 rounds полных (TL clamped 617s).
- **Verdict:** NOISE для overall (0.9082 vs trial9 0.9065 = +0.0017 < 0.005). НО: CD polish demonstrated **-0.5% within-run** улучшение (0.9124→0.9082). Низкий best обусловлен худшим стартовым seed'ом (0.9124 vs trial9 0.9065 — это per-run stochasticity).
- **Что пошло не так в v1:** time_budget проверялся только перед раундом → Round 2 сf=0.25 началось при elapsed=8min, могло идти 15+ min после cap. Fix: per-chunk и per-macro проверки + global WALL_TL который дополнительно клампит cd_time_budget. v2 завершился в budget.
- **Per-call time confirmed:** 1.82s/TILOS call (consistent с hessian.md "~2с/вызов") — НЕ 0.34s как ранее предполагалось reviewer'ом из старых logs. Per-round при sf=0.25: ~93 calls × 1.82s = 170s (быстрее чем sf=0.5 потому что больше rejected by border/overlap).
- **Skip breakdown sf=0.5:** 152 border + 1451 overlap + 222 no_improve / 23 accept (1968 cand attempts → 245 actual TILOS calls due to rate of skip_overlap dominance).
- **Next:**
  - Multi-seed validation: повторить run 2-3 раза чтобы определить distribution variance vs CD-improvement signal.
  - Tighter integration: применить CD polish к top-N seeds (не только best) и взять best of N polished.
  - Path C ablation: Lanczos restart + line search на per-macro level (combined с CD finalization).

---

## 8. Iteration prompt (для coder/reviewer/orchestrator triad)

Скопировать этот блок целиком при запуске нового раунда работы:

```
ROUND N: попытка №N пробить ibm01 best ≤0.85 (vmallela=0.7644, текущий best=0.904).

ПЕРВЫЙ ШАГ КАЖДОГО АГЕНТА — прочесть readme/hessian.md ПОЛНОСТЬЮ. Не пропускать секции; знание прошлых ошибок предотвращает повторение их.

РОЛИ:

1. CODER:
   - Читает hessian.md
   - Выбирает один path из секции 5 (A/B/C/D) ИЛИ предлагает новую гипотезу
   - Реализует код в submissions/straple/* или scripts/gpu_run_one.py
   - Гейтит через env-var (don't break trial9 default behavior)
   - Пишет короткий design note: что меняем, почему, ожидаемый эффект
   - Передаёт reviewer'у

2. REVIEWER:
   - Читает hessian.md + coder's diff + design note
   - Критикует. Точки внимания:
     * Math correctness (особенно для Hessian/Lanczos/HVP — легко ошибиться с signs, indices)
     * Memory: T4 16GB — peak forward = 10.2 GB, остаётся ~5.8 GB для дополнительных tensors
     * Runtime: total ≤30 мин wall time. Forward step ~1.5с. HVP × 2 forward = ~3с. Lanczos k=20 = ~60с/trigger.
     * Backward compat: trial9 без новых env-vars должен работать как раньше
     * Edge cases: NaN propagation, division by zero, OOM при per-K resize
   - Может спорить, оспаривать coder. Это нормально и поощряется. Хорошее решение растёт из спора.
   - Approves только если уверен. Иначе возвращает coder'у с конкретными requirements.

3. ORCHESTRATOR (тот кто следит):
   - Координирует диалог coder ↔ reviewer (через SendMessage если subagents)
   - После approval: pushes код, запускает test (template из секции 7.4)
   - Wall time ≤ 30 минут на ibm01 K=384 time-budget=600
   - Если CD polish — ограничить ROUNDS=1 и SF=0.5 чтобы не переполнить time budget
   - Pulls stats, добавляет в hessian.md секцию «Лог раундов» новую запись с результатами
   - Обновляет «открытые пути» если path исчерпан или новый открылся

CONSTRAINTS:
- Не трогать baseline trial9 поведение без env-флага
- Не делать destructive actions (удаление файлов, force push)
- Не сидеть >30 мин на тесте — лучше fail fast и анализировать
- Каждый код-эксперимент должен быть гейтаемый через STRAPLE_BATCH_* env

GOAL за раунд: либо WIN (новый best ≤ предыдущего на >0.005 = вне noise), либо conclusive LOSE (с пониманием почему). NOISE-результат = недостаточно — нужно либо tune, либо менять подход.

После раунда обновить hessian.md и предложить план следующего раунда.
```

---

## 9. Quick reference — что НЕ делать

- ❌ Power iteration `v_new = -H·v` для smallest eig — найдёт largest |eig|. Используйте shifted `(σI − H)v` или Lanczos.
- ❌ Step size 0.5·canvas_min при K=384 — OOM на T4. Ограничивать ≤0.2.
- ❌ Plateau gate при saddle escape — plateau ≠ saddle (плато может быть локальным минимумом с λ_min > 0). Лучше gate по `λ < threshold`.
- ❌ CD polish full 4 rounds dirs=8 в 1 ч лимит — не помещается. Use 1 round dirs=4 sf=0.5 для быстрых iterations.
- ❌ Trust но verify: distribution из stats имеет variance ±0.02 между runs — single-best win может быть noise. Сравнивать median/p25/mean.
