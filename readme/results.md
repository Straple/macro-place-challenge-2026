# 📊 Результаты замеров

> Полный журнал прогонов на нашем железе. Краткая сводка — в [todo.md](todo.md#5-журнал-экспериментов).

---

## 🏆 Текущий best score

| | Значение |
|---|---|
| **AVG4 proxy** (ibm01/10/14/17) | **1.4688** ⭐ #11 |
| AVG17 proxy (--all) | 1.5181 (#4, перезамерить) |
| Placer | `submissions/straple/placer.py` (C++ + adaptive LNS + чередование congested/random) |
| Дата замера | 2026-05-03 |
| Wall time (fast_check) | ~90-115с на 4 бенчах parallel |
| Hardware | Mac ARM (aarch64-apple-darwin), Python 3.14.4, torch 2.10.0 |

**Гэп до целей:**

| Цель | Score | Δ от текущего | % улучшения нужно |
|---|---|---|---|
| RePlAce baseline (порог) | 1.4578 | -0.0603 | **-4.0%** |
| Топ-10 leaderboard | 1.4076 | -0.1105 | -7.3% |
| **Топ-7 → Гран-при $20K** | **1.3479** | **-0.1702** | **-11.2%** |
| Топ-3-5 | ~ 1.32 | -0.20 | -13% |
| Первое место (Cezar) | 1.2224 | -0.2957 | -19.5% |

---

## 📂 Журнал прогонов

### #11 · 2026-05-03 · `submissions/straple/placer.py` (multi-start N=3 для больших) — НОВЫЙ BEST 🏆

**Идея**: для больших дизайнов (`num_movable >= 300`) запустить 3 независимых старта с разными seed (42/43/44) полным циклом legalize→SA→LNS, выбрать минимальный по proxy_cost. Маленькие (ibm01) — один старт, как раньше (overhead не оправдан).

**Источник**: ALNS / SA literature — multi-start как стандарт diversification. На больших дизайнах разные seed → разные attractor'ы.

**Изменения**:
- `submissions/straple/placer.py` `StraplePlacer.place()`: цикл по `num_starts = 3 if num_movable >= 300 else 1`. `_build_proxy_evaluator` строится один раз и переиспользуется. Best по trial_cost.

**Сводка** (медиана 3 запусков, 3/3 идентичны):

| Bench | Proxy | vs #8 (1.4707) | vs Straple #4 baseline | starts |
|---|---|---|---|---|
| ibm01 | 1.1411 | 0.00% | -3.14% ⭐ | 1 (skip multi-start) |
| ibm10 | **1.3828** | **-0.53%** | -0.11% ⭐ | 3 (best of 3) |
| ibm14 | 1.6092 | 0.00% | -1.15% ⭐ | 3 (1st seed = best) |
| ibm17 | 1.7422 | 0.00% | -0.17% | 3 (1st seed = best) |
| **AVG4** | **1.4688** ⭐ | **-0.13%** | **-1.02%** ⭐ | wall ~88-96s |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+3.46%** (хуже, продолжаем приближаться)
- vs прошлый best AVG4 (#8: 1.4707): **-0.13%** ▼ (новый best)
- vs Straple #4 baseline (1.4839): **-1.02%** ▼ — впервые превышаем -1%
- Overlaps: 0, Smoke 9/9

**Что сработало**:
- ibm10 (387 макросов) — впервые улучшение, **теперь меньше baseline**: 1.3828 vs RePlAce 1.4928 (-7.4% от RePlAce!) — обходим RePlAce на этом бенче.
- Реализация чистая: один общий evaluator, минимальный overhead.
- Время растёт пропорционально (ibm10/14/17 ×3). На fast_check всё ещё <100с total wall (parallel).

**Что не сработало**:
- ibm14 и ibm17: первый seed оказался best (или все 3 сходятся в одну точку). Multi-start не дал diversification. Гипотеза: на ibm14/17 локальный минимум очень глубокий (basin of attraction большой), все 3 seeds валятся в него.

**Главный урок**:
- Multi-start работает на ibm10, но не на ibm14/17. На очень больших дизайнах одного multi-start недостаточно — нужно больше N или structurally разные initial configurations (например, force-directed + grid-based).
- Размер basin of attraction зависит от netlist — на ibm10 он маленький, easy to escape.

**Следующие шаги**:
- Cycle #8: попробовать N=5 или больше (runtime budget огромный).
- Или: structurally разные seeds — random scatter vs grid vs centroid-based initial.
- Или: best result оставить + ещё короткий run проксированный с heavy LNS.

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`

---

### #10 · 2026-05-03 · попытка: wider congested_percent 5%→10% — НЕЙТРАЛ, откатили

**Идея**: расширить top-N% hot-cells с 5% до 10%, дать destroy-оператору больше таргетов.

**Результат**: AVG4 1.4712 (vs #8 1.4707, +0.03% — нейтрал, в шуме). ibm01-17 без значимых изменений.

**Урок**: ширина hot-cells не bottleneck — даже с 10% top хватает destroy-таргетов. Реальный bottleneck в чём-то другом (структурном). Откатили.

---

### #9 · 2026-05-03 · попытка: tighter threshold (P30 вместо median) — РЕГРЕССИЯ, откатили

**Идея**: В `repairMacroAware` заменить median (P50) на P30 — более жёсткий threshold. Гипотеза: на ibm10 (cong-grid гладкий) median всегда срабатывает, поведение = старое; tighter threshold заставит spiral искать действительно cool зоны.

**Результат** (1 fast_check):

| Bench | Proxy | vs #8 (1.4707) | Что |
|---|---|---|---|
| ibm01 | 1.1467 | +0.49% ❌ | density 0.914→0.921 |
| ibm10 | 1.3902 | 0.00% | как и было |
| ibm14 | 1.6149 | +0.35% ❌ | density 0.980→0.982 |
| ibm17 | 1.7436 | +0.08% | нейтрал |
| **AVG4** | **1.4739** | **+0.22% ❌** | — |

**Что не сработало**: P30 слишком жёсткий — fallback на старый repair срабатывает чаще, теряем aware-эффект. На ibm10 действительно ничего не изменилось (как и предсказывали), но на ibm01/14 — регрессия.

**Урок**: median (P50) — оптимальный threshold для текущей архитектуры. Жёстче = больше fallback. Реверт через `git checkout placer_core.cpp`.

**Гипотеза для cycle #6**: атаковать ibm10/17 не через threshold, а другим способом — например, **wider congested-percent** (top 10% вместо top 5%, больше hot-cells → больше destroy-таргетов). Или: **scale outer iterations specifically для ibm17** (lift upper bound 60→80).

---

### #8 · 2026-05-03 · `submissions/straple/placer.py` (congestion-aware repair) — НОВЫЙ BEST 🚀

**Идея**: На congested-destroy ветке LNS использовать **congestion-aware spiral search** в repair: принимать non-overlap позицию ТОЛЬКО если её max-cong по перекрытым cells < median(cong-grid). Если за spiral (r∈[1..20]) не нашли — fallback на старый «слепой» repair. Random-destroy ветка не трогается (exploration сохранена).

**Источник**: прямой follow-up из #5 («repair away from hot»). Теория ALNS: destroy + domain-aware repair > destroy + blind repair.

**Изменения**:
- `submissions/straple/cpp/proxy_cost.cpp` — новый pybind метод `getCongestionGrid()` экспонирует кэш cong-grid (hRouting+vRouting) как np.array shape [gridRows, gridCols].
- `submissions/straple/cpp/placer_core.cpp` — новый `repairMacroAware(state, idx, congGrid, threshold)` — клон существующего `repairMacro` со cong-проверкой trial-позиций. Spiral ограничен r∈[1..20] (вместо 80) для перформанса. Fallback на repairMacro при non-найдено. `destroyCongestedAndRepair` принимает новые параметры congGrid и зовёт repairMacroAware.
- `submissions/straple/placer.py` `_lns_loop` — на нечётной (congested) итерации передаёт `cong_grid` в C++; на чётной (random) не трогает.

**Сводка** (медиана 3 запусков, 3/3 идентичны):

| Bench | Proxy | vs #7 (1.4795) | vs Straple #4 (1.4839) | Что улучшилось |
|---|---|---|---|---|
| ibm01 | **1.1411** ⭐ | **-1.91%** | **-3.14%** ⭐ | density 0.923→0.914, cong 1.258→1.222 |
| ibm10 | 1.3902 | 0.00% | +0.43% | нейтрал (random ветка не менялась) |
| ibm14 | **1.6092** | **-0.71%** | **-1.15%** ⭐ | density 0.995→0.980, cong 2.140→2.130 |
| ibm17 | 1.7422 | -0.09% | -0.17% | cong 2.427→2.425 |
| **AVG4** | **1.4707** | **-0.59%** ⭐ | **-0.89%** ⭐ | wall ~88-91s |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+3.59%** (хуже, но впервые ближе чем +4%)
- vs прошлый best AVG4 (#7: 1.4795): **-0.59%** ▼ — большой шаг
- Overlaps: 0, Smoke 9/9, Build OK без warnings

**Что сработало**:
- ibm01 -1.91% и ibm14 -0.71% — congestion-aware repair размещает макросы в зоны с реально низкой нагрузкой, не «первое же свободное место рядом». Параллельно падает density (макросы распределяются равномернее).
- Fallback архитектура работает: на ibm01 (где hot-зон мало, threshold редко срабатывает) репэйр в нужный момент откатывается на старое поведение.
- Минимальный runtime overhead (+1-3%).

**Что не сработало**:
- ibm10 0% (нейтрал): congested-итерации возможно вообще не находили threshold-удовлетворяющие позиции, fallback'или на старое. Гипотеза: cong-grid у ibm10 более «гладкий» (cong=1.92, низкий) — median high, threshold почти любой trial проходит → behavior идентичен старому.
- ibm17 -0.09% — почти ничего. Главный bottleneck: на ibm17 макросов 517, congested-destroy на 26 итерациях × 13 = 338 переразмещений; если каждое placeholder в hot-зоне даёт лишь 1-2% локального улучшения, на global proxy эффект -0.5-1%. Возможно нужны ДВЕ вещи: (1) больше congested-итераций на больших, (2) более узкий threshold (quantile 0.7 вместо median).

**Главный урок**:
- Domain-aware repair > blind repair (как в ALNS-литературе).
- Fallback архитектура ключевая для robustness — не ломает маленькие, помогает большим.
- ibm01 и ibm14 теперь **в зоне RePlAce-level** (на ibm01 наш 1.14, RePlAce 1.00 → +14%; на ibm14 наш 1.61, RePlAce 1.54 → +4.4%).

**Следующие шаги**:
- Cycle #5: попробовать **более узкий threshold** (quantile 0.7 / 0.6 вместо median) — для ibm10/17, где median не срабатывает.
- Или: scale congested-percent (топ 5% → топ 10%) для большей разрядки на ibm17.
- Или: multi-start для ibm17.

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`

---

### #7 · 2026-05-03 · `submissions/straple/placer.py` (adaptive LNS + чередование) — best после cycle #3

**Идея**: Минимальный фикс к #6. Вернуть чередование random/congested (exploration важна, особенно для ibm01), но СОХРАНИТЬ adaptive_destroy и adaptive_outer от #6.

**Источник**: прямой вывод из cycle #6 (где удаление чередования дало +2.5% регрессии на ibm01).

**Изменения**:
- `submissions/straple/placer.py` `_lns_loop`: восстановлено `if iteration % 2 == 1:` для congested branch, на чётных — random destroy. `adaptive_destroy=clamp(8, ceil(0.025·N), 16)` и `adaptive_outer=clamp(30, ceil(0.10·N), 60)` остались от #6.

**Сводка** (медиана 3 запусков, 3/3 идентичны — детерминизм):

| Bench | Proxy | vs #5 (1.4822) | vs Straple #4 baseline | params (k, outer) |
|---|---|---|---|---|
| ibm01 | **1.1633** | **-0.37%** | -1.26% ⭐ | k=8, outer=30 |
| ibm10 | 1.3902 | +0.09% | +0.43% | k=10, outer=39 |
| ibm14 | **1.6207** | **-0.45%** ⭐ | -0.45% ⭐ | k=12, outer=46 |
| ibm17 | 1.7437 | -0.03% | -0.08% | k=13, outer=52 |
| **AVG4** | **1.4795** | **-0.18%** | **-0.30%** ⭐ | wall=88-115s |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+4.21%** (хуже, но ближе всех)
- vs прошлый best AVG4 (#5: 1.4822): **-0.18%** ▼ (новый best)
- vs Straple #4 baseline (AVG4=1.4839): **-0.30%** ▼ (превосходит исходный)
- Overlaps: 0
- Smoke 9/9 PASS, build OK без warnings

**Что сработало**:
- Чередование сохранило exploration на маленьких (ibm01: 1.1676 → 1.1633, ещё -0.37%).
- Adaptive params на больших дали реальный эффект **впервые на ibm14** (-0.45%): congestion 2.150 → 2.140. Большее число итераций (46 vs 30) позволяет глубже разрабатывать area.
- ibm17 нейтрал (-0.03%) — adaptive params не повредили, но и не помогли. Возможно нужен ещё больший budget, или проблема не в количестве итераций.
- Время растёт умеренно: ibm17 2.64s vs 2.59s в #5 (+1.9%) — wall ~88-115с (variability fork).

**Что не сработало**:
- ibm10 +0.09% (микро-регрессия, в шуме). Гипотеза: для ibm10 (387 макросов, k=10) лимит 0.025 даёт мало преимущества, а 0.10 outer = 39 уже близко к минимуму 30. Возможно стоит выше lower bound для outer.
- На ibm17 эффект минимальный — возможно нужен другой подход (repair-aware-of-hot, multi-start).

**Главный урок**:
- Чередование + adaptive — комбинация работает.
- Adaptive params **сами по себе** дают эффект ТОЛЬКО при сохранении exploration (random destroy).
- Подтверждена ALNS-литература: domain-aware operators хорошо работают в комбинации с random для diversification.

**Следующие шаги**:
- Cycle #4: попробовать **repair-aware-of-hot** — в spiral search избегать high-cong cells (адресует ibm17).
- Или: multi-start с 3-5 разных seed'ов на больших.
- Или: tune adaptive lower bounds (выше floor для outer на средних).

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`

---

### #6 · 2026-05-03 · `submissions/straple/placer.py` (adaptive LNS, без чередования) — РЕГРЕССИЯ

**Идея**: На больших дизайнах дать LNS реальную мощность по разгрузке congestion: (1) убрать чередование random/congested — всегда congested (random — fallback в C++ при пустом overlap); (2) `adaptive_destroy = clamp(8, ceil(0.025·N), 16)`; (3) `adaptive_outer = clamp(30, ceil(0.10·N), 60)`. Бонус: `nth_element` вместо `sort` в C++ для top-k.

**Источник**: собственная гипотеза + 2 из 3 follow-up'ов из #5.

**Результаты** (1 fast_check):

| Bench | Proxy | vs #5 (1.4822 baseline) | params (k, outer) |
|---|---|---|---|
| ibm01 | 1.1964 | **+2.47%** ⬆ ❌ | k=8, outer=30 |
| ibm10 | 1.3940 | +0.36% | k=10, outer=39 |
| ibm14 | 1.6286 | +0.03% | k=12, outer=46 |
| ibm17 | 1.7441 | -0.01% | k=13, outer=52 |
| **AVG4** | **1.4908** | **+0.58%** | — |

- vs Straple #4 baseline (AVG4=1.4839): **+0.46%** (хуже)
- vs RePlAce: +5.01% (хуже)

**Что сработало**:
- Реализация технически корректная, build OK, smoke 9/9, 0 overlaps.
- Даже на ibm17 60×13 итераций успевают за 2.82с (раньше было 30×8 за 2.59с) — масштабирование adekvatное.

**Что не сработало**:
- **ibm01: -0.89% улучшение #5 → +2.47% регрессия** (= откат на хуже-чем-baseline). Параметры на ibm01 не изменились (k=8, outer=30 — те же), значит **виновник — удаление чередования**. Random destroy служил exploration; congested-only застрял в локальном.
- На ibm10/14/17 increased k/outer **не помог** — гипотеза «больше work на больших → лучше» опровергнута. Скорее всего: (а) при k=12-13 разрушается слишком много макросов одновременно, repair (centroid+spiral) не успевает качественно переразместить; (б) congested-only без exploration легко зацикливается на одних и тех же hot-cells.

**Главный урок**:
- **Чередование random/congested оказалось важной диверсификацией** — нельзя убирать.
- Adaptive params БЕЗ exploration не дают эффекта.
- На больших нужны другие подходы — может быть multi-start или repair-aware-of-hot, не наращивание выборки.

**Следующие шаги**:
- Cycle #3: вернуть чередование, попробовать только adaptive params (k и outer).
- Альтернатива: repair-aware-of-hot (в spiral search избегать high-cong cells).

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`

---

### #5 · 2026-05-03 · `submissions/straple/placer.py` (congestion-aware destroy в LNS)

**Идея**: чередовать в LNS-петле два destroy-оператора — старый random и новый **congestion-aware**: выбирать k макросов, которые перекрывают top-5% самых перегруженных congestion-ячеек (взвешено по площади перекрытия и ценности ячейки).

**Источник**: собственная гипотеза, опираясь на декомпозицию cost (congestion = 66% AVG) и на принцип Ropke & Pisinger 2006 (ALNS — domain-aware destroy operators).

**Изменения**:
- `submissions/straple/cpp/proxy_cost.cpp` — кэш cong-grid после `evaluate()`, новый метод `getTopCongestedCells(percent)` возвращает Nx3 (row, col, weight).
- `submissions/straple/cpp/placer_core.cpp` — новый `destroyCongestedAndRepair(positions, hot_cells, gridRows, gridCols, k)`: считает overlap каждого movable макроса с hot-cells, выбирает top-k и репэйрит существующим weighted-centroid + spiral search. Fallback на random если ни один макрос не перекрывает hot-зону.
- `submissions/straple/placer.py` — `_lns_loop` чередует random/congested destroy по чётности итерации; congested_percent=0.05.

**Сводка** (медиана из 3 fast_check запусков, ibm01/10/14/17, parallel 4 workers):

| Bench | Proxy | WL | Density | Cong | Time | vs Straple #4 |
|---|---|---|---|---|---|---|
| ibm01 | **1.1676** ⭐ | 0.073 | 0.927 | 1.262 | 0.40s | **-0.89%** |
| ibm10 | 1.3890 | 0.070 | 0.717 | 1.921 | 1.74s | +0.34% |
| ibm14 | 1.6281 | 0.053 | 1.000 | 2.150 | 1.92s | +0.01% |
| ibm17 | 1.7442 | 0.054 | 0.952 | 2.428 | 2.59s | -0.05% |
| **AVG4** | **1.4822** | — | — | — | wall=105s | **-0.11%** |

- vs RePlAce (на 4 бенчах AVG=1.4197): **+4.40%** (хуже)
- vs Straple #4 baseline (AVG4=1.4839): **-0.11%** ▼ (микроулучшение, в пределах шума 0.5%)
- Overlaps: 0
- Bounds-violations: pre-existing на ibm10/14/17 (есть и в baseline, не наша регрессия — см. "Известные проблемы" в [improve.md](improve.md))

**Что сработало**:
- Заметное улучшение на **ibm01** (-0.89%): congestion=1.284→1.262 (-1.7%), density=без изменения, WL=без изменения. Congestion действительно снижается на маленьком дизайне, где hot-cells более локализованы.
- Build чистый, smoke 9/9, 0 overlaps, fallback на random при отсутствии overlap корректно работает.
- Реализация **детерминирована** (3 запуска идентичны).

**Что не сработало**:
- На больших ibm10/14/17 эффект **в шуме (~±0.05%)**. Гипотеза: при большом количестве макросов (387-517) top-5% hot-cells не локализуют bottleneck — 30 LNS-итераций × 4 destroy = 120 макросов перемещено из ~500, недостаточно для глобальной разгрузки.
- Чередование с random может разводить улучшения (random destroy «возвращает» макросы обратно в hot-зону).

**Что попробовать дальше**:
- Чисто congestion destroy (не чередовать). Сравнить.
- Увеличить destroy_size для больших дизайнов (4 → 8-12).
- Larger LNS budget на больших (60-100 итераций вместо 30).
- "Repair away from hot": в spiral search избегать high-cong cells, не просто двигать в любую non-overlap позицию.

**Команда**: `$HOME/.local/bin/uv run python scripts/fast_check.py`
**Reviewer verdict**: ACK (5 minor notes — кэш macroCong неиспользуется, nth_element вместо sort, namedconst для 0.05, last_was_accepted флаг, унификация fallback shape).

---

### #4 · 2026-05-03 · `submissions/straple/placer.py` (pure C++: placer_core + proxy_cost)

**Что менялось от #3:**
- Реализован полный `compute_proxy_cost` на C++ (`cpp/proxy_cost.cpp`):
  - `get_wirelength`: HPWL по pin bbox каждого нета
  - `get_density_cost`: top-10% самых плотных grid-ячеек
  - `get_congestion_cost`: routing 2/3/N-pin + macro-route-over-grid + smoothing + ABU top-5%
- Python теперь только тонкая обёртка: извлекает pin connectivity из plc и передаёт в `_proxy_cost.ProxyEvaluator`
- LNS inner loop звонит `evaluator.evaluate(positions)` напрямую в C++ — без вызовов Python plc

**Результаты:**
- AVG proxy: **1.5181** (бит-в-бит совпадает с #3, т.е. наш C++ exact replicates plc.compute_proxy_cost)
- Total runtime: **16.19s** (vs 173.81s в #3 = **10.7× быстрее**)
- Best ibm01: **0.34s** (vs 3.4s)
- Slowest ibm17: 2.15s (vs 24.5s)

**Как это влияет на стратегию:**
- Раньше LNS ограничен ~30 itters (45с/bench overhead). Теперь можно гонять 300+ itters за то же время.
- Можно делать multi-restart с 5-10 сидов за разумное время.
- Можно делать proxy-aware SA (proxy_cost внутри inner SA loop).

**Команда:** `$HOME/.local/bin/uv run evaluate submissions/straple/placer.py --all`

---

### #3 · 2026-05-03 · `submissions/straple/placer.py` (C++ core via pybind11)

**Сводка:**
- AVG proxy: **1.5181**
- Best: **1.1781** на `ibm01` (246 макросов)
- Worst: **1.7944** на `ibm06` (318 макросов)
- Total runtime: **173.81s**
- Overlaps: 0 на всех ✅
- vs SA baseline: **+28.6%** (better)
- vs RePlAce baseline: **-4.1%** (хуже, но ближе чем will_seed)
- vs will_seed: **-1.01%** (lower is better)

**Архитектура:**
- C++ ядро (`submissions/straple/cpp/placer_core.cpp`) — legalize, SA refinement, LNS destroy/repair
- Python обёртка — extract edges из плк, оркестрирует пайплайн, оценивает proxy_cost через TILOS plc
- pybind11 bindings, компилируется через `cpp/build.sh`
- monkey-patch `plc.{soft,hard}_macro_pin_indices` → set (28× ускорение `compute_proxy_cost`)

**Параметры:**
- refine_iters=3000 (SA на HPWL)
- lns_outer_iters=30
- lns_destroy_size=8 (random destroy), repair=weighted-centroid + spiral search

**Per-benchmark детали:**

| Benchmark | Macros | Proxy | WL | Density | Congestion | Time | vs SA | vs RePlAce |
|---|---|---|---|---|---|---|---|---|
| ibm01 | 246 | **1.1781** ⭐ | 0.073 | 0.927 | 1.284 | 3.4s | +10.5% | -18.1% |
| ibm02 | 254 | 1.6316 | 0.078 | 0.809 | 2.299 | 5.1s | +14.5% | **+11.2%** ⭐ |
| ibm03 | 269 | 1.4148 | 0.081 | 0.841 | 1.827 | 4.6s | +18.7% | -7.0% |
| ibm04 | 285 | 1.4092 | 0.074 | 0.874 | 1.797 | 4.6s | +6.3% | -8.2% |
| ibm06 | 318 | 1.7944 | 0.068 | 0.834 | 2.619 | 4.5s | +28.4% | -10.9% |
| ibm07 | 335 | 1.5736 | 0.067 | 0.948 | 2.066 | 6.3s | +22.2% | -7.5% |
| ibm08 | 352 | 1.5037 | 0.069 | 0.871 | 1.998 | 7.0s | +21.8% | -5.3% |
| ibm09 | 369 | 1.1928 | 0.059 | 0.944 | 1.324 | 5.4s | +14.0% | -6.6% |
| ibm10 | 387 | 1.3843 | 0.069 | 0.716 | 1.915 | 18.1s | +34.4% | **+7.8%** ⭐ |
| ibm11 | 405 | 1.2855 | 0.056 | 0.965 | 1.495 | 7.6s | +24.9% | -9.2% |
| ibm12 | 423 | 1.6679 | 0.061 | 0.824 | 2.390 | 17.9s | +41.0% | **+3.4%** ⭐ |
| ibm13 | 441 | 1.4258 | 0.055 | 0.935 | 1.808 | 9.9s | +25.5% | -6.8% |
| ibm14 | 460 | 1.6280 | 0.053 | 1.000 | 2.150 | 16.6s | +28.4% | -5.5% |
| ibm15 | 479 | 1.6349 | 0.060 | 0.965 | 2.185 | 11.4s | +28.9% | -7.8% |
| ibm16 | 498 | 1.5458 | 0.050 | 0.895 | 2.097 | 15.6s | +30.8% | -4.6% |
| ibm17 | 517 | 1.7451 | 0.054 | 0.952 | 2.430 | 24.5s | +52.5% | -6.1% |
| ibm18 | 537 | 1.7916 | 0.054 | 1.041 | 2.435 | 11.3s | +35.5% | -1.1% |
| **AVG** | — | **1.5181** | — | — | — | **174s total** | **+28.6%** | **-4.1%** |

⭐ — обходит RePlAce baseline (3 из 17: ibm02, ibm10, ibm12). Will_seed обходил те же 3.

**Что менялось от will_seed:**
1. Та же архитектура (legalize + SA + опц. LNS) — но переписана на C++ (pybind11)
2. Добавлена outer LNS-фаза: 30 итераций × destroy 8 random + weighted-centroid repair + accept by full proxy_cost
3. Monkey-patch `pin_indices` → set: устранил O(N) lookup внутри `compute_proxy_cost`, дав ~28× ускорение

**Что сработало:**
- LNS post-processing спрятал ~1% улучшения над чистым SA (особенно сильно на ibm01: 1.2920 → 1.1781 = -8.8%)
- C++ переписывание SA: ~10× быстрее inner SA loop → можем себе позволить full 30 LNS iters везде

**Что не сработало (отвергнуто в ходе сессии):**
- SA-acceptance в LNS — ломает greedy улучшения
- Variable destroy + random repair — не консистентно
- Soft macro centroid update — массивная регрессия (WL)
- Spectral seed (Laplacian eigenvectors) — гораздо хуже initial
- `plc.optimize_stdcells()` — слишком медленно даже с маленьким budget

**Команда:**
```bash
$HOME/.local/bin/uv run evaluate submissions/straple/placer.py --all
```

---

### #2 · 2026-05-03 · `submissions/will_seed/placer.py` (reference)

**Сводка:**
- AVG proxy: **1.5336**
- Best: **1.1625** на `ibm09` (369 макросов)
- Worst: **1.7921** на `ibm18` (537 макросов)
- Total runtime: **34.74s**
- Slowest benchmark: `ibm17` — 6.05s
- Overlaps: 0 на каждом бенчмарке ✅
- vs SA baseline: +27.8% (better)
- vs RePlAce baseline: -5.2% (хуже)

**Per-benchmark детали:**

| Benchmark | Macros | Proxy | WL | Density | Congestion | Time | vs SA | vs RePlAce |
|---|---|---|---|---|---|---|---|---|
| ibm01 | 246 | 1.2920 | 0.074 | 1.047 | 1.390 | 3.23s | +1.9% | -29.5% |
| ibm02 | 254 | 1.6798 | 0.077 | 0.839 | 2.366 | 1.18s | +11.9% | **+8.6%** ⭐ |
| ibm03 | 269 | 1.4043 | 0.080 | 0.830 | 1.818 | 0.87s | +19.3% | -6.2% |
| ibm04 | 285 | 1.4478 | 0.073 | 0.907 | 1.842 | 1.54s | +3.7% | -11.2% |
| ibm06 | 318 | 1.7965 | 0.069 | 0.819 | 2.636 | 0.81s | +28.3% | -11.0% |
| ibm07 | 335 | 1.5903 | 0.067 | 0.978 | 2.069 | 0.84s | +21.4% | -8.7% |
| ibm08 | 352 | 1.5877 | 0.071 | 0.925 | 2.108 | 1.56s | +17.5% | -11.1% |
| ibm09 | 369 | **1.1625** ⭐ | 0.059 | 0.889 | 1.318 | 0.93s | +16.2% | -3.8% |
| ibm10 | 387 | 1.4116 | 0.071 | 0.752 | 1.929 | 4.06s | +33.1% | **+6.0%** ⭐ |
| ibm11 | 405 | 1.2547 | 0.055 | 0.921 | 1.479 | 1.24s | +26.7% | -6.6% |
| ibm12 | 423 | 1.6528 | 0.060 | 0.811 | 2.374 | 2.80s | +41.5% | **+4.2%** ⭐ |
| ibm13 | 441 | 1.4113 | 0.054 | 0.920 | 1.794 | 1.48s | +26.3% | -5.7% |
| ibm14 | 460 | 1.6515 | 0.053 | 1.013 | 2.184 | 2.97s | +27.4% | -7.0% |
| ibm15 | 479 | 1.6379 | 0.059 | 0.976 | 2.181 | 1.82s | +28.8% | -8.1% |
| ibm16 | 498 | 1.5484 | 0.050 | 0.893 | 2.103 | 1.91s | +30.7% | -4.8% |
| ibm17 | 517 | 1.7493 | 0.054 | 0.956 | 2.434 | **6.05s** ⬅ | +52.4% | -6.4% |
| ibm18 | 537 | 1.7921 | 0.054 | 1.042 | 2.435 | 1.44s | +35.4% | -1.1% |
| **AVG** | — | **1.5336** | — | — | — | **34.74s total** | **+27.8%** | **-5.2%** |

⭐ — обходит RePlAce baseline на этом конкретном бенчмарке (4 из 17).

**Декомпозиция cost (усреднено):**

| Компонент | Вклад в AVG | % от total |
|---|---|---|
| Wirelength (вес 1.0) | ~0.064 | ~4% |
| Density × 0.5 | ~0.92 × 0.5 = 0.46 | ~30% |
| Congestion × 0.5 | ~2.04 × 0.5 = **1.02** | **~66%** |

→ **Congestion — основной источник стоимости.** WL уже хорошо оптимизирован SA-фазой, density на среднем уровне, но congestion остаётся высоким. LNS должен фокусироваться на разгрузке congestion-hotspot'ов.

**Заметные паттерны:**
- На больших benchmark'ах (ibm12-ibm18, 423-537 макросов) — стабильно высокий congestion (>2.0), стабильно проигрываем RePlAce
- На маленьких (ibm01, 246 макросов) — proxy 1.29, разрыв с RePlAce 30% (RePlAce там 0.99)
- ibm02, ibm10, ibm12 — единственные где **обходим RePlAce** (на 4-9%) — стоит понять, почему именно эти
- ibm17 (517 макросов) самый дорогой по времени — 6 секунд из 34.74

**Чтение команды:**
```bash
$HOME/.local/bin/uv run evaluate submissions/will_seed/placer.py --all
```

---

### #1 · 2026-05-03 · `submissions/examples/greedy_row_placer.py` (демо)

**Сводка:**
- AVG proxy: **2.2109** (точно как в leaderboard)
- Best: 1.6728 на `ibm09`
- Worst: 2.7696 на `ibm12`
- Total runtime: **0.05s**
- Overlaps: 0 на каждом бенчмарке ✅
- vs SA baseline: -4.0% (хуже)
- vs RePlAce baseline: -51.7% (сильно хуже)

**Per-benchmark детали:**

| Benchmark | Proxy | WL | Density | Congestion | vs SA | vs RePlAce | Overlaps |
|---|---|---|---|---|---|---|---|
| ibm01 | 2.0463 | 0.121 | 1.245 | 2.606 | -55.4% | -105.1% | 0 |
| ibm02 | 2.0431 | 0.092 | 1.085 | 2.818 | -7.1% | -11.2% | 0 |
| ibm03 | 2.1484 | 0.111 | 1.075 | 2.999 | -23.5% | -62.5% | 0 |
| ibm04 | 1.9321 | 0.090 | 0.973 | 2.711 | -28.5% | -48.3% | 0 |
| ibm06 | 2.1620 | 0.076 | 1.044 | 3.127 | +13.7% | -33.6% | 0 |
| ibm07 | 2.1144 | 0.084 | 1.124 | 2.936 | -4.5% | -44.5% | 0 |
| ibm08 | 2.4704 | 0.095 | 1.194 | 3.557 | -28.4% | -72.9% | 0 |
| ibm09 | **1.6728** ⭐ | 0.079 | 1.098 | 2.089 | -20.6% | -49.4% | 0 |
| ibm10 | 2.4523 | 0.100 | 1.026 | 3.678 | -16.2% | -63.4% | 0 |
| ibm11 | 2.0604 | 0.075 | 1.165 | 2.806 | -20.4% | -75.0% | 0 |
| ibm12 | **2.7696** ⬅ | 0.095 | 1.090 | 4.258 | +2.0% | -60.5% | 0 |
| ibm13 | 2.1004 | 0.074 | 1.102 | 2.950 | -9.7% | -57.3% | 0 |
| ibm14 | 2.4473 | 0.088 | 1.098 | 3.621 | -7.6% | -58.5% | 0 |
| ibm15 | 1.9488 | 0.077 | 1.050 | 2.694 | +15.3% | -28.6% | 0 |
| ibm16 | 2.5767 | 0.069 | 1.065 | 3.949 | -15.4% | -74.3% | 0 |
| ibm17 | 2.5411 | 0.067 | 1.093 | 3.856 | +30.8% | -54.5% | 0 |
| ibm18 | 2.0998 | 0.062 | 1.094 | 2.982 | +24.3% | -18.5% | 0 |
| **AVG** | **2.2109** | — | — | — | **-4.0%** | **-51.7%** | 0 |

**Что показывает:** Shelf-pack даёт самый маленький WL из всех (0.05-0.12, потому что компактно укладывает), но **density и congestion катастрофические** — макросы не учитывают netlist, всё запихивается в нижнюю часть канваса.

**Чтение команды:**
```bash
$HOME/.local/bin/uv run evaluate submissions/examples/greedy_row_placer.py --all
```

---

## 🏅 Где мы в leaderboard прямо сейчас

Если бы сабмитили текущий best (will_seed 1.5336):

```
...
| 24 | "#5 ubc cpen student" (Gene Pool Shuffle) | 1.5337 | ...
| 25 | Will Seed (Partcl) | 1.5336 | ⬅ МЫ
| 26 | "UT Austin" - RH (DREAMPlace) | 1.6037 | ...
...
| — | RePlAce (baseline) | 1.4578 | ⬅ хотим перепрыгнуть
...
| — | SA (baseline) | 2.1251 |
```

Нас обошли бы **24 команды**. Чтобы выйти в плюс относительно RePlAce — нужно улучшение хотя бы на **5.2%**.

### Ближайшие соседи и что они делают

| Место | Команда | Score | Подход (если известен) |
|---|---|---|---|
| 22 | SEVmakers | 1.5200 | Hybrid Legalization + SA |
| 23 | "CA" (congestion_aware) | 1.5247 | Congestion-aware |
| 24 | UBC student | 1.5337 | Gene Pool Shuffle |
| **25** | **Will Seed (мы)** | **1.5336** | SA refinement |
| 21 | oracleX | 1.5130 | — |
| 20 | UTAUSTIN-CT | 1.5062 | PLC-Exact Congestion-Aware SA |
| 19 | Jiangban Ya | 1.4943 | Spectral-Seed + Adaptive Legalizer |
| 18 | W3 Solutions | 1.4824 | GRACE |
| **—** | **RePlAce baseline** | **1.4578** | ⬅ цель минимум |

Чтобы достичь **топ-10 (1.4076)** — нужно улучшение на 8.9%.
Чтобы попасть в **топ-7 (Гран-при, ~1.348)** — на 12.7%.

---

## 🔬 Гипотезы / на что смотреть дальше

### 1. Congestion — главный bottleneck

В декомпозиции will_seed: ~66% AVG proxy идёт от congestion. SA в will_seed оптимизирует HPWL (wirelength), но congestion напрямую не таргетит → его не сильно понижает.

**Идея:** добавить в LNS-фазу destroy operator, который **атакует congested-области** — снимает макросы из routing-hotspot'ов и переразмещает их в менее загруженные зоны.

### 2. Большие benchmark'ы (ibm15-ibm18) — стабильное проседание

На больших дизайнах will_seed почти не догоняет RePlAce (-7..-1%), на маленьких сильнее (-29% на ibm01). Скорее всего, SA с 3000 итераций недостаточно для исследования пространства 500+ макросов.

**Идея:** на больших benchmark'ах — **больше итераций** или **multi-start**. У нас runtime запас огромный (1 час vs 6 секунд).

### 3. ibm02, ibm10, ibm12 — обходим RePlAce даже простым SA

Это маленькая статистика, но интересно понять, что в их структуре нетлиста делает SA эффективным.

**Идея:** проанализировать графы netlist'ов этих benchmark'ов — может быть, у них специфическая топология (мало больших нетов? кластеры?), и под неё подстроить наш алгоритм.

### 4. Runtime запас 600× — есть пространство для экспериментов

ibm17 (worst case) = 6 секунд из 3600 секунд лимита. Можем гонять multi-start с 100 разными seeds, или делать adaptive ALNS с долгой историей operator weights, или добавить RL-bandit на operator selection.

---

## 📐 Технические детали

**Окружение:**
- OS: macOS (darwin), aarch64 (Apple Silicon)
- Python: 3.14.4
- uv: 0.11.8
- torch: 2.10.0 (CPU build, MPS не используется will_seed'ом — он на numpy)
- numpy: 2.4.2
- macro-place: 0.1.0 (editable из репо)

**Нюансы железа:**
- Single-threaded numpy в will_seed → используется ~1 ядро
- Eval-машина (16 ядер EPYC) → если distribute through `multiprocessing` → можно ускорить linearly
- Mac ARM — нет CUDA → DRP интеграция нужна на Linux/CUDA-машине

**Команды для воспроизведения:**
```bash
# Текущий best (will_seed)
$HOME/.local/bin/uv run evaluate submissions/will_seed/placer.py --all

# Демо (greedy)
$HOME/.local/bin/uv run evaluate submissions/examples/greedy_row_placer.py --all

# Один benchmark (быстрая отладка)
$HOME/.local/bin/uv run evaluate submissions/will_seed/placer.py -b ibm01 --vis

# Smoke tests
$HOME/.local/bin/uv run pytest test/test_smoke.py -v
```

---

## 📋 Шаблон для будущих записей

```markdown
### #N · YYYY-MM-DD · `путь/к/placer.py` (короткое описание)

**Сводка:**
- AVG proxy: X.XXXX
- Best: X.XXXX на ibmXX
- Worst: X.XXXX на ibmXX
- Total runtime: X.XXs
- Overlaps: 0
- vs SA: ±X%
- vs RePlAce: ±X%

**Что менялось от прошлого прогона:**
- ...

**Что сработало:**
- ...

**Что не сработало:**
- ...

**Per-benchmark детали:** [таблица как у will_seed выше]

**Команда:** `uv run evaluate ... --all`
```
