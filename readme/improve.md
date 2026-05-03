# Промпт для следующей сессии — поиск и реализация улучшений

> **Контекст**: Partcl/HRT Macro Placement Challenge 2026, команда **Straple**, дедлайн **21 мая 2026**.
> Перед началом прочитай: [INDEX.md](INDEX.md), [todo.md](todo.md), [results.md](results.md), [PROBLEM.md](PROBLEM.md).

## Текущая точка

| | Значение |
|---|---|
| Best AVG (наш) | **1.5181** |
| RePlAce baseline (порог отсечки) | 1.4578 — **нужно пробить -4.0%** |
| Топ-10 leaderboard | 1.4076 |
| Топ-7 (Гран-при $20K) | 1.3479 |
| Лимит | 1 час на бенчмарк, 16 cores + 100GB + RTX 6000 Ada GPU |
| Наш runtime | 17.66с на --all (запас **~1700×**) |

Архитектура — pure C++ placer:
- `submissions/straple/placer.py` — тонкая Python обёртка
- `submissions/straple/cpp/placer_core.cpp` — legalize + SA + LNS destroy/repair
- `submissions/straple/cpp/proxy_cost.cpp` — exact replica `plc.compute_proxy_cost` в C++ (WL/density/congestion)
- Сборка: `submissions/straple/cpp/build.sh`
- Тесты: `uv run pytest test/test_smoke.py -v` → 9/9 PASS

## Что делать в этой сессии

**Цель**: пробить RePlAce 1.4578 (минимум), потом топ-10 (1.4076).

### Workflow

1. **Найти идеи** (источники):
   - **Научные статьи** по macro placement / VLSI: ePlace, RePlAce, DREAMPlace, MOSAIC (UToronto), TierPlace (V5), AutoDMP (Archgen #7), Multi-DREAMPlace (Shoom #5), ALNS+Thompson Sampling (TAISPlAce), HyperPlace SA+LNS (ArzunPD), Spectral seed (Jiangban Ya). Конкретные ссылки — в [ALGORITHMS.md](ALGORITHMS.md).
   - **Performance**: SIMD (AVX2/NEON), multi-threading (OpenMP, std::thread), GPU offload (CUDA для density/congestion), incremental cost updates.
   - **Алгоритмические улучшения**: лучший initial seed, more sophisticated destroy/repair, adaptive operator selection (ALNS), proxy-aware SA, multi-start, simulated quenching, Thompson sampling для bandit operators.

2. **Записать всё в [todo.md](todo.md)** в секцию `## 4. Идеи / Pool`:
   - Дата идеи
   - Источник (paper / leaderboard team / собственная)
   - Ожидаемая выгода (1-15%) и сложность (1-5)
   - Каждую идею пометить `[ ]` (pending) / `[~]` (in progress) / `[x]` (done) / `[-]` (rejected)
   - В журнале экспериментов фиксировать: дата, изменение, замер до/после, инсайты

3. **Реализовать топ-3 по ROI** (выгода / сложность):
   - Каждую идею — отдельный файл/коммит, чтобы можно было откатить
   - Сначала — debug на одном бенчмарке (ibm01, ibm03, ibm12 — три размера)
   - Если debug проходит → прогон на --all с записью полного per-benchmark результата
   - Если хуже — откатить, но записать в [results.md](results.md) что не сработало

4. **Тестировать**:
   - **Smoke**: `uv run pytest test/test_smoke.py -v` после каждого изменения C++ — должно оставаться 9/9 PASS
   - **Benchmark**: `$HOME/.local/bin/uv run evaluate submissions/straple/placer.py --all` — обновлять [results.md](results.md) по шаблону внизу того же файла
   - **Validation**: 0 overlaps на каждом бенчмарке, иначе DQ

### Ключевые правила

- **Никакого хардкода** под имя бенчмарка (`if benchmark.name == "ibm17"` = DQ). Адаптивные параметры по структуре (`if num_movable > 500`) — OK.
- **Не модифицировать** `macro_place/` (evaluator). Свой код только в `submissions/straple/`.
- **Всё что можно — в C++**, Python — только обёртка. Если идея требует python loop в hot path, сначала вынести в C++.
- **Проверять что C++ proxy_cost совпадает** с `plc.compute_proxy_cost` (тест `test_straple_proxy_cost_matches_plc`) — при изменении proxy не сломать exact match (WL/density: 1e-4, congestion: 5e-3).
- **Бюджет на debug-итерацию ≤ 1 минута** (один бенчмарк — debug, --all — full validation).

### LNS — таймер 300с на бенчмарк

Текущий LNS: 30 фиксированных итераций. Заменить на **time-budget loop**:

```cpp
auto deadline = steady_clock::now() + 300s;
while (steady_clock::now() < deadline) {
    auto trial = destroy_and_repair(...);
    auto new_cost = evaluator.evaluate(trial);
    if (new_cost < best_cost) {
        best_cost = new_cost;
        best_pos = trial;
    }
}
```

**Параметризовать**: `lns_time_budget_seconds` (default 300). На --all из 17 benches это **~85 минут** (внутри 1-часового лимита на каждый, с запасом). 300с/bench — раскрытие LNS на полную: при текущем 0.5-2с на итерацию это **150-600 LNS итераций**, vs 30 сейчас.

Можно **adaptive budget**: маленькие бенчмарки (<300 macros) — 60с, средние — 180с, большие (>500) — 300с. Цель — не упереться в 1-часовой timeout даже на самых больших.

### Конкретные направления для исследования

#### A. Performance / параллелизм
- [ ] **OpenMP в C++ inner SA loop** — несколько SA-цепочек параллельно (multi-walker), периодическая синхронизация лучшим
- [ ] **OpenMP в congestion smoothing** — независимый pass по строкам/столбцам
- [ ] **SIMD для overlap check** — vectorized `dx < sep_x[i] && dy < sep_y[i]` через AVX2 / NEON
- [ ] **Incremental proxy_cost** — после destroy k macros пересчитывать только затронутые grid cells (k×grid_w cells), не всё. Это даёт ещё ~10× для inner LNS.
- [ ] **CUDA для density/congestion**: rasterize macros на GPU, top-K через `thrust::sort`. Только если время C++ optimization исчерпано.

#### B. Алгоритмические идеи (по ROI убыванию)
- [ ] **Multi-start с n=5 сидов**: запустить full pipeline 5×, вернуть лучший по proxy. Известно работает (Shoom #5, Archgen #7, ~1-2% бонус).
- [ ] **Proxy-aware SA**: вместо HPWL inner-objective, каждые 50 моих SA-ходов делать proxy_cost call. Принять последний батч если proxy улучшился. Замечание: SA на чистом HPWL после 3000 итераций ухудшает proxy (см. [results.md](results.md) #2 hypotheses).
- [ ] **ALNS / Thompson Sampling**: набор destroy operators (random / spatial-cluster / worst-cost / net-based) с весами, обновляемыми по успеху. TAISPlAce (#13, 1.4321) и MOSAIC так делают.
- [ ] **Лучший seed**:
   - [ ] **Свой analytical placer** — gradient descent на smooth surrogate (log-sum-exp WL + bell density). Прототип в [submissions/straple/analytical_seed.py](../submissions/straple/analytical_seed.py) — портировать в C++.
   - [ ] **Min-cut bisection** через recursive partitioning (METIS / KaHIP) — классика для seed.
   - [ ] **Force-directed initial** перед SA — пружинка по нетам, отталкивание от плотных зон.
- [ ] **Soft-macro joint optimization**: реализовать свой быстрый soft updater в C++ (центроид связанных hard'ов + force-directed correction). `plc.optimize_stdcells()` слишком медленный (5+ мин).
- [ ] **Mirror/flip orientation** (Klein-4: N, FN, FS, S): добавить orientation как параметр placement. Сохранить в `orientations.pt` рядом с placement.

#### C. Идеи про congestion (главный bottleneck — 66% cost)
- [ ] **Congestion-aware destroy**: пикать макросы из congested grid cells (top-5%) для destroy
- [ ] **Routing-aware repair**: предпочитать positions с низкой текущей congestion для repair
- [ ] **Net clustering**: группировать макросы по net affinity, размещать кластеры в регионах canvas — снижает routing crossings

#### D. Идеи про density (40% cost)
- [ ] **Bell-curve density penalty** в SA inner loop вместо строгого overlap-check — позволяет macros проходить друг через друга при низкой температуре, разгружая регионы
- [ ] **Spread move**: новый тип SA-хода — двигать N соседей вместе, equally spaced

#### E. Использование открытости Tier 1 тестов
- [ ] **Per-bench feature analysis**: на основе `num_movable`, `density`, `congestion-pattern` подбирать параметры (без hardcode под имя)
- [ ] **GNN/RL обучение на 16 IBM, оценка на 17-м**: leave-one-out CV, ловит overfitting. Выходной формат — pretrained веса в submissions/straple/.

## Шаблон записи в todo.md

```markdown
| Дата | Идея | Источник | Ожид.выгода | Сложность | Статус | Замер |
|---|---|---|---|---|---|---|
| YYYY-MM-DD | Краткое описание | RePlAce paper / Shoom #5 / своё | -2% | 3 | [~] | -1.3% AVG (повторено 3 запуска) |
```

## Шаблон записи в results.md

Внизу [results.md](results.md) — секция "Шаблон для будущих записей". Использовать `### #N · YYYY-MM-DD · placer.py · variant_name`.

## Финальная проверка перед сабмитом

См. [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md). Главное: **прогон в чистой среде** (новый `uv sync`, чистый clone), чтобы избежать ловушек Mike Gao / BakaBobo.

## Команды

```bash
# Build C++
submissions/straple/cpp/build.sh

# Smoke tests
$HOME/.local/bin/uv run pytest test/test_smoke.py -v

# Single benchmark debug
$HOME/.local/bin/uv run evaluate submissions/straple/placer.py -b ibm01

# Full Tier 1
$HOME/.local/bin/uv run evaluate submissions/straple/placer.py --all

# Full Tier 1 with visualization
$HOME/.local/bin/uv run evaluate submissions/straple/placer.py --all --vis

# NG45 (Tier 2)
$HOME/.local/bin/uv run evaluate submissions/straple/placer.py --ng45
```
