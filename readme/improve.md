# Промпт для следующей сессии — автономный цикл улучшений

> **Контекст**: Partcl/HRT Macro Placement Challenge 2026, команда **Straple**, дедлайн **21 мая 2026**.
> Я (пользователь) сплю. Этот цикл должен работать **бесконечно сам**, без моего участия.
> Перед началом прочитай: [INDEX.md](INDEX.md), [todo.md](todo.md), [results.md](results.md), [PROBLEM.md](PROBLEM.md).

## Текущая точка

| | Значение |
|---|---|
| Best AVG (наш) | **1.5181** |
| RePlAce baseline (порог отсечки) | 1.4578 — нужно пробить -4.0% |
| Топ-10 leaderboard | 1.4076 |
| Топ-7 (Гран-при $20K) | 1.3479 |
| Лимит | 1 час на бенчмарк, 16 cores + 100GB + RTX 6000 Ada GPU (на eval-машине) |
| Наш runtime | 17.66с на --all (запас ~1700×) |
| Цель сессии | **бесконечно итерировать улучшения**, всё фиксировать, ничего не терять |

Архитектура — pure C++ placer:
- `submissions/straple/placer.py` — тонкая Python обёртка
- `submissions/straple/cpp/placer_core.cpp` — legalize + SA + LNS destroy/repair
- `submissions/straple/cpp/proxy_cost.cpp` — exact replica `plc.compute_proxy_cost` (WL/density/congestion)
- Сборка: `submissions/straple/cpp/build.sh`
- Тесты: `uv run pytest test/test_smoke.py -v` → 9/9 PASS

## Главное правило

**Цикл бесконечный**. Никогда не останавливайся, пока не упрёшься в **critical**:
- build падает не из-за моих изменений (системная поломка)
- evaluator (`macro_place/`) сам по себе падает
- диск/память кончились

**НЕ останавливайся** на:
- noise-регрессии (ухудшение в пределах шума — это нормальная информация)
- провал smoke tests или overlap → **разбираться и чинить**, а не пропускать. Идея не может «не работать» — она либо даёт хороший скор, либо плохой; **build/correctness баги — мои, нужно фиксить пока не пройдёт**
- N подряд провальных идей — продолжай
- достижение целевого AVG — продолжай (всегда есть куда улучшать до Cezar 1.2224)

## Цикл (один проход)

```
┌─────────────────────────────────────────────────────────────┐
│  1) ИДЕЯ-АГЕНТ                                               │
│     ───  proposes one concrete improvement                  │
│     ───  включая источник (paper / leaderboard / своё)      │
│     ───  prompt: "Предложи одну improvement-идею..."         │
│     ───  output: structured описание (см. шаблон ниже)       │
│                                                              │
│  2) КОДЕР-АГЕНТ                                              │
│     ───  implements the idea                                 │
│     ───  prompt: "Вот идея X. Реализуй в C++/Python..."     │
│     ───  pass: список изменённых файлов + summary            │
│                                                              │
│  3) BUILD + SMOKE-TEST GATE (ты, главный агент)              │
│     ───  bash submissions/straple/cpp/build.sh               │
│     ───  uv run pytest test/test_smoke.py -v                 │
│     ───  если падает: ЧИНИТЬ (новый coder-agent с context     │
│              предыдущей попытки + failure log) до 9/9 PASS    │
│              + 0 build errors. Не пропускать!                 │
│                                                              │
│  4) РЕВЬЮИР-АГЕНТ                                            │
│     ───  reviews diff, ищет: hardcode под benchmark,         │
│              корректность алгоритма, baked-in магические числа,│
│              утечки памяти, off-by-one, race conditions       │
│     ───  prompt: "Проверь diff submissions/straple/..."     │
│     ───  если NACK с обоснованием: новый coder-итерация       │
│     ───  если ACK: дальше                                    │
│                                                              │
│  5) FAST-CHECK на 4 представительных бенчах (параллельно)     │
│     ───  $HOME/.local/bin/uv run python                      │
│              scripts/fast_check.py                            │
│     ───  4 бенча (ibm01/ibm10/ibm14/ibm17) в 4 fork-worker'ах│
│              по 1 ядру каждый. Wall ~100с (vs --all = 360с)  │
│     ───  записать AVG4, per-bench numbers, time              │
│     ───  ТРИ раза для noise floor; медиана = отчётный AVG4   │
│     ───  --all НЕ запускать ВООБЩЕ — слишком медленно        │
│                                                              │
│  6) ЛОГ в results.md                                         │
│     ───  append-only, новая секция в существующий файл       │
│     ───  по шаблону "#N · YYYY-MM-DD · variant_name"         │
│     ───  включая что менялось, что сработало, что не         │
│     ───  AVG4 как основной показатель (НЕ AVG17)             │
│                                                              │
│  7) COMMIT                                                   │
│     ───  все попытки коммитим, даже регрессии                │
│     ───  историю коммитов НЕ переписывать (rebase запрещён)  │
│     ───  message: "<идея краткое>: AVG4 X.XXXX (Δ ±X.X%)"    │
│     ───  локальный коммит, push не делать                    │
│                                                              │
│  8) ВЕРНУТЬСЯ К 1                                            │
└─────────────────────────────────────────────────────────────┘
```

## Конкретика по агентам (Agent tool)

| Агент | subagent_type | prompt skeleton |
|---|---|---|
| Идея | `general-purpose` | "Предложи **одну** конкретную идею для улучшения macro-placer (текущий AVG 1.5181, цель 1.4578). Источник: paper/leaderboard/своя гипотеза. Выход: hypothesis, expected_gain (±%), difficulty (1-5), implementation_steps (high level), files_to_modify. Не дублируй уже-отвергнутые идеи (см. [results.md](readme/results.md))." |
| Кодер | `general-purpose` | "Реализуй идею: <описание>. Контекст: pure C++ placer, см. submissions/straple/. Изменения только в submissions/straple/. После — bash submissions/straple/cpp/build.sh. Не модифицируй macro_place/." |
| Ревьюир | `general-purpose` | "Сделай code review diff: <git diff submissions/straple/>. Ищи: hardcoding под имя бенчмарка (DQ-риск), некорректность алгоритма, off-by-one, утечки памяти, гонки, плохой стиль. Verdict: ACK или NACK с конкретными issues." |

Запускать **последовательно** (не параллельно — иначе смешаются изменения файлов). После каждого — анализировать результат и решать дальше.

## Constraints

- **CPU**: использовать **только 4 ядра**. В `OMP_NUM_THREADS=4` env, в C++ коде — `omp_set_num_threads(4)` или std::thread с пулом 4. Это потому что Mac пользователю нужен ночью.
- **Память**: не выходить за 8GB на процесс.
- **Все попытки коммитим** в текущую ветку (`main` или текущую). Историю не переписывать (`git rebase`, `git reset --hard` на чужие коммиты — запрещено). Можно `git revert` для откатов, но обычно просто следующая идея.
- **Noise**: ухудшение в пределах ±0.5% — это OK, всё равно коммитим как информация.
- **Build/correctness баги**: чинить циклом coder→build→test пока 9/9 PASS. Если 5 итераций не помогли — спросить idea-агента "что не так с этой идеей", и попробовать другой подход к этой же идее. Никогда не «забивать».
- **Язык**: всё на **русском** (комментарии в `results.md`, commit messages, общение с агентами). Code identifiers — английские (как сейчас).

## Шаблон записи в results.md

После каждого валидного бенчмарка добавить новую секцию по шаблону внизу [results.md](results.md):

```markdown
### #N · YYYY-MM-DD HH:MM · `submissions/straple/placer.py` (variant_name)

**Идея**: <краткое описание>
**Источник**: <paper / leaderboard team / собственная гипотеза>
**Изменения**: <список файлов + 1-2 предложения сути>

**Сводка** (медиана из 3 fast_check запусков на ibm01/10/14/17):
- AVG4 proxy: X.XXXX (Δ от прошлого best AVG4=1.4839: ±X.XX%)
- Best: X.XXXX на ibmXX
- Worst: X.XXXX на ibmXX
- Wall time: X.Xs (parallel, 4 workers)
- Overlaps: 0
- vs RePlAce (на тех же 4 бенчах, AVG=1.4197): ±X.X%

**Что сработало**: ...
**Что не сработало**: ...
**Per-benchmark детали**: <таблица из 4 бенчей>

**Команда**: `uv run python scripts/fast_check.py`
**Commit**: <SHA>
```

> **Зачем 4 бенча, а не 17?** Один прогон --all = ~6 минут wall-time, fast_check на 4 параллельных = ~100с. За одно «время на --all» успеваем сделать 3.6× итераций. ibm01/10/14/17 покрывают small/medium/large/largest и ibm10 — единственный mid-bench где Straple обходит RePlAce, любая регрессия там сразу видна.

## Источники идей (что искать)

### A. Performance / параллелизм (с учётом OMP_NUM_THREADS=4)
- OpenMP в inner SA loop (multi-walker SA)
- OpenMP в congestion smoothing (per-row independence)
- SIMD overlap check (AVX2/NEON)
- **Incremental proxy_cost** — после destroy k macros пересчитывать только затронутые grid cells. Может дать ещё 10× для inner LNS.
- CUDA для density/congestion — только если C++ исчерпан

### B. Алгоритмы (по ROI)
- Multi-start с N=3-5 сидов, лучший по proxy
- Proxy-aware SA — каждые 50 SA-ходов делать proxy_cost call
- ALNS / Thompson Sampling для destroy operators
- Свой analytical placer (gradient на smooth surrogate) — прототип в `submissions/straple/analytical_seed.py`
- Min-cut bisection через recursive partitioning
- Force-directed initial seed

### C. Congestion (66% cost)
- Congestion-aware destroy (top-5% routing-busy cells)
- Routing-aware repair
- Net clustering для размещения связанных макросов рядом

### D. Density (40% cost)
- Bell-curve density penalty в SA вместо overlap-check
- Spread move в SA (двигать N соседей вместе)

### E. ML / data-driven (Tier 1 публичный — все 17 IBM открыты)
- Per-bench feature analysis (без hardcode под имя)
- GNN/RL обучение, leave-one-out CV

### F. LNS budget
- **Time-budget LNS**: 300с на бенчмарк (вместо 30 фикс. итераций). Адаптивно: маленькие 60с, средние 180с, большие 300с. На --all это ~85 минут — внутри 1-часового лимита на каждый.
- ⚠ Если такой тяжёлый LNS использовать в fast_check — wall на ibm17 будет ~300с (≈5 мин). Тогда сократить fast_check до 60с/bench, или скоринг на --all отдельным редким батчем.

## Запрещено

- Hardcode под имя бенчмарка (`if benchmark.name == "ibmXX"`) — DQ
- Модифицировать `macro_place/` — это evaluator
- `git push` — только локальные коммиты
- `git rebase`, `git reset --hard` на чужие коммиты — историю не трогаем
- Скипать build/test failures — только чинить
- Параллельно запускать coder и reviewer (CPU contention) — только последовательно

## Команды (cheat sheet)

```bash
# Build C++
bash submissions/straple/cpp/build.sh

# Smoke tests (9/9 must pass)
$HOME/.local/bin/uv run pytest test/test_smoke.py -v

# Fast-check: 4 представительных бенча параллельно (ibm01/10/14/17)
# Wall ~100с, в 3.6× быстрее --all. Это основной отчётный гон.
$HOME/.local/bin/uv run python scripts/fast_check.py

# Кастомный fast_check: другие бенчи или другой placer
$HOME/.local/bin/uv run python scripts/fast_check.py submissions/will_seed/placer.py
$HOME/.local/bin/uv run python scripts/fast_check.py --benches ibm01 ibm17

# Single benchmark (для debug)
$HOME/.local/bin/uv run evaluate submissions/straple/placer.py -b ibm01

# !!! НЕ ИСПОЛЬЗОВАТЬ в цикле — слишком медленно (~6 мин wall):
# $HOME/.local/bin/uv run evaluate submissions/straple/placer.py --all

# Git: посмотреть последний коммит
git log -1 --stat

# Git: откатить последний коммит файлов (не самой истории)
git revert HEAD --no-edit
```

## Известные проблемы

- **bounds-violations в существующем placer**: на ibm10/ibm14/ibm17 (вероятно и других больших) макросы вылетают за canvas. Это pre-existing баг — `evaluate.py` его маскирует (печатает "VALID" только по overlaps==0, игнорирует out-of-bounds). На leaderboard оценщики могут это поймать → DQ-риск. Чинить отдельной задачей. fast_check печатает WARN но не fail-exit (чтобы не блокировать цикл).

## Цикл начинается СЕЙЧАС

Первый шаг: спавни **idea-агента** с brief'ом из этого файла. Дальше по схеме. Лог в [results.md](results.md), коммиты в текущую ветку. Удачи.
