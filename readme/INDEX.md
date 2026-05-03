# 📚 readme/ — навигация по русским материалам

Это рабочая папка с русскими переводами и справочными материалами для участия в **Partcl/HRT Macro Placement Challenge 2026**.

> Английские оригиналы официальных документов — в корне репо. Они источник истины при расхождениях.

## Файлы здесь

### Условие задачи и быстрый старт

| Файл | Что внутри | Когда смотреть |
|---|---|---|
| [PROBLEM.md](PROBLEM.md) | **Формальная постановка задачи** — что дано, что найти, что минимизировать, при каких ограничениях. С формулами для proxy cost / WL / density / congestion и описанием Tier 2 scoring. | **В первую очередь.** Чтобы понять, что вообще нужно сделать |
| [QUICKSTART.md](QUICKSTART.md) | **Пошаговое руководство** от чистой системы: установка `uv`, submodule, `uv sync`, smoke-tests, запуск дефолтных placer'ов (`greedy_row_placer`, `will_seed`), создание скелета своего, цикл разработки | **После PROBLEM.md.** Когда нужно реально запустить и получить первые числа |
| [HARDWARE.md](HARDWARE.md) | **Про оценочное железо** (AMD EPYC 9655P 16 cores + 100GB + RTX 6000 Ada 48GB) и как его использовать в коде: GPU/CUDA, multi-threading (`torch.set_num_threads`), multiprocessing для multi-start, eval_docker контейнер, профилирование | Когда упёрся в runtime или планируешь GPU-путь |

### Переводы официальных документов

| Файл | Что внутри | Когда смотреть |
|---|---|---|
| [README.md](README.md) | Перевод корневого README — обзор соревнования, правила, leaderboard | Если нужно понять правила/призы/формат, не лезя в английский оригинал |
| [SETUP.md](SETUP.md) | Перевод SETUP — установка, структура проекта, API референс, soft macro helper, ORFS | Когда пишешь свой placer и забыл сигнатуру |
| [SCORING.md](SCORING.md) | Перевод SCORING — правила Гран-при (Tier 2), feasibility gate, geometric mean WNS:TNS:Area = 3:2:1 | Когда думаешь о шансах на Гран-при |

### Справочники / cheatsheets

| Файл | Что внутри | Когда смотреть |
|---|---|---|
| [GLOSSARY.md](GLOSSARY.md) | Глоссарий терминов EDA: HPWL, WNS, TNS, density, congestion, hard/soft macro и т.д. | Когда встретил незнакомый термин и хочется быстро понять |
| [API_CHEATSHEET.md](API_CHEATSHEET.md) | Компактная шпаргалка по API: импорты, сигнатуры, форматы тензоров, минимальный шаблон placer'а | Во время написания/отладки кода — бегло свериться |
| [ALGORITHMS.md](ALGORITHMS.md) | Обзор подходов к macro placement: SA, RePlAce, DREAMPlace, ePlace, LNS, Spectral, GNN — что в leaderboard и что работает | На этапе выбора архитектуры решения |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Типичные проблемы: overlap'ы float-precision, runtime > 1 час, NaN в proxy cost, validator-fail, и т.д. | Когда что-то сломалось и нужны идеи |
| [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) | Чеклист перед сабмитом — список из 20+ пунктов, которые нужно проверить, чтобы избежать DQ | Перед каждым сабмитом, обязательно |

### Рабочий план и результаты

| Файл | Что внутри |
|---|---|
| [todo.md](todo.md) | Личный план: цели, анализ топов, стратегия (DREAMPlace seed → LNS), roadmap по фазам Day 1-28, журнал экспериментов, идеи, риски |
| [results.md](results.md) | **Полные результаты замеров** — текущий best, per-benchmark таблицы, декомпозиция cost (WL/density/congestion), позиция в leaderboard, гипотезы на следующие шаги |

---

## Рекомендованный порядок чтения для нового участника

1. [PROBLEM.md](PROBLEM.md) — **что вообще нужно сделать** (формальная постановка)
2. [README.md](README.md) — что за соревнование, какие призы, формат
3. [QUICKSTART.md](QUICKSTART.md) — **поставить окружение, запустить дефолтное решение**
4. [GLOSSARY.md](GLOSSARY.md) — освоить EDA-терминологию
5. [SETUP.md](SETUP.md) — детальный API референс
6. [API_CHEATSHEET.md](API_CHEATSHEET.md) — закрепить шпаргалкой
7. [ALGORITHMS.md](ALGORITHMS.md) — понять, какие подходы существуют
8. [HARDWARE.md](HARDWARE.md) — как использовать GPU/multi-core
9. [todo.md](todo.md) — наш текущий план (его и редактируем по ходу)
10. [SCORING.md](SCORING.md) — когда задумаешься о Гран-при
11. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — справочник, когда что-то сломалось
12. [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md) — обязательно перед каждым сабмитом

---

## Где что в самом репо

- [../README.md](../README.md), [../SETUP.md](../SETUP.md), [../SCORING.md](../SCORING.md), [../LICENSE.md](../LICENSE.md) — английские оригиналы (нужны GitHub'у)
- [../macro_place/](../macro_place/) — код evaluator'а (`benchmark.py`, `loader.py`, `objective.py`, `utils.py`, `evaluate.py`, `def_writer.py`)
- [../submissions/examples/](../submissions/examples/) — `greedy_row_placer.py` и `simple_random_placer.py`
- [../submissions/will_seed/](../submissions/will_seed/) — reference placer от организаторов (изучить!)
- [../benchmarks/processed/public/](../benchmarks/processed/public/) — pre-processed `.pt` бенчмарки (можно работать без submodule)
- [../external/MacroPlacement/](../external/MacroPlacement/) — TILOS submodule (нужно `git submodule update --init`)
- [../scripts/](../scripts/) — `evaluate_with_orfs.py` для Tier 2, конверторы и т.п.
- [../test/test_smoke.py](../test/test_smoke.py) — pytest smoke-тесты
- [../eval_docker/](../eval_docker/) — docker-окружение, в котором судьи прогоняют сабмишены
