# Partcl/HRT Macro Placement Challenge

> Русский перевод официального [README.md](../README.md). Оригинал на английском — источник истины при расхождениях.

<img src="../assets/HRT.png" alt="Hudson River Trading" height="80"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <img src="../assets/partcl.png" alt="Partcl" height="80">

**Выиграй $20,000, разработав алгоритм размещения макроблоков лучше существующих!**

Partcl и Hudson River Trading рады совместно провести соревнование по решению задачи размещения макроблоков (macro placement).

## О задаче macro placement

Macro placement — задача размещения больших фиксированных по размеру блоков (SRAM, IP-блоки, аналоговые макросы и т.п.) на флорплане чипа таким образом, чтобы сбалансировать перегруженность маршрутизации (routing congestion), тайминг, доставку питания и площадь. В отличие от размещения стандартных ячеек, макросы имеют сильные геометрические ограничения и ограничения на связность, поэтому задача — исследовать сильно дискретное пространство решений, минимизируя длину проводов, избегая блокировок и сохраняя последующую трассируемость и качество тайминга.

Например, бенчмарк **ibm01** содержит:
- **246 жёстких макросов (hard macros)** разного размера (от 0.8 до 27 μm², размах в 33×)
- **7,269 нетов (nets)**, соединяющих макросы между собой и с 894 предразмещёнными кластерами стандартных ячеек
- **Канвас 22.9 × 23.0 μm** с утилизацией площади 42.8%

<p align="center">
  <img src="../assets/sa_ibm01.gif" alt="Simulated annealing on ibm01" width="600"><br>
  <img src="../assets/fd_ibm01.gif" alt="Force-directed placement on ibm01" width="600">
</p>

## О HRT Hardware

Hudson River Trading (HRT) — ведущая компания количественной торговли, передовая в технических инновациях на глобальных финансовых рынках.

Команда HRT Hardware строит высокопроизводительные вычислительные системы, лежащие в основе торговой инфраструктуры. Используются FPGA и ASIC для принятия решений с минимальной задержкой и для индивидуальных решений во всём торговом стеке — от заказных схем до ML-ускорителей.

Мы спонсируем это соревнование, потому что прогресс в macro placement и низкоуровневой оптимизации железа напрямую совпадает с типом инженерных задач, над которыми работают наши команды каждый день.

Присоединение к HRT Hardware означает работу с лучшими инженерами в одной из самых продвинутых вычислительных сред глобальных финансов. Подробнее об открытых вакансиях: [hudsonrivertrading.com](https://www.hudsonrivertrading.com/).

## О Partcl

Partcl перестраивает инфраструктуру проектирования чипов с нуля под эпоху GPU.

Современное проектирование чипов медленное, фрагментированное и фундаментально ограничено инструментами, построенными десятилетия назад. Критические рабочие процессы — анализ тайминга, размещение — всё ещё занимают часы и дни, ограничивая возможности инженеров для исследования и оптимизации.

Мы это меняем.

Partcl разрабатывает GPU-ускоренные системы для физического проектирования, работающие на порядки быстрее legacy-инструментов. Цель проста: сделать итерации настолько дешёвыми, чтобы исследование пространства решений стало нормой, а не исключением.

## Базовые статьи

[1] [An Updated Assessment of Reinforcement Learning for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304)

[2] [Assessment of Reinforcement Learning for Macro Placement](https://vlsicad.ucsd.edu/Publications/Conferences/396/c396.pdf)

[3] [Reevaluating Google's Reinforcement Learning for IC Macro Placement](https://cacm.acm.org/research/reevaluating-googles-reinforcement-learning-for-ic-macro-placement/)

[4] [A graph placement methodology for fast chip design](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)

## 🏆 Призы

- **$20,000 — Гран-при:** Топ-7 сабмишенов по proxy score проходят через OpenROAD flow на NG45 designs (включая скрытые). Из этих 7 сабмишен, который победит SA и RePlAce baselines (из статьи [An Updated Assessment of Reinforcement Learning for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304)) с наибольшим запасом по WNS, TNS и Area, выигрывает Гран-при.
- **$20,000 — Первое место (Proxy):** Присуждается сабмишену #1 по proxy score. Выдаётся только если ни один сабмишен не квалифицировался на Гран-при.
- **$5,000 — Второе место:** Присуждается раннер-апу Гран-при. Если ни один сабмишен не квалифицировался на Гран-при — присуждается сабмишену #2 по proxy score.
- **$4,000 — Innovation Award:** Самому креативному или технически инновационному подходу среди топ-сабмишенов, по решению жюри.
- **Swag:** Каждый валидный сабмишен получает HRT swag!
- **Замечание:** Дополнительная корректировка score применяется на основе экспертного человеческого анализа результирующего размещения.

Полные правила scoring для Гран-при, gate по feasibility, тай-брейкинг и обработка ORFS-сбоев — см. [SCORING.md](SCORING.md).

## Формат сабмишена

- Все сабмишены — через Google-форму. Сабмишены могут быть публичными или приватными до окончания судейства.
- Приватные сабмишены обязаны расшарить репозиторий с судьями для клонирования и оценки.
- Команды — до 5 человек.
- Дедлайн сабмишенов: **21 мая 2026, 23:59 PT**.
- Каждая команда может подать только один алгоритм.
- **Все победившие реализации должны быть опубликованы под Apache 2.0 или GPL.**
- Все сабмишены регистрируются через [форму сабмишена](https://forms.gle/YDRtYV5Vq68SZgKW9).
- Все сабмишены должны иметь end-to-end runtime ≤ 1 час на бенчмарк для алгоритма размещения.
- Все сабмишены оцениваются на AMD EPYC 9655P с 16 ядрами + 100GB памяти + NVIDIA RTX 6000 Ada 48GB.

## Дополнительные правила

### Разрешено

- **Любой алгоритмический подход:** SA, RL, GNN, аналитические методы, гибриды, обучение и т.п.
- **Любой фреймворк:** PyTorch, TensorFlow, JAX, или чистый Python/C++.
- **Любая техника оптимизации:** градиентный спуск, эволюционные алгоритмы, локальный поиск и т.п.
- **Обучение на публичных бенчмарках:** можно учиться на IBM benchmark данных.
- **Hard-macro orientation flips** (только Klein-4: `N`, `FN`, `FS`, `S`) — переносятся в Tier 2 через опциональный сайдкар `orientations.pt`.

### Не разрешено

- Модификация функций оценки (нужно использовать TILOS MacroPlacement evaluator как есть).
- Хардкод решений под конкретные бенчмарки (должен быть универсальный алгоритм).
- Использование внешних/проприетарных placement-инструментов (сабмишен должен быть open-source).
- Превышение лимитов времени (1 час на бенчмарк — жёсткий таймаут).
- Пересечения в результирующем размещении (строго ноль перекрытий между hard macros — без допуска. Участники должны добавлять небольшие зазоры в легализации, чтобы избежать float-precision артефактов).
- Повороты hard macros на 90° (`R90`, `R270`, `FE`, `FW`) — fakeram45 SRAM в наших бенчмарках не предназначены для поворотов (доступ к пинам и направление внутреннего металла предполагают фиксированную ориентацию).
- Изменение размеров soft macros — размер soft macro это proxy-only концепт для density/congestion, не переносится в Tier 2; размеры зафиксированы значениями из исходного `.plc` при каждом вызове `compute_proxy_cost`.

## Детали оценки

Оценка двухэтапная:

### Tier 1: Ранжирование по proxy cost (все сабмишены)

Все сабмишены ранжируются по **proxy cost** на 18 IBM benchmarks. Это основная квалифицирующая метрика. Proxy cost вычисляется через TILOS MacroPlacement evaluator:

> **Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion**

Базовые числа взяты из: [An Updated Assessment of Reinforcement Learning for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304).

### Tier 2: OpenROAD Flow Validation (топ сабмишены)

Топ-7 сабмишенов по proxy score оцениваются через полный **OpenROAD flow** на NG45 designs для измерения реальных PnR результатов: **WNS, TNS, Area**.

- **Гран-при ($20K)** присуждается сабмишену с наивысшим score по **взвешенному геометрическому среднему улучшений** по WNS, TNS, Area относительно усреднённого SA/RePlAce baseline.
- Для квалификации сабмишен должен пройти **gate по feasibility** — тайминг (WNS, TNS) не может регрессировать ниже обоих baselines на любом дизайне.
- Чтобы избежать переобучения, мы также оцениваем на 1-2 скрытых NG45 дизайнах.
- **Полные правила скоринга:** [SCORING.md](SCORING.md).

## 🚀 Быстрый старт

### Установка

```bash
# Клонировать репозиторий
git clone https://github.com/partcleda/partcl-macro-place-challenge.git
cd partcl-macro-place-challenge

# Инициализировать TILOS MacroPlacement submodule (нужен для оценки)
git submodule update --init external/MacroPlacement

# Установить пакет и все зависимости
uv sync

# Проверить setup
uv run evaluate submissions/examples/greedy_row_placer.py -b ibm01
```

### Запустить первый пример

```bash
# Запустить greedy row placer на ibm01
uv run evaluate submissions/examples/greedy_row_placer.py

# Запустить на всех 17 IBM бенчмарках
uv run evaluate submissions/examples/greedy_row_placer.py --all

# Запустить на NG45 коммерческих дизайнах (ariane133, ariane136, mempool_tile, nvdla)
uv run evaluate submissions/examples/greedy_row_placer.py --ng45

# Визуализировать результат
uv run evaluate submissions/examples/greedy_row_placer.py --vis
uv run evaluate submissions/examples/greedy_row_placer.py --all --vis
```

Запуск на всех бенчмарках выводит таблицу:
```
Benchmark     Proxy        SA   RePlAce     vs SA  vs RePlAce  Overlaps
   ibm01    2.0463    1.3166    0.9976    -55.4%     -105.1%         0
   ibm02    2.0431    1.9072    1.8370     -7.1%      -11.2%         0
   ...
     AVG    2.2109    2.1251    1.4578     -4.0%      -51.7%         0
```

Greedy placer достигает нуля overlaps, но не пытается оптимизировать wirelength и связность — твоя задача сделать лучше! См. [SETUP.md](SETUP.md) для полного API референса и [`submissions/examples/`](../submissions/examples/) для рабочих примеров.

## 🎯 IBM Benchmark Suite (ICCAD04)

Мы оцениваем на полном ICCAD04 IBM benchmark suite:

| Бенчмарк | Macros | Nets | Canvas (μm) | Util. | SA Baseline | RePlAce Baseline |
|-----------|--------|------|-------------|------------|-------------|------------------|
| **ibm01** | 246 | 7,269 | 22.9×23.0 | 42.8% | 1.3166 | **0.9976** ⭐ |
| **ibm02** | 254 | 7,538 | 23.2×23.5 | 43.1% | 1.9072 | **1.8370** ⭐ |
| **ibm03** | 269 | 8,045 | 24.1×24.3 | 44.2% | 1.7401 | **1.3222** ⭐ |
| **ibm04** | 285 | 8,654 | 24.8×25.1 | 44.8% | 1.5037 | **1.3024** ⭐ |
| **ibm06** | 318 | 9,745 | 26.1×26.5 | 46.1% | 2.5057 | **1.6187** ⭐ |
| **ibm07** | 335 | 10,328 | 26.8×27.2 | 46.8% | 2.0229 | **1.4633** ⭐ |
| **ibm08** | 352 | 10,901 | 27.5×27.9 | 47.4% | 1.9239 | **1.4285** ⭐ |
| **ibm09** | 369 | 11,463 | 28.1×28.5 | 48.0% | 1.3875 | **1.1194** ⭐ |
| **ibm10** | 387 | 12,018 | 28.8×29.2 | 48.6% | 2.1108 | **1.5009** ⭐ |
| **ibm11** | 405 | 12,568 | 29.4×29.8 | 49.2% | 1.7111 | **1.1774** ⭐ |
| **ibm12** | 423 | 13,111 | 30.1×30.5 | 49.8% | 2.8261 | **1.7261** ⭐ |
| **ibm13** | 441 | 13,647 | 30.7×31.1 | 50.4% | 1.9141 | **1.3355** ⭐ |
| **ibm14** | 460 | 14,178 | 31.4×31.8 | 51.0% | 2.2750 | **1.5436** ⭐ |
| **ibm15** | 479 | 14,704 | 32.0×32.4 | 51.6% | 2.3000 | **1.5159** ⭐ |
| **ibm16** | 498 | 15,225 | 32.7×33.1 | 52.2% | 2.2337 | **1.4780** ⭐ |
| **ibm17** | 517 | 15,741 | 33.3×33.7 | 52.8% | 3.6726 | **1.6446** ⭐ |
| **ibm18** | 537 | 16,253 | 34.0×34.4 | 53.4% | 2.7755 | **1.7722** ⭐ |

Каждый бенчмарк содержит:
- Hard macros (ты их размещаешь)
- Soft macros (можешь тоже размещать)
- Сети, соединяющие все компоненты
- Initial placement (рукотворное, как референс)

**Анализ baselines:**
- RePlAce (⭐) стабильно превосходит SA на всех бенчмарках
- RePlAce достигает на 15-55% меньшего proxy cost, чем SA
- **Чтобы квалифицироваться на Гран-при, твоё размещение должно также давать лучше WNS, TNS и Area, чем оба baselines, при оценке через OpenROAD flow на NG45 дизайнах**
- Оба baselines дают ноль overlaps (жёсткое ограничение)

## 💡 Почему это сложно

Несмотря на "всего" 246-537 макросов, задача очень сложная:

1. **Огромное пространство поиска**: ~10^800 возможных размещений (даже с ограничениями).
2. **Конфликтующие цели**: wirelength хочет кластеризации, density хочет распределения, congestion хочет места для маршрутизации.
3. **Невыпуклый ландшафт**: миллионы локальных минимумов, разрывы, плато.
4. **Зависимости с большим радиусом**: перемещение одного макро влияет на стоимости глобально через тысячи нетов.
5. **Жёсткие ограничения**: ноль перекрытий между гетерогенными размерами (33× размах).
6. **Плотная упаковка**: 43-53% утилизации площади оставляет мало запаса.
7. **Время важно**: должно быть достаточно быстро для практического применения (< 5 минут идеально).

Классические методы (SA, RePlAce) оттачивались десятилетиями — но место для улучшений есть!

## 📖 Документация

- **Setup и API референс:** [SETUP.md](SETUP.md) — детали инфраструктуры, формат бенчмарков, вычисление cost, тестирование.
- **Примеры сабмишенов:** [`submissions/examples/`](../submissions/examples/) — рабочие примеры placer'ов.

## 📚 Источники

- **TILOS MacroPlacement:** [GitHub](https://github.com/TILOS-AI-Institute/MacroPlacement)
  - Источник evaluation infrastructure
  - ICCAD04 benchmarks
  - Эталонные реализации SA и RePlAce

- **ICCAD04 Benchmarks:** Классический benchmark-набор для macro placement, используется в академических исследованиях.

## 🏅 Leaderboard

Сабмишены ранжируются по **среднему proxy cost** на всех 17 IBM benchmarks (меньше — лучше). Ноль overlaps требуется на каждом бенчмарке. Скоры неверифицированы до подтверждения судьями.

| Ранг | Команда | Avg Proxy | Best | Worst | Overlaps | Runtime | Verified | Заметки |
|------|------|---------------|------|-------|----------|---------|----------|-------|
| 1 | "Cezar" (ReFine) | **1.2224** | 0.8843 | 1.5115 | 0 | 5min/bench | :white_check_mark: | Verified 1.2224 vs self-reported 1.0666; обходит RePlAce на 16.2%, SA на 42.5%; пересабмит 4/25, оспаривает результат — re-verification pending |
| 2 | "MTK" (DreamPlace++) | **1.2818** | 0.9073 | 1.6529 | 0 | 37s/bench (GPU) | :white_check_mark: | Verified лучше self-reported 1.317; обходит RePlAce на всех 17 бенчмарках |
| 3 | "RoRa" (RipPlace) | **1.3241** | — | — | 0 | 694s/bench | | |
| 4 | "UToronto Analytical" (MOSAIC) | **1.3323** | 0.9371 | 1.6545 | 0 | 24min total | :white_check_mark: | Verified 1.3323 (self-reported 1.3325 — точное совпадение); gradient-based с smooth surrogates, hard+soft |
| 5 | "Shoom" (MultiDREAMPlace) | **1.3381** | — | — | 0 | 350s/bench | | New 4/27; multi-start DREAMPlace + min-displacement legalization + SA |
| 6 | "V5" (TierPlace) | **1.3382** | — | — | 0 | 850s/bench | | New 4/23; GPU-based, multi-density-formulation pilot + phased optimization |
| 7 | "Archgen" (AutoDMP++) | **1.3479** | — | — | 0 | 2404s total | | New 4/24; multi-start + fast proxy screening + bounded refinement |
| 8 | "Electric Beatel" (ePlace-Lite) | **1.3913** | 0.9773 | 1.7253 | 0 | 155s/bench (GPU) | :white_check_mark: | |
| 9 | "Varun's Parallel Worlds" (GRPlace) | **1.4017** | 1.0362 | 1.7298 | 0 | 27s/bench | :white_check_mark: | |
| 10 | "UT Austin" - AS (DREAMPlace Analytical) | **1.4076** | — | — | 0 | 17s/bench | | |
| 11 | "ByteDancer" (Incremental CD) | **1.4151** | 1.0236 | 1.7792 | 0 | 38min/bench | :white_check_mark: | |
| 12 | "vmallela" (Incremental CD+LNS) | **1.4152** | 1.0236 | 1.7817 | 0 | 12h total | :white_check_mark: | Verified 1.4152 (self-reported 1.1172 — на 27% хуже на нашем железе); pure Python+numpy, single-threaded |
| 13 | "TAISPlAce" (ALNS + Thompson Sampling) | **1.4321** | — | — | 0 | 28min/bench | | |
| 14 | "ArzunPD" (HyperPlace SA+LNS) | **1.4421** | 1.0323 | 1.7851 | 0 | 6h total | :white_check_mark: | Verified 1.4421 (self-reported 1.4174); Stage 5 LNS отключён — отсутствует `networkit` зависимость, fallback на Stage 4 на всех бенчмарках |
| 15 | "Pragnay" (SweepingBellPlacement) | **1.4427** | — | — | 0 | 632s/bench | | |
| 16 | "Convex Optimization" (UWaterloo Student) | **1.4556** | 1.0432 | 1.7867 | 0 | 11s/bench | :white_check_mark: | Пересабмит 4/13; исправлен из DQ (было 846 overlaps) |
| 17 | "another Waterloo kid" (Batched Nesterov GP) | **1.4568** | — | — | 0 | 118s/bench | | |
| — | RePlAce (baseline) | **1.4578** | 0.9976 | 1.8370 | 0 | — | :white_check_mark: | |
| 18 | "W3 Solutions" (GRACE) | **1.4824** | — | — | 0 | 90s/bench | | |
| 19 | "Jiangban Ya" (Spectral-Seed + Adaptive Legalizer) | **1.4943** | 1.0891 | 1.8099 | 0 | 49s/bench | :white_check_mark: | |
| 20 | "UTAUSTIN-CT" (PLC-Exact Congestion-Aware SA) | **1.5062** | 1.1363 | 1.7941 | 0 | 6s/bench | :white_check_mark: | |
| 21 | "oracleX" (Oracle) | **1.5130** | 1.1340 | 1.7937 | 0 | 11s/bench | :white_check_mark: | |
| 22 | "SEVmakers" (Hybrid Legalization + SA) | **1.5200** | — | — | 0 | 200s/bench | | Приватный репо — ждём доступа судьям |
| 23 | "CA" (congestion_aware) | **1.5247** | 1.2226 | 1.7945 | 0 | 2s/bench | :white_check_mark: | Verified 1.5247 vs self-reported 1.5238 |
| 24 | "#5 ubc cpen student" (Gene Pool Shuffle) | **1.5337** | 1.1411 | 1.8084 | 0 | 13s/bench | :white_check_mark: | |
| 25 | Will Seed (Partcl) | **1.5338** | 1.1625 | 1.7965 | 0 | 35s total | :white_check_mark: | |
| 26 | "UT Austin" - RH (DREAMPlace) | **1.6037** | — | — | 0 | 4.5s/bench | | |
| 27 | "UT Austin" - CT (PROXYCost) | **1.8706** | — | — | 0 | 187s/bench | | |
| 28 | "AS" (Shelf Stacker) | **1.9121** | 1.4614 | 2.3508 | 0 | 0.16s total | :white_check_mark: | |
| 29 | "Adi's Team" (GNN-ePlace Hybrid) | **2.0025** | — | — | 0 | 3726s/bench | | |
| 30 | "Sharc #1" (Auction Placer) | **2.0433** | 1.5143 | 2.4336 | 0 | 96s/bench | :white_check_mark: | |
| — | SA (baseline) | 2.1251 | 1.3166 | 3.6726 | 0 | — | :white_check_mark: | |
| — | Greedy Row (demo) | 2.2109 | 1.6728 | 2.7696 | 0 | 0.3s total | :white_check_mark: | |
| — | "Binghamton" (feng shui) | pending | — | — | — | — | | |
| — | "MacroBio" (Two-Opt Swap) | pending | — | — | — | — | | |
| DQ | "Mike Gao" (autoresearch) | self-reported 1.3255 | — | — | 1939 | 16min/bench | | DREAMPlace silently сбоит в eval-окружении, возвращает нелегализованные placements (47–189 overlaps на бенчмарк) |
| DQ | "BakaBobo" (Global Relocation Sweep) | self-reported 1.4044 | — | — | — | 282s/bench | | Импортирует `macro_place.fast_proxy`, которого нет в репо и нет в evaluator — код не запускается |

*Сабмить результаты через [форму сабмишена](https://forms.gle/YDRtYV5Vq68SZgKW9)!*

## 🤔 FAQ

**В: Какие бенчмарки используются?**
О: Tier 1 (proxy cost) использует 17 IBM ICCAD04 бенчмарков — стандартный академический набор с устоявшимися baselines. Tier 2 (OpenROAD flow) использует NG45 коммерческие дизайны (ariane133, ariane136, mempool_tile, nvdla) плюс 1-2 скрытых дизайна. Можно оценивать на обоих через `--all` (IBM) и `--ng45` (NG45).

**В: Что если я обхожу один baseline, но не другой?**
О: Чтобы квалифицироваться на Гран-при, нужно обойти ОБА — SA и RePlAce — по WNS, TNS, Area. Можно всё равно выиграть Proxy или Innovation призы независимо от этого.

**В: Есть ли скрытые тест-кейсы?**
О: Все 17 IBM benchmarks для proxy cost ranking — публичные. 4 NG45 designs тоже публичные. Для OpenROAD flow evaluation (Tier 2) дополнительно тестируем на 1-2 скрытых NG45 дизайнах для проверки обобщения.

**В: Что считается "обходом" baseline?**
О: Для proxy cost (Tier 1) твой агрегированный score по всем IBM benchmarks должен быть ниже baselines. Для Гран-при (Tier 2) твои OpenROAD результаты по WNS, TNS и Area должны превышать оба — SA и RePlAce — на NG45 designs.

## 📧 Контакты

- **Issues:** [GitHub Issues](https://github.com/partcleda/partcl-macro-place-challenge/issues)
- **Email:** contact@partcl.com

## 📄 Лицензия

Проект под лицензией Apache License 2.0 — см. [LICENSE.md](../LICENSE.md).

## Обновления соревнования

Организаторы могут обновлять или уточнять правила, детали оценки, сроки, призы или инфраструктуру для обеспечения честности, технической точности и плавной работы соревнования. Любые обновления будут сообщаться через официальные каналы и будут применяться с момента анонса.

Участие в соревновании подразумевает принятие текущих правил и любых последующих обновлений. Решения организаторов касательно скоринга, eligibility и интерпретации правил — окончательные.

Сабмишены и контактная информация могут быть переданы спонсорам.
