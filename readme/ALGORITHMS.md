# Алгоритмы macro placement — обзор подходов

> Что существует, кто это использует в leaderboard, плюсы/минусы, когда брать. Ссылки на реализации в репо и внешние референсы.

## Карта подходов

```
                  ┌─────────────────────────────────────┐
                  │  Простые / baseline (для разогрева)  │
                  ├─────────────────────────────────────┤
                  │ • Random                             │
                  │ • Greedy / Shelf-pack                │
                  │ • Force-directed                     │
                  └─────────────────────────────────────┘

                  ┌─────────────────────────────────────┐
                  │  Метаэвристики (классика, baseline)  │
                  ├─────────────────────────────────────┤
                  │ • Simulated Annealing (SA)           │
                  │ • LNS / ALNS                         │
                  │ • Gene Pool / Genetic                │
                  └─────────────────────────────────────┘

                  ┌─────────────────────────────────────┐
                  │  Аналитические (топ leaderboard)     │
                  ├─────────────────────────────────────┤
                  │ • Quadratic placement                │
                  │ • RePlAce                            │
                  │ • ePlace (electrostatic)             │
                  │ • DREAMPlace (GPU ePlace) ⭐          │
                  └─────────────────────────────────────┘

                  ┌─────────────────────────────────────┐
                  │  Структурные (хорошие seed'ы)        │
                  ├─────────────────────────────────────┤
                  │ • Spectral (eigenvectors Лапласиана) │
                  │ • Min-cut bisection                  │
                  └─────────────────────────────────────┘

                  ┌─────────────────────────────────────┐
                  │  Обучаемые (экспериментально)        │
                  ├─────────────────────────────────────┤
                  │ • RL (Google paper, спорная)         │
                  │ • GNN-предсказание                   │
                  │ • Bandit / RL для destroy operators  │
                  └─────────────────────────────────────┘
```

---

## 1. Random / Greedy / Shelf-pack

### Random
Случайные координаты внутри canvas.

- **Плюсы:** одна строка кода, годится как unit-test
- **Минусы:** overlaps, ужасный proxy (~1.9-2.5)
- **Где:** [submissions/examples/simple_random_placer.py](../submissions/examples/simple_random_placer.py)

### Greedy / Shelf-pack
Сортируем по высоте, заполняем рядами слева направо.

- **Плюсы:** zero overlaps бесплатно, быстро (0.3s total на 17 IBM), детерминировано
- **Минусы:** игнорирует netlist → плохой wirelength (~2.21 avg)
- **Где:** [submissions/examples/greedy_row_placer.py](../submissions/examples/greedy_row_placer.py)
- **Leaderboard:** `Greedy Row demo` = 2.2109; `AS (Shelf Stacker)` = 1.9121

### Force-directed
Макросы соединены пружинами через net'ы (притягиваются), отталкиваются для density. Итеративно двигаем.

- **Плюсы:** интуитивно, хорошо стартует
- **Минусы:** легко застревает в локальном минимуме, нужна явная legalization
- **Где использовать:** в качестве seed'а или soft macro updater (как `plc.optimize_stdcells()`)

---

## 2. Simulated Annealing (SA)

Случайные ходы (shift / swap / move-toward-neighbor), принимаем худшее с убывающей температурой по правилу Метрополиса:
```
P(accept) = exp(-(new_cost - old_cost) / T)
```

- **Плюсы:** не застревает в локальных минимумах, легко комбинируется с любым cost'ом
- **Минусы:** медленный (10⁴-10⁶ итераций), нужно подбирать schedule
- **Baseline в соревновании:** **2.1251 avg** — заметно слабее RePlAce
- **Лучшие в leaderboard:** `UTAUSTIN-CT (PLC-Exact Congestion-Aware SA)` = 1.5062, `SEVmakers (Hybrid Legalization + SA)` = 1.5200
- **Где в репо:** [submissions/will_seed/placer.py](../submissions/will_seed/placer.py) — SA-refinement с 3 типами ходов (SHIFT/SWAP/MOVE_TOWARD_NEIGHBOR)

### Когда брать
- Как **refinement-фазу** после хорошего seed (DRP+SA = топ-5 у Shoom)
- Когда нужна простота и нет GPU
- Не как одиночное решение для топ-10

---

## 3. LNS / ALNS

**Large Neighborhood Search:** на каждой итерации **destroy** часть текущего решения (убираем k макросов) и **repair** (восстанавливаем иначе), принимаем улучшение.

**Adaptive LNS:** держим набор destroy/repair operator'ов, веса обновляются по успеху каждого.

### Destroy operators (примеры)
- Random k macros
- Spatially-clustered (соседи по координатам)
- Worst-cost (макросы с худшим вкладом в HPWL)
- Net-based (макросы из самого "плохого" нета)

### Repair operators (примеры)
- Greedy min-delta-cost insertion
- Mini-SA на subset
- Force-directed re-insertion
- Random insertion

- **Плюсы:** мощный refinement, работает на любом seed'е, гибкий
- **Минусы:** сам по себе застревает на ~1.42-1.45 (нужен хороший seed!)
- **Leaderboard (чистые LNS):** `TAISPlAce (ALNS+Thompson Sampling)` = 1.4321, `ArzunPD (HyperPlace SA+LNS)` = 1.4421, `vmallela (Incremental CD+LNS)` = 1.4152
- **Гибрид DREAMPlace+LNS:** `Shoom (MultiDREAMPlace+SA)` = 1.3381 — попадает в топ-5

### Когда брать
- **Поверх DREAMPlace** seed'а — наша стратегия, см. [todo.md](todo.md)
- Если нет GPU — alternative к чистому SA
- Для Innovation Award — если придумаешь что-то нестандартное (RL-bandit для operator'ов и т.п.)

---

## 4. Аналитические (RePlAce / ePlace / DREAMPlace)

**Идея:** заменить overlap-constraint и discrete WL дифференцируемыми surrogate'ами, потом градиентный спуск.

- **Wirelength** заменяется на log-sum-exp: `WL ≈ log(Σ exp(x_i/γ))` (smooth maximum)
- **Density** заменяется на bell-shaped или electrostatic potential

### RePlAce (CMU/UC San Diego)
Аналитический placement с density-aware penalty.
- **Плюсы:** топ-результаты в академии, надёжно работает
- **Минусы:** CPU, медленно на больших дизайнах
- **Baseline в соревновании:** **1.4578 avg** — порог отсечки

### ePlace
Использует electrostatic-аналогию: density как заряд, gradient = electric field.
- Базис для DREAMPlace

### DREAMPlace ⭐
GPU-ускоренная PyTorch-реализация ePlace с CUDA-ядрами.
- **Плюсы:** **топ-2 leaderboard** (37s/bench!), open-source, активно развивается
- **Минусы:** требует GPU, специфический setup, силы DREAMPlace в тонкой настройке (DRP++)
- **Где взять:** [github.com/limbo018/DREAMPlace](https://github.com/limbo018/DREAMPlace) (community fork: [tilos-ai-institute/DREAMPlace](https://github.com/TILOS-AI-Institute/DREAMPlace))
- **Leaderboard на DREAMPlace:**
  - `MTK (DreamPlace++)` = 1.2818 — #2 (37s/bench, GPU)
  - `Shoom (MultiDREAMPlace+SA)` = 1.3381 — #5
  - `UT Austin AS (DREAMPlace Analytical)` = 1.4076 — #10 (17s/bench)
  - `Mike Gao (autoresearch)` = **DQ** — DRP silently provides unlegalized placements в eval env! Учиться на ошибках.

### AutoDMP (`Archgen` #7)
Multi-start DREAMPlace + screening — берём несколько runs, оставляем лучший. = 1.3479.

### Когда брать
- Когда есть GPU — **наш план "DRP как seed"**
- Это де-факто стандарт для топ-10 leaderboard
- Если без GPU — пиши свой mini-analytical (gradient-based с smooth surrogates), как UToronto MOSAIC #4 = 1.3323

---

## 5. Spectral / Min-cut

### Spectral placement
Считаем **Laplacian** netlist-графа, берём первые `k` собственных векторов как координаты — макросы со связанными net'ами оказываются рядом.

- **Плюсы:** хороший seed без overlap-aware фазы, быстро (через `scipy.sparse.linalg.eigsh`)
- **Минусы:** не учитывает density/canvas-shape, нужна legalization
- **Leaderboard:** `Jiangban Ya (Spectral-Seed + Adaptive Legalizer)` = 1.4943 (#19)

### Min-cut bisection
Рекурсивно делим netlist на половины с минимальным числом пересекающих net'ов (через `metis` / `KaHIP`), пока не дойдём до отдельных позиций.

- **Плюсы:** классика 90-х, до сих пор работает для seed
- **Минусы:** не оптимально, нужен refinement сверху
- **Используется в:** многих коммерческих flow'ах для floorplanning

---

## 6. Обучаемые подходы

### RL (Google Nature 2021)
Policy network размещает макросы один за другим, награда = -proxy_cost.
- **Плюсы:** в идеале — обучается под benchmark distribution, потенциально лучше всех
- **Минусы:** очень спорная воспроизводимость (см. paper #2 и #3 в README), требует огромных compute, hardcoded под benchmark'и не разрешён
- **Leaderboard:** нет известных RL-сабмишенов в топ-10

### GNN-предсказание placement
Графовая нейросеть на netlist'е → координаты или порядок инсёрта.
- **Плюсы:** может выучить хорошие паттерны
- **Минусы:** обучение долгое, генерализация под скрытые benchmark'и сомнительна
- **Leaderboard:** `Adi's Team (GNN-ePlace Hybrid)` = 2.0025 (#29) — пока не топ

### Замечание про обучение (Tier 1)

**Tier 1 тесты открытые** — все 17 IBM benchmarks доступны и нам, и судьям ([README.md FAQ](../README.md#L281)). Это означает:

✅ **Можно обучать модели** на этих самых данных. Например:
- GNN, тренированная на 16 IBM, тестируется на 17-м (leave-one-out CV) — ловим overfitting
- RL agent, обученный с reward = -proxy_cost на 17 IBM
- Эмбеддинги netlist-структур из обучения на наборе

❌ **Нельзя хардкодить** под имя/конкретный benchmark. Граница такая:
- `if num_macros > 500: more_iters` — ✅ адаптивная логика по структуре
- `if benchmark.name == "ibm17": ...` — ❌ DQ
- `lookup_table[benchmark.name]` — ❌ DQ
- Веса нейронки, дискриминирующие конкретные benchmark'и через паттерны нетлиста — серая зона, может быть DQ при аудите

⚠️ **Tier 2 (NG45) — частично скрыт.** 4 публичных дизайна + 1-2 hidden. Если оптимизируешься только под публичные — рискуешь провалить Гран-при на скрытых. Train на разных типах дизайнов, не только NG45.

### Bandit / RL для destroy operators (для LNS)
Многоруковый бандит выбирает destroy operator на основе их прошлой успешности.
- **Плюсы:** просто, работает (TAISPlAce использует Thompson Sampling)
- **Подходит для Innovation Award**

---

## 7. Гибридные подходы (что делают топ-3-7)

### Pattern 1: Аналитический + SA refinement
**Shoom (#5):** Multi-start DREAMPlace → min-displacement legalization → SA. = 1.3381.

### Pattern 2: Multi-start + screening
**Archgen (#7):** Несколько DREAMPlace runs с разными seeds → быстрый proxy screening → bounded refinement. = 1.3479.

### Pattern 3: Phased GPU optimization
**V5 TierPlace (#6):** GPU, multi-density-formulation pilot + phased optimization. = 1.3382.

### Pattern 4: Joint hard+soft аналитический
**UToronto MOSAIC (#4):** Gradient-based с smooth surrogates, оптимизирует hard и soft вместе. = 1.3323.

### Pattern 5: Incremental coordinate descent
**ByteDancer (#11):** Incremental CD = 1.4151 (38min/bench). Двигаем по одному макро за раз с локальной оптимизацией.

---

## Что выбрать нам

С учётом стратегии в [todo.md](todo.md): **DREAMPlace seed → LNS refinement**.

| Этап | Наш выбор | Альтернатива |
|---|---|---|
| **Seed** | DREAMPlace (если получится прицепить) | Свой analytical (gradient-based с log-sum-exp WL + bell density) → как MOSAIC |
| **Refinement** | ALNS с adaptive operator selection | Чистый SA как у `will_seed` |
| **Soft macros** | Joint в seed-фазе или `plc.optimize_stdcells()` периодически | Свой быстрый force-directed на GPU |
| **Multi-start** | Если время позволит — несколько DRP runs с разных seeds | — |
| **Innovation hook** | RL-bandit на destroy operators | Predictive cost neural net |

---

## Сравнительная таблица из leaderboard

| Подход | Чистый score | Гибрид со seed | Вычислительная стоимость |
|---|---|---|---|
| Random / Greedy | 2.0-2.2 | — | <1s |
| SA | 2.13 (TILOS baseline) | 1.50-1.55 (Hybrid+SA) | минуты |
| RePlAce | 1.46 (TILOS baseline) | — | минуты CPU |
| LNS / ALNS | 1.42-1.45 | 1.34-1.42 (с seed) | минуты-часы |
| DREAMPlace | 1.41 (UT Austin AS plain) | **1.28-1.34** | **секунды-минуты GPU** |
| Spectral seed | 1.49 (Jiangban Ya) | — | секунды |
| Аналитический custom (MOSAIC) | 1.33 | — | минуты CPU |
| GNN | 2.0 (Adi's Team) | — | долгое обучение |

**Главные уроки:**
1. Без хорошего seed нельзя в топ-10
2. DREAMPlace — самый простой путь к топ-10
3. Refinement (SA/LNS) поверх DRP даёт ~3-5% extra
4. Multi-start работает, если есть compute
5. Soft macros нельзя игнорировать
