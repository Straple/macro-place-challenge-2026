# Условие задачи — Macro Placement Challenge 2026

> Формальная постановка с математикой, ограничениями и примерами. Собрано из [README.md](README.md), [SCORING.md](SCORING.md), [SETUP.md](SETUP.md). Если что-то непонятно — посмотри [GLOSSARY.md](GLOSSARY.md).

---

## TL;DR в трёх предложениях

Дан прямоугольный канвас и набор прямоугольных блоков (макросов), часть из которых жёсткие (hard) — они не должны пересекаться, часть мягкие (soft) — они представляют кластеры стандартных ячеек. Между макросами проложены электрические соединения (неты). Нужно расставить макросы в координатах канваса так, чтобы минимизировать составную метрику **proxy cost = 1.0 × wirelength + 0.5 × density + 0.5 × congestion** при жёстком ограничении: ноль пересечений между hard макросами.

---

## Формальная постановка

### Дано (input)

Один **бенчмарк** = (`Canvas`, `Macros`, `Nets`, `Pins`, `Grid`).

#### Канвас
Прямоугольная область:
```
W × H            где W = canvas_width, H = canvas_height (в μm)
[0, W] × [0, H]  область, в которой могут находиться макросы
```

#### Макросы
Множество $\mathcal{M} = \{m_1, m_2, \ldots, m_n\}$, $n$ = `num_macros`.

Разделены на две группы:
- **Hard macros** — индексы $[0, n_h)$, где $n_h$ = `num_hard_macros`.
- **Soft macros** — индексы $[n_h, n)$, где $n - n_h$ = `num_soft_macros`.

Каждый макрос $m_i$ имеет:
- Размеры: $w_i, h_i > 0$ (`macro_sizes[i]`)
- Позицию центра: $(x_i, y_i)$ (`macro_positions[i]`) — это **исходное** размещение, мы его меняем
- Флаг "зафиксирован": $f_i \in \{0, 1\}$ (`macro_fixed[i]`) — если 1, **двигать нельзя**
- Имя $\text{name}_i$ (для дебага)

> **Жёсткие vs мягкие.** Жёсткие — реальные физические блоки (SRAM, IP, аналог). Мягкие — это абстракция: кластер стандартных ячеек, который для целей Tier 1 представлен одним прямоугольником. **Размер soft макро менять нельзя** ни в одной из фаз.

#### Неты (соединения)
Множество $\mathcal{N} = \{N_1, N_2, \ldots, N_{|\mathcal{N}|}\}$, $|\mathcal{N}|$ = `num_nets`.

Каждый нет $N_k$ — это **гиперребро**: множество пинов, соединённых одним проводом. Каждый пин принадлежит какому-то макросу $m_i$ и имеет смещение $(\delta_{ij}^x, \delta_{ij}^y)$ относительно центра макро (для slot $j$). Координата пина:
$$
\text{pin}_{ij} = (x_i + \delta_{ij}^x, \ y_i + \delta_{ij}^y)
$$

Нет имеет вес $w_{N_k}$ (`net_weights[k]`).

#### Сетка для density/congestion
Канвас разбит на $G_r \times G_c$ ячеек (`grid_rows × grid_cols`). Параметры маршрутизации:
- $\rho_h$ горизонтальных треков на μm (`hroutes_per_micron`)
- $\rho_v$ вертикальных треков на μm (`vroutes_per_micron`)

#### Пример: ibm01
| Параметр | Значение |
|---|---|
| Canvas | 22.9 μm × 23.0 μm |
| Hard macros | 246 |
| Soft macros | 894 |
| Nets | 7,269 |
| Утилизация | 42.8% |

(Полная таблица 17 IBM benchmarks — в [README.md:165-186](../README.md#L165))

---

### Найти (output)

**Размещение** — тензор $\mathbf{P} \in \mathbb{R}^{n \times 2}$:
$$
\mathbf{P} = \begin{pmatrix} x_1 & y_1 \\ x_2 & y_2 \\ \vdots & \vdots \\ x_n & y_n \end{pmatrix}
$$

где $(x_i, y_i)$ — **координаты центра** макро $m_i$ в μm. **Не углы!**

В коде:
```python
def place(self, benchmark: Benchmark) -> torch.Tensor:
    # Возвращает torch.Tensor формы [num_macros, 2]
    ...
```

---

### Минимизировать (целевая функция)

**Proxy cost** на бенчмарке $B$:
$$
\boxed{\text{ProxyCost}(\mathbf{P}, B) = 1.0 \cdot \text{WL}(\mathbf{P}, B) + 0.5 \cdot D(\mathbf{P}, B) + 0.5 \cdot C(\mathbf{P}, B)}
$$

Финальный score Tier 1 — **среднее по 17 IBM-бенчмаркам**:
$$
\text{Score}_{\text{Tier 1}} = \frac{1}{17} \sum_{B \in \text{IBM}} \text{ProxyCost}(\mathbf{P}_B, B)
$$

Меньше — лучше.

**Baselines (порог):**
- SA: 2.1251 (слабее)
- **RePlAce: 1.4578** ⬅ это нужно побить, чтобы хоть как-то светить

---

### При условии (ограничения)

#### 1. Никаких пересечений между hard macros

Для любых двух **hard** макросов $i, j$ ($i \neq j$, $i, j < n_h$):
$$
|x_i - x_j| \geq \frac{w_i + w_j}{2} \quad \lor \quad |y_i - y_j| \geq \frac{h_i + h_j}{2}
$$

(Хотя бы по одной оси они должны быть достаточно далеко.)

> **Для soft макросов это не требование** — soft могут пересекаться. Они абстракция.

> **Tolerance — ноль.** Float-precision overlap = DQ. Поэтому в легализации стандартно добавляют gap ≥ 0.001 μm.

#### 2. Все макросы внутри канваса

Для всех $i = 1..n$:
$$
\frac{w_i}{2} \leq x_i \leq W - \frac{w_i}{2}, \quad \frac{h_i}{2} \leq y_i \leq H - \frac{h_i}{2}
$$

(Учитывая что $(x_i, y_i)$ — центр.)

#### 3. Зафиксированные макросы остаются на месте

Для всех $i$ с $f_i = 1$:
$$
(x_i, y_i) = (x_i^{\text{init}}, y_i^{\text{init}})
$$

#### 4. Размеры soft макросов не меняем

При каждом вызове `compute_proxy_cost` размеры soft макросов восстанавливаются из исходных `.plc`. Невозможно "схитрить", уменьшив их.

#### 5. Ориентации макросов — только Klein-4

Разрешены: `N`, `FN`, `FS`, `S` (нормальная, отражение по вертикали, поворот на 180° + отражение, поворот на 180°).
**Запрещены:** `R90`, `R270`, `FE`, `FW` (повороты на 90° → fakeram45 SRAM пины не работают).

Ориентации передаются в Tier 2 через опциональный сайдкар `orientations.pt`. Если не указано — все макросы остаются в `N`.

#### 6. Время

**Runtime ≤ 1 час** на бенчмарк (hard timeout). Оценочное железо:
- AMD EPYC 9655P, 16 ядер
- 100 GB RAM
- NVIDIA RTX 6000 Ada, 48 GB

#### 7. Алгоритмический honesty

- Один универсальный алгоритм для всех бенчмарков (никакого `if benchmark.name == 'ibm01': ...`)
- Нельзя модифицировать evaluator (`macro_place/`)
- Нельзя использовать проприетарные placement-инструменты
- Открытый код (Apache 2.0 / MIT при выигрыше призов)

---

## Компоненты целевой функции

### Wirelength (WL) — длина проводов

Используется **HPWL (Half-Perimeter Wirelength)**: для каждого нета строится bounding box по координатам пинов, берётся половина периметра. Нормализованная сумма по всем нетам:
$$
\text{HPWL}(N_k) = \max_{p \in N_k} x_p - \min_{p \in N_k} x_p + \max_{p \in N_k} y_p - \min_{p \in N_k} y_p
$$
$$
\text{WL} = \sum_{k=1}^{|\mathcal{N}|} w_{N_k} \cdot \text{HPWL}(N_k) \cdot \text{(normalization)}
$$

Хотим минимизировать → макросы, связанные многими нетами, должны быть **близко друг к другу**.

### Density (D) — плотность

Канвас разбит на сетку $G_r \times G_c$. В каждую ячейку $(r, c)$ суммируется площадь макросов, попавших в неё. Density — топ-10% самых плотных ячеек (после нормализации):
$$
D = \text{top}_{10\%} \left( \frac{\text{area\_in\_cell}(r, c)}{\text{cell\_size}} \right)
$$

Хотим минимизировать → макросы должны быть **распределены равномерно**, не сваливаться в кучу.

### Congestion (C) — перегруженность маршрутизации

Для каждого horizontal/vertical routing-сегмента считается, сколько проводов через него попытаются пройти (на основе HPWL bbox'ов нетов). Congestion — топ-5% самых перегруженных:
$$
C = \text{top}_{5\%} (\text{routing\_demand} / \text{routing\_capacity})
$$

Хотим минимизировать → должно остаться **место для проводов** между макросами.

### Конфликт целей

Эти три слагаемых **тянут в разные стороны**:
- WL хочет, чтобы связанные макросы были близко → кластеризация
- Density хочет, чтобы макросы были разбросаны → анти-кластеризация
- Congestion хочет, чтобы остался простор для проводов → промежутки

Поэтому хорошее решение — **компромисс**, и поэтому задача нетривиальна.

---

## Tier 2 — Гран-при ($20K)

Только для **топ-7 сабмишенов по Tier 1**.

### Что оценивается

Полный OpenROAD flow (synthesis → floorplan → placement → CTS → routing) на 4 публичных NG45 дизайнах + 1-2 скрытых:
- `ariane133` (RISC-V CPU, 133 макроса)
- `ariane136` (RISC-V CPU, 136 макросов)
- `mempool_tile` (memory, 20 макросов)
- `nvdla` (AI accelerator, 128 макросов)
- `???` (1-2 hidden — для anti-overfitting)

Измеряется три метрики на дизайн:
- **WNS** (Worst Negative Slack, нс) — выше (ближе к 0) лучше
- **TNS** (Total Negative Slack, нс) — выше (ближе к 0) лучше
- **Area** (μm²) — меньше лучше

### Stage 1 — Feasibility gate

Для **каждого** дизайна:
$$
\text{WNS}_{\text{sub}} \geq \min(\text{WNS}_{\text{SA}}, \text{WNS}_{\text{RP}})
$$
$$
\text{TNS}_{\text{sub}} \geq \min(\text{TNS}_{\text{SA}}, \text{TNS}_{\text{RP}})
$$

Если хотя бы на одном дизайне условие нарушено → DQ из Гран-при (но Tier 1 ranking остаётся).

### Stage 2 — Scoring (взвешенное геометрическое среднее)

Для каждого дизайна:
$$
R_{\text{WNS}} = \frac{\text{WNS}_{\text{avg}}}{\text{WNS}_{\text{sub}}}, \quad R_{\text{TNS}} = \frac{\text{TNS}_{\text{avg}}}{\text{TNS}_{\text{sub}}}, \quad R_{\text{Area}} = \frac{\text{Area}_{\text{avg}}}{\text{Area}_{\text{sub}}}
$$

где $\text{X}_{\text{avg}} = (\text{X}_{\text{SA}} + \text{X}_{\text{RP}}) / 2$.

Per-design score (веса WNS:TNS:Area = 3:2:1):
$$
\text{Score}_{\text{design}} = \left( R_{\text{WNS}}^3 \cdot R_{\text{TNS}}^2 \cdot R_{\text{Area}}^1 \right)^{1/6}
$$

Total score:
$$
\text{Score}_{\text{Tier 2}} = \left( \prod_{d \in \text{designs}} \text{Score}_d \right)^{1/N}
$$

Больше — лучше. Score > 1 значит "лучше среднего baseline'а".

Полные правила, тай-брейкинг, обработка edge cases (WNS=0, ORFS-сбои) — в [SCORING.md](SCORING.md).

---

## Призы

| Приз | Сумма | Кому |
|---|---|---|
| Гран-при | $20,000 | Лучший по Score Tier 2 (если есть feasible-сабмишен) |
| Первое место (Proxy) | $20,000 | #1 по Tier 1, если **никто** не квалифицировался на Гран-при |
| Второе место | $5,000 | Раннер-ап Гран-при (или #2 по Tier 1, если Гран-при не присуждён) |
| Innovation Award | $4,000 | Самый креативный/технически инновационный подход (по решению жюри) |
| Swag | — | Каждый валидный сабмишен |

Все суммы в долларах США.

> Важно: Grand Prize **присуждается только если есть сабмишен, прошедший feasibility gate**. Если ни один из топ-7 не прошёл — Grand Prize не выдаётся, $20K идут как First Place (Proxy).

---

## Submission

### Формат

- Один Python-класс с методом `place(benchmark) → torch.Tensor` в файле `submissions/<твоя_команда>/placer.py`
- См. минимальный шаблон в [API_CHEATSHEET.md](API_CHEATSHEET.md)

### Регистрация

- Через [Google form](https://forms.gle/YDRtYV5Vq68SZgKW9)
- Команда до 5 человек
- Один алгоритм на команду (можно пересабмитить — см. [README.md:65-66](../README.md#L65))

### Дедлайн

**21 мая 2026, 23:59 PT** (Pacific Time, тихоокеанский часовой пояс).

### Лицензия

Если выигрываешь cash prize → реализация должна быть открыта под Apache 2.0 или MIT.

---

## Открытость тестов — важное практическое следствие

**Tier 1 — все 17 IBM benchmarks полностью открыты** ([README.md:281-282](../README.md#L281)):

> Q: Are there hidden test cases?
> A: All 17 IBM benchmarks for proxy cost ranking are public.

То есть:
- Те же `netlist.pb.txt` и `initial.plc`, что лежат у тебя в `external/MacroPlacement/Testcases/ICCAD04/ibmXX/`, судьи **используют без изменений** для оценки.
- **Self-reported AVG = verified AVG** (с поправкой на железо). Не будет сюрпризов вроде vmallela, у которого 1.12 → 1.42 из-за single-threaded numpy.
- Можно глубоко анализировать каждый benchmark индивидуально.
- Можно тренировать ML-модели (GNN, RL) **на этих самых данных** — это разрешено.

**Tier 2 — частично скрыто:**
- 4 публичных NG45 дизайна (`ariane133`, `ariane136`, `mempool_tile`, `nvdla`) лежат в репо.
- + **1-2 скрытых NG45** — для анти-overfitting'а. Узнаешь о них, только если попадёшь в топ-7.

**Что НЕ разрешено** ([README.md:88](../README.md#L88)):
> Hardcoding solutions for specific benchmarks (must be general algorithm)

Запрещён `if benchmark.name == "ibm17": return precomputed_solution()`. Алгоритм должен быть **один универсальный**, но **гиперпараметры и адаптивная логика по структуре дизайна (например `if num_macros > 500: ...`) разрешены**. Граница тонкая — судьи могут проводить аудит кода, ища подозрительные lookup-таблицы и magic numbers по именам.

---

## Где взять данные для разработки

### IBM benchmarks (Tier 1)

```bash
git submodule update --init external/MacroPlacement
# Бенчмарки в external/MacroPlacement/Testcases/ICCAD04/ibm{01..18}/
```

Или pre-processed:
```python
from macro_place.benchmark import Benchmark
b = Benchmark.load("benchmarks/processed/public/ibm01.pt")
```

(Все 17 + 4 NG45 + ASAP7 версии лежат в [../benchmarks/processed/public/](../benchmarks/processed/public/) — можно работать без submodule для отладки логики.)

### NG45 designs (Tier 2)

```
external/MacroPlacement/Flows/NanGate45/{ariane133,ariane136,mempool_tile,nvdla}/
```

### ICCAD04 paper

Бенчмарки описаны в оригинальной статье ICCAD'04: "Mixed-Size Placement: A Floor Plan-Aware Approach". Скачать с [TILOS репо](https://github.com/TILOS-AI-Institute/MacroPlacement).

---

## Что особенно сложного

(Из [README.md:199-211](../README.md#L199), вольно)

1. **Огромное пространство поиска** — ~10⁸⁰⁰ возможных размещений даже с ограничениями.
2. **Конфликтующие цели** — wirelength/density/congestion тянут в разные стороны.
3. **Невыпуклый ландшафт** — миллионы локальных минимумов.
4. **Дальние зависимости** — перемещение одного макро влияет глобально через тысячи нетов.
5. **Жёсткие constraints** — ноль перекрытий при размахе размеров 33×.
6. **Плотная упаковка** — 43-53% утилизации, мало запаса.
7. **Время важно** — < 5 минут идеально, ≤ 1 час обязательно.

Классика (SA, RePlAce) затачивалась десятилетиями — но место для улучшений есть. Это и подтверждает leaderboard, где топ-1 на 16% лучше RePlAce.

---

## Что дальше читать

- Алгоритмические подходы (что работает, что нет): [ALGORITHMS.md](ALGORITHMS.md)
- API для написания placer'а: [API_CHEATSHEET.md](API_CHEATSHEET.md), [SETUP.md](SETUP.md)
- Стратегия и план команды: [todo.md](todo.md)
- Полные правила Гран-при с примером: [SCORING.md](SCORING.md)
- Чеклист перед сабмитом: [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)
