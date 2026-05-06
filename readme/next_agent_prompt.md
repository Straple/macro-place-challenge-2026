# Промпт для следующего агента

Скопируй всё это в новую сессию Claude.

---

Привет. Ты продолжаешь работу над Macro Placement Challenge 2026. Контекст:

## TL;DR

GPU-based pure-gradient placer на T4 (Yandex Cloud, `89.169.164.17`, user `evyukhnevich`).
Текущий best на **ibm01 = 0.9453** (vs ALNS submission 1.0584 = **−10.7%**, vs MTK target 0.91 = **+3.9%**).
Время run = 10 мин. Лимит judging system = **1 час per bench** — можем использовать **до 50 мин на gradient + 10 мин на eval/legalize**.

**Работать ТОЛЬКО на сервере, ТОЛЬКО ibm01**, пока пользователь не скажет другое.

## Главные файлы

- `submissions/straple/gradient_batch.py` — GPU batch K-параллельный gradient placer (~700 строк)
- `scripts/gpu_run_one.py` — runner с eval+legalize+HTML pipeline
- `scripts/make_simple_viz.py` — JS canvas HTML viewer (без matplotlib)
- `run_remote.sh` — push/run/pull для сервера
- `readme/todo.md` секция 11-12 — последняя сессия + план идей
- `readme/results.md` cycle #30 — таблица breakthrough на ibm01

## Текущий best config

```bash
STRAPLE_BATCH_EPLACE=1 STRAPLE_BATCH_EPLACE_GRID=128 \
STRAPLE_BATCH_CONG_W=10 \
STRAPLE_BATCH_COHESION_START=5 STRAPLE_BATCH_COHESION_END=0.001 \
STRAPLE_BATCH_DIVERSITY=1 \
STRAPLE_BATCH_OVERLAP_FORM=rect_quad \
./run_remote.sh gpu --bench ibm01 --K 384 --time-budget 600 --no-vis
```

→ proxy ≈ 0.945-0.97 (variance ±0.02 между runs)

## Pipeline (gpu_run_one.py)

1. **gradient_batch K=384** на GPU (5-10 мин) — параллельный градиентный спуск 384 seeds
2. **Eval top-32** + C++ legalize (~2 мин)
3. **Parallel legalize ВСЕХ 384 seeds** через mp.Pool 16 workers (~2.5 мин)
4. **Distribution stats** — min/p05/p25/median/mean±σ/p75/p95/max + 90% CI (новое, тестируй)
5. **Save** best pkl + pos_K npz + stats json
6. (optional) **HTML viz** (simple JS canvas, ~1-2 мин)

Loss components:
```
loss = WL_smooth(pin-LSE) + λ_d × density(ePlace_FFT) + λ_o × overlap(rect_quad)
       + cong_w × congestion(top-10% smooth bbox demand)
       + cohesion(dynamic_centroid) × β_decay
```

## Цели от пользователя (приоритет)

### 🎯 Цель 1 — Distribution stats first
Запусти один run с **новым distribution stats** (уже в коде в gpu_run_one.py, в первой строке после parallel legalize). Покажи пользователю:
- min / p05 / p25 / median / mean±σ / p75 / p95 / max
- 90% confidence interval
- valid count / K

Это поможет оценить **качество** gradient'а, не только лучший случай.

### 🎯 Цель 2 — Plateau escape (random ops)

Реализуй в `gradient_batch.py`:
- **Detect plateau per K**: track loss[k] trajectory, если |Δ| < ε за last 30 steps → seed k в плато
- **Random ops** для seeds в плато (выполнять не каждый step, раз в 50-100 steps):
  - **Teleport** k=20 random macros в random position (cluster-aware: keep cluster compact = teleport вместе с anchor)
  - **Swap k pairs**: обменять positions
  - **Cluster shake**: gaussian σ=0.05·canvas для всех макросов одного random кластера
- После op — zero Adam exp_avg/exp_avg_sq для перетеленных macros (не тянуть назад)
- Trigger: только в фазе 2 (refining), не в spreading или settling
- env vars: `STRAPLE_BATCH_PLATEAU_OPS=1`, `STRAPLE_BATCH_PLATEAU_PATIENCE=30`

### 🎯 Цель 3 — Genetic evolution ⭐⭐⭐ (главная фича)

Каждый seed K = "геном" = `pos[k, n_total, 2]`. После каждых N итераций (или при plateau):

1. **Compute fitness** (proxy proxy через legalize или approximate из current loss)
2. **Selection**: keep top-K/4 (elite), убить худшие 3K/4
3. **Crossover** для каждого нового потомка:
   - Pick parent_A, parent_B из elite (random)
   - **Subset crossover**: для каждого cluster c подбрось монетку → pos[c, :] от A или B
   - ИЛИ **half-half**: pos[macros_in_first_half] от A, остальные от B
4. **Mutation**: gaussian noise σ=0.02·canvas + 5% chance random teleport per macro
5. **Adam state reset** для новых genomes

Важно: реализуй **аккуратно через GPU tensors** (не Python loop по K).

env vars: `STRAPLE_BATCH_GA_ENABLE=1`, `STRAPLE_BATCH_GA_INTERVAL=200`, `STRAPLE_BATCH_GA_ELITE_PCT=0.25`, `STRAPLE_BATCH_GA_MUTATION_RATE=0.05`, `STRAPLE_BATCH_GA_MUTATION_SIGMA=0.02`

Это самая крутая идея — каждый K seed имеет свою trajectory, лучшие "размножаются" с recombination.

### 🎯 Цель 4 — Submission run

После реализации plateau ops + GA, запустить с **time_budget=3000 (50 мин)** на ibm01 и измерить:
- Best proxy
- Distribution stats
- Сравнение с current best 0.9453

Цель: **< 0.91** (бить MTK).

## Технические заметки

### Server (Yandex Cloud)
- IP: `89.169.164.17`
- User: `evyukhnevich`
- GPU: NVIDIA T4 (16 GB)
- 16 vCPU, 64 GB RAM
- При stop/start VM Yandex меняет host key — приходится `ssh-keygen -R 89.169.164.17` и заново принимать
- nvidia driver уже установлен и `sudo modprobe nvidia` должно работать без reboot
- uv env уже установлен в `~/macro-place/.venv`

### Использовать
- `./run_remote.sh push` — синк code (БЕЗ vis/, results/)
- `./run_remote.sh ssh` — interactive shell
- `./run_remote.sh gpu --bench ibm01 --K 384 --time-budget 600 --no-vis` — single run
- `./run_remote.sh pull` — стянуть vis/, results/, .remote_runs/ обратно
- Для backgrounded long runs — пиши `nohup` напрямую через ssh:
  ```
  ssh evyukhnevich@89.169.164.17 "cd macro-place && nohup ... > .remote_runs/run.log 2>&1 &"
  ```
  потом `Monitor` на лог.

### НЕ запускать
- Local benchmarks (только server)
- Multi-seed-runs с разными RNG seed (не diversity, просто шум)
- gauss_overlap (создаёт circular dead zones, MTK от него отказались)
- anchor_loss с cohesion одновременно (конфликтуют)

### Текущие memory limits на T4
- K=384 + ePlace 128 chunked = 9.3 GB peak (safe)
- K=512 + ePlace 128 = OOM (нужен chunk_n меньше)
- ePlace 256 = OOM даже на K=128
- pos snapshots каждый step = +875 MB CPU memory (для K=384, n=1140, 250 шагов)

### Variance ±0.02 между runs
Один и тот же конфиг даёт разный best (0.9453 ↔ 0.9667). Это потому что **GPU non-determinism** + **Louvain Python random**. Без fix random_seed это нормально.

## История достижений (для контекста)

| step | config | proxy ibm01 |
|---|---|---|
| baseline center init | default | 1.6868 |
| anchor_soft init | + cluster init | 1.2450 |
| multi-phase scheduler | + 3 phases | 1.1156 |
| ePlace FFT density 128 | + Poisson via FFT | 1.0366 (пробили ALNS) |
| per-K diversity | + random per-K hyperparams | 0.9998 (<1.0!) |
| rect_quad overlap | убрал gauss halo | 0.9707 |
| cluster_cohesion β=10 | dynamic centroid | 0.9637 |
| cohesion=5 + no anchor | anchor мешал | 0.9470 |
| **time_budget 600s** | 10 мин run | **0.9453** ← current best |

## Что НЕ работало (не повторяй)
- Nesterov с lr=0.3 (нужен lr=0.05)
- AdamW + grad_clip (как baseline)
- rect_cubic, rect_hinge overlap (хуже)
- MTK center init (single point) — наш centroid лучше
- cohesion=2,3,7,10,20,50 (5 best)
- Hybrid gradient → ALNS polish (basin trap)

## Финальный submission цель

Если pure-gradient на ibm01 даст < 0.95 стабильно — запустить на ВСЕХ 17 IBM benches и обновить submission. ALNS submission AVG17=1.4445, цель: AVG17 < 1.30 (топ-7 = Гран-при).

Работай. Удачи.
