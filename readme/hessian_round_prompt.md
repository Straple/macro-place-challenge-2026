# Iteration prompt — для запуска нового раунда работы над Hessian saddle escape / CD polish

Скопируй этот промпт в новую сессию. Он самодостаточен: orchestrator (главный агент) сразу спавнит coder и reviewer как subagents, они работают в диалоге, потом запускается test.

---

## ПРОМПТ (скопировать целиком)

```
Ты — orchestrator команды из 3 ролей: coder + reviewer + ты сам (test runner).

ЦЕЛЬ РАУНДА: пробить ibm01 best ≤ 0.85 на pure-gradient placer (vmallela=0.7644 рекорд, наш текущий best=0.9040). Реализуем идеи из секции 5 readme/hessian.md.

ШАГ 0 — ОБЯЗАТЕЛЬНЫЙ:
Прочесть `readme/hessian.md` полностью. Это лог всех прошлых попыток. Без этого знания — повторишь чужие ошибки.

ШАГ 1 — Spawn CODER (Agent tool, subagent_type=general-purpose):
Передать ему промпт:
"""
Ты — coder в команде из 3 ролей. Твоя задача — выбрать один путь из секции 5 readme/hessian.md (A/B/C/D) ИЛИ предложить новую гипотезу, и реализовать.

Шаги:
1. Прочесть readme/hessian.md полностью (это критично — там 6 уже-провалившихся попыток).
2. Прочесть текущий код:
   - submissions/straple/gradient_batch.py (saddle escape ~ строки 1054-1180)
   - submissions/straple/cd_polish.py (CD polish)
   - scripts/gpu_run_one.py (integration)
3. Выбрать ОДИН узкий путь работы. Не пытаться сразу реализовать несколько path'ов.
4. Написать design note (≤200 слов): что делаешь, почему это должно работать, ожидаемый эффект (target: -3% от 0.9040).
5. Реализовать код. Гейтить через новый STRAPLE_BATCH_* env-var (не ломать trial9 baseline).
6. Передать reviewer'у (через SendMessage или output) с design note + ссылками на изменённые файлы.
7. Когда reviewer вернётся с критикой — ОТСТАИВАТЬ свою позицию если уверен, или фиксить если согласен. Не сдаваться сразу — спор ведёт к лучшему решению.

CONSTRAINTS:
- Memory: T4 16GB, baseline forward 10.2GB, остаётся ~5GB запаса
- Wall time test: ≤ 30 мин (forward 10 мин + legalize 3 мин + твой overhead ≤15 мин)
- Backwards compat: env-var off → identical to trial9
- Не делать destructive actions (rm, force push)

OUTPUT в финале (когда reviewer approve):
- design note
- список изменённых файлов
- предлагаемая experimental команда (env-vars + .remote_runs/ROUND_NAME)
"""

ШАГ 2 — Spawn REVIEWER (параллельно coder'у не делай — пусть coder сначала закончит draft, потом reviewer):
Передать ему промпт:
"""
Ты — reviewer в команде из 3 ролей. Твоя задача — критически пройтись по diff'у coder'а и выявить ошибки до того как они попадут на тест.

Шаги:
1. Прочесть readme/hessian.md полностью.
2. Прочесть design note coder'а + его изменения файлов.
3. Проверить:
   * Math correctness — особенно signs, indices в Lanczos/HVP/eigvec recovery. Прошлые баги: power-on-(-H) не находит smallest eig (нужен shift). Cross-check с numerical example в голове.
   * Memory: peak forward = 10.2GB. Coder's код может добавить 1-3GB на double-backward graph / Krylov basis / candidate batch. Если за 14GB — fail OOM на T4.
   * Runtime: forward step ~1.5с, HVP via FD = 2 forwards = ~3с, Lanczos k=20 = ~60с/trigger. Если coder делает >5 trigger'ов в 600s + еще оверхед — проверить time budget.
   * Backwards compat: убедиться что trial9 без новых env работает как раньше (не накладывается случайно код-путь).
   * Edge cases: NaN при beta=0 в Lanczos? Division by zero при normalize? OOM при K=384?
4. ОСПАРИВАТЬ — не одобрять «на доверии». Если что-то математически сомнительно — спросить «откуда ты взял что (σI−H)v сходится к v_min» и потребовать reasoning. Если coder защищается с веским аргументом — принять.
5. Approve ТОЛЬКО когда:
   * Все math claims обоснованы
   * Memory/runtime budget ОК
   * Backwards compat verified
   * Edge cases handled

OUTPUT:
- Список найденных проблем (каждая с severity: CRITICAL / MAJOR / MINOR)
- Конкретные требования для approve если есть unfinished
- ИЛИ APPROVED с финальным green light
"""

ШАГ 3 — DIALOG (coder ↔ reviewer):
Используй SendMessage чтобы coder и reviewer пинговали друг друга:
- coder отдаёт design note → reviewer
- reviewer возвращает критику → coder
- coder фиксит / спорит → reviewer
- Loop до APPROVED, max 3 round'а спора. Если 3 раунда не сошлись — escalate до user (это редкий случай).

ШАГ 4 — TEST (выполняет orchestrator сам):
1. Push approved code: `./run_remote.sh push`
2. Запустить test (заполнить experimental env-vars из coder's note):

```bash
ssh evyukhnevich@103.76.52.240 'cd macro-place && export PATH="$HOME/.local/bin:$PATH" && nohup bash -lc "
source .venv/bin/activate
export STRAPLE_BATCH_EPLACE=1 STRAPLE_BATCH_EPLACE_GRID=128 STRAPLE_BATCH_CONG_W=10 \
  STRAPLE_BATCH_COHESION_START=5 STRAPLE_BATCH_COHESION_END=0.001 \
  STRAPLE_BATCH_DIVERSITY=1 STRAPLE_BATCH_OVERLAP_FORM=rect_quad \
  STRAPLE_BATCH_OVERFLOW_LAMBDA=1 STRAPLE_BATCH_OVERFLOW_TARGET=0.13 \
  STRAPLE_BATCH_OVERFLOW_EXP=0.7 STRAPLE_BATCH_OVERFLOW_COEF_HI=1.5 \
  STRAPLE_BATCH_BLOCKAGE_W=50 \
  # ↓ experimental env-vars от coder ↓
  EXPERIMENTAL_ENV_HERE=value
uv run python scripts/gpu_run_one.py --bench ibm01 --K 384 --time-budget 600 --no-vis > .remote_runs/round_N.log 2>&1
cp results/gpu_stats_ibm01.json .remote_runs/stats_round_N.json
echo ROUND_N_DONE >> .remote_runs/round_N.log
" > /dev/null 2>&1 < /dev/null & disown'
```

3. Опросить статус (sleep 600 + check, до 30 минут max):

```bash
ssh evyukhnevich@103.76.52.240 'grep -c ROUND_N_DONE macro-place/.remote_runs/round_N.log; tail -22 macro-place/.remote_runs/round_N.log'
```

4. Pull stats:
```bash
scp evyukhnevich@103.76.52.240:macro-place/.remote_runs/stats_round_N.json /tmp/
```

5. Сравнить с trial9 baseline (min=0.9065, median=1.0616, mean=1.0485, std=0.0745).

ШАГ 5 — ОБНОВИТЬ hessian.md:
Добавить новую запись в секцию «Лог раундов» (секция 7.5):
- Hypothesis (от coder'а)
- Code change (короткое описание)
- Run config (env-vars)
- Result: min/p25/median/mean/std + saddle/CD counters
- Verdict: WIN (>0.005 better) / LOSE / NOISE / OOM
- Next: что предлагается дальше

Если open paths в секции 5 изменились — обновить.

ШАГ 6 — REPORT user:
- ≤200 слов summary: что попробовали, результат, verdict, что дальше
- Не дублировать содержимое hessian.md, ссылаться на него

CONSTRAINTS для всего раунда:
- ≤30 минут wall time на test (если больше — fail и сообщить)
- Не писать в старые файлы кроме (1) submissions/straple/* (2) scripts/gpu_run_one.py (3) hessian.md (только секция «Лог раундов»)
- Не запускать параллельные runs — sequential, один за другим
- Если OOM — coder должен фиксить chunked compute, не просто disable идею
- Если NaN — coder фиксит numerical safety, не disable

GOAL: либо WIN (новый best на ≥0.005 ниже предыдущего, что вне noise), либо conclusive LOSE с пониманием. NOISE-результат недостаточен — копать дальше.
```

---

## Пример вызова

После того как промпт скопирован в новую сессию, можно сразу выполнять:

```
> Запусти Round 1 по схеме из hessian_round_prompt.md
```

Orchestrator должен:
1. Прочитать hessian.md
2. Spawn coder (Agent tool, subagent_type=general-purpose, прокинуть промпт coder из ШАГ 1)
3. Дождаться coder draft
4. Spawn reviewer (Agent tool, прокинуть промпт reviewer из ШАГ 2 + coder's draft)
5. Использовать SendMessage между coder и reviewer для dialog
6. Когда APPROVED — push, run, wait, pull, update hessian.md, report

---

## Дополнительно: критерии WIN / LOSE / NOISE / OOM

| Verdict | Условие |
|---|---|
| **WIN** | new best на ibm01 K=384 600s ≥ 0.005 ниже trial9 baseline (т.е. ≤ 0.901) |
| **LOSE** | new best на ≥0.005 выше baseline (≥ 0.912) |
| **NOISE** | разница ±0.005 в пределах variance — нужны ≥3 повторных run для достоверности |
| **OOM** | CUDA out of memory — coder фиксит chunked compute |
| **CRASH** | NaN/Inf/exception — coder фиксит numerical safety |

Только WIN считается успехом раунда.
