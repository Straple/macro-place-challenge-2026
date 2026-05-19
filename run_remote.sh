#!/usr/bin/env bash
#
# Macro Placement Challenge: rsync + run на удалённом GPU-сервере.
# Сервер: Intel Ice Lake + NVIDIA Tesla T4 (16 vCPU, 64GB RAM, 128GB disk).
#
# Quick start:
#   ./run_remote.sh push                # синк кода на сервер
#   ./run_remote.sh bootstrap           # установка uv, deps, build C++
#   ./run_remote.sh ssh                 # интерактивный shell
#   ./run_remote.sh eval -b ibm01       # uv run evaluate ... -b ibm01
#   ./run_remote.sh eval --all          # все 17 IBM benches
#   ./run_remote.sh gpu --bench ibm01   # GPU multi-start через scripts/gpu_search.py
#   ./run_remote.sh sweep -- --bench ibm01 --trials 64
#   ./run_remote.sh pull                # тянет vis/, results/ обратно
#
# Подкоманды:
#   push       — rsync репо на сервер (без .git, vis/, __pycache__, и т.д.)
#   bootstrap  — установка uv, sync deps, git submodule update, build C++
#   ssh        — интерактивный shell в REMOTE_DIR
#   eval ARGS  — `uv run evaluate submissions/straple/placer.py ARGS`
#                с пробросом всех STRAPLE_*/CUDA_* env vars из локального окружения
#   gpu ARGS   — запускает scripts/gpu_search.py на сервере (multi-start GPU)
#   sweep ARGS — запускает scripts/gpu_sweep.py — long-running hyperparam sweep
#   logs       — tail -f последнего remote run log
#   pull       — rsync vis/, results/, .remote_runs/ обратно
#   clean      — rm -rf .remote_runs/ на сервере
#
# Env vars (с дефолтами):
#   REMOTE_USER=evyukhnevich
#   REMOTE_HOST=89.169.161.15
#   REMOTE_DIR=macro-place
#   SSH_OPTS=""
#   STRAPLE_*    — все пробрасываются на сервер для подкоманд eval/gpu/sweep
#   CUDA_VISIBLE_DEVICES=0  (default)

set -euo pipefail

REMOTE_USER="${REMOTE_USER:-evyukhnevich}"
REMOTE_HOST="${REMOTE_HOST:-89.169.161.15}"
REMOTE_DIR="${REMOTE_DIR:-macro-place}"
SSH_KEEPALIVE_OPTS="-o ServerAliveInterval=10 -o ServerAliveCountMax=6 -o TCPKeepAlive=yes -o StrictHostKeyChecking=accept-new"
SSH_OPTS="${SSH_KEEPALIVE_OPTS} ${SSH_OPTS:-}"
REMOTE_POLL_INTERVAL="${REMOTE_POLL_INTERVAL:-15}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"

RSYNC_EXCLUDES=(
  ".git/"
  ".claude/"
  ".vscode/"
  ".DS_Store"
  ".remote_runs/"
  "vis/"
  "build/"
  "*.egg-info/"
  ".venv/"
  "__pycache__/"
  "*.pyc"
  "*.so"
  "*.so.dSYM/"
  "external/MacroPlacement/.git/"
  "external/DREAMPlace/"
  "results/"
  "logs/"
)

log() {
  printf '\033[1;34m[remote]\033[0m %s\n' "$*" >&2
}

err() {
  printf '\033[1;31m[remote]\033[0m %s\n' "$*" >&2
}

rsync_excludes_args() {
  local e
  for e in "${RSYNC_EXCLUDES[@]}"; do
    printf -- '--exclude=%s\n' "$e"
  done
}

# Список STRAPLE_*/CUDA_* env vars из локального окружения, для проброса на сервер.
collect_env_vars() {
  local out=""
  local k
  for k in $(env | grep -E '^(STRAPLE_|CUDA_)' | cut -d= -f1); do
    local v="${!k}"
    out+=" ${k}=$(printf '%q' "${v}")"
  done
  printf '%s' "${out}"
}

push_code() {
  log "push ${REPO_ROOT} -> ${SSH_TARGET}:${REMOTE_DIR}"
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${SSH_TARGET}" "mkdir -p ${REMOTE_DIR}"
  # shellcheck disable=SC2046
  rsync -az --delete --info=stats1,progress2 \
    $(rsync_excludes_args) \
    -e "ssh ${SSH_OPTS}" \
    "${REPO_ROOT}/" "${SSH_TARGET}:${REMOTE_DIR}/"
}

pull_results() {
  local sub
  for sub in vis results .remote_runs; do
    # shellcheck disable=SC2086
    if ssh ${SSH_OPTS} "${SSH_TARGET}" "test -d ${REMOTE_DIR}/${sub}" 2>/dev/null; then
      log "pull ${SSH_TARGET}:${REMOTE_DIR}/${sub}/ -> ${REPO_ROOT}/${sub}/"
      mkdir -p "${REPO_ROOT}/${sub}"
      rsync -az --info=stats1,progress2 \
        -e "ssh ${SSH_OPTS}" \
        "${SSH_TARGET}:${REMOTE_DIR}/${sub}/" "${REPO_ROOT}/${sub}/"
    fi
  done
}

bootstrap() {
  log "bootstrap на сервере: apt deps, uv install, project sync, C++ build"
  push_code
  # shellcheck disable=SC2086
  ssh -t ${SSH_OPTS} "${SSH_TARGET}" "bash -lc '
    set -euo pipefail
    cd ${REMOTE_DIR}
    echo \"---- system python: \$(python3 --version)\"
    echo \"---- gpu check: nvidia-smi -L\"
    nvidia-smi -L 2>/dev/null || echo NO_GPU

    if ! command -v uv >/dev/null 2>&1; then
      echo \"---- installing uv\"
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH=\"\$HOME/.local/bin:\$PATH\"
    fi
    export PATH=\"\$HOME/.local/bin:\$PATH\"

    echo \"---- apt deps: build-essential, cmake, pybind11-dev, ffmpeg\"
    if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
    \$SUDO apt-get update -y
    \$SUDO apt-get install -y build-essential cmake python3-pybind11 ffmpeg git rsync

    echo \"---- uv sync (CPU + CUDA torch)\"
    uv sync || true

    echo \"---- ensuring CUDA torch installed\"
    uv pip install --upgrade \"torch>=2.0\" --index-url https://download.pytorch.org/whl/cu121 || \\
      uv pip install --upgrade \"torch>=2.0\"
    uv pip install networkx scipy

    echo \"---- git submodule update\"
    git submodule update --init external/MacroPlacement || true

    echo \"---- C++ build (placer_core + proxy_cost)\"
    bash submissions/straple/cpp/build.sh || true

    echo \"---- smoke check torch CUDA\"
    uv run python -c \"import torch; print(\\\"cuda available:\\\", torch.cuda.is_available()); print(\\\"device:\\\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \\\"cpu\\\")\"

    echo \"---- bootstrap done\"
  '"
}

remote_shell() {
  # shellcheck disable=SC2086
  exec ssh -t ${SSH_OPTS} "${SSH_TARGET}" "cd ${REMOTE_DIR} && exec bash -l"
}

# Запустить команду на сервере отвязно от ssh-сессии, с polling логов.
# Args: 1=run_id 2=remote_command (single string, will be exec'd in bash -c)
run_remote_detached() {
  local run_id="$1"
  local remote_cmd="$2"
  local log_rel=".remote_runs/${run_id}.log"
  local exit_rel=".remote_runs/${run_id}.exit"
  local pid_rel=".remote_runs/${run_id}.pid"

  log "remote run ${run_id}"
  log "  log:  ${REMOTE_DIR}/${log_rel}"

  local wrapper
  wrapper="cd ${REMOTE_DIR}; mkdir -p .remote_runs; export PATH=\$HOME/.local/bin:\$PATH; ${remote_cmd}; echo \$? > ${exit_rel}"
  local wrapper_q
  wrapper_q="$(printf '%q' "${wrapper}")"

  local launch
  launch=$(cat <<EOSH
mkdir -p ${REMOTE_DIR}/.remote_runs
rm -f ${REMOTE_DIR}/${pid_rel}
setsid bash -c "echo \$\$ > ${REMOTE_DIR}/${pid_rel}; bash -c ${wrapper_q} > ${REMOTE_DIR}/${log_rel} 2>&1" < /dev/null > /dev/null 2>&1 &
disown || true
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  if [ -s ${REMOTE_DIR}/${pid_rel} ]; then
    cat ${REMOTE_DIR}/${pid_rel}
    exit 0
  fi
  sleep 0.1
done
echo "ERR: pid file not written" >&2
exit 1
EOSH
)
  local pid
  # shellcheck disable=SC2086
  pid=$(ssh ${SSH_OPTS} "${SSH_TARGET}" "${launch}") || {
    err "не удалось запустить процесс на сервере"
    return 1
  }
  pid="${pid//[$'\t\r\n ']}"
  if [[ -z "${pid}" || ! "${pid}" =~ ^[0-9]+$ ]]; then
    err "невалидный PID: '${pid}'"
    return 1
  fi
  log "remote pid=${pid} (poll каждые ${REMOTE_POLL_INTERVAL}s)"

  trap "interrupt_remote ${pid}" INT TERM

  local offset=1
  local fail_count=0
  local fail_max=20
  local first=1
  while :; do
    if [[ "${first}" == "1" ]]; then
      first=0
    else
      sleep "${REMOTE_POLL_INTERVAL}"
    fi

    local status_block
    # shellcheck disable=SC2086
    if ! status_block=$(ssh ${SSH_OPTS} "${SSH_TARGET}" "
if kill -0 ${pid} 2>/dev/null; then alive=1; else alive=0; fi
log_size=\$(stat -c %s ${REMOTE_DIR}/${log_rel} 2>/dev/null || echo 0)
printf '__STATUS__ alive=%s log_size=%s\\n' \"\$alive\" \"\$log_size\"
tail -c +${offset} ${REMOTE_DIR}/${log_rel} 2>/dev/null || true
printf '__POLL_END__'
" 2>/dev/null); then
      fail_count=$((fail_count + 1))
      log "poll ssh failed (${fail_count}/${fail_max})"
      if (( fail_count >= fail_max )); then
        err "слишком много неудачных poll'ов; remote pid=${pid} продолжает"
        return 1
      fi
      continue
    fi
    fail_count=0
    status_block="${status_block%__POLL_END__}"

    local header rest
    if [[ "${status_block}" == *$'\n'* ]]; then
      header="${status_block%%$'\n'*}"
      rest="${status_block#*$'\n'}"
    else
      header="${status_block}"
      rest=""
    fi
    header="${header%$'\r'}"
    if [[ "${header}" != __STATUS__* ]]; then
      log "unexpected: ${header}"
      continue
    fi
    local alive log_size
    alive="${header#*alive=}"; alive="${alive%% *}"
    log_size="${header##*log_size=}"
    log_size="${log_size%$'\r'}"
    if [[ -n "${rest}" ]]; then
      printf '%s' "${rest}"
    fi
    if [[ "${log_size}" =~ ^[0-9]+$ ]]; then
      offset=$((log_size + 1))
    fi
    if [[ "${alive}" == "0" ]]; then
      local tail_final
      # shellcheck disable=SC2086
      tail_final=$(ssh ${SSH_OPTS} "${SSH_TARGET}" \
        "tail -c +${offset} ${REMOTE_DIR}/${log_rel} 2>/dev/null || true; printf '__POLL_END__'" \
        2>/dev/null) || tail_final=""
      tail_final="${tail_final%__POLL_END__}"
      if [[ -n "${tail_final}" ]]; then
        printf '%s' "${tail_final}"
      fi
      break
    fi
  done

  trap - INT TERM

  local rc
  # shellcheck disable=SC2086
  rc=$(ssh ${SSH_OPTS} "${SSH_TARGET}" "cat ${REMOTE_DIR}/${exit_rel} 2>/dev/null" 2>/dev/null || echo "1")
  rc="${rc//[$'\t\r\n ']}"
  if [[ ! "${rc}" =~ ^[0-9]+$ ]]; then rc=1; fi
  log "remote rc=${rc}"
  return "${rc}"
}

interrupt_remote() {
  local pid="$1"
  log "interrupted — TERM на process group ${pid}"
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${SSH_TARGET}" "
kill -TERM -- -${pid} 2>/dev/null || true
for _ in 1 2 3 4 5 6; do
  kill -0 -- -${pid} 2>/dev/null || exit 0
  sleep 0.5
done
kill -KILL -- -${pid} 2>/dev/null || true
" >/dev/null 2>&1 || true
  exit 130
}

run_eval() {
  push_code
  local env_vars
  env_vars="$(collect_env_vars)"
  local args=""
  for a in "$@"; do
    args+=" $(printf '%q' "$a")"
  done
  local run_id
  run_id="eval_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  local cmd="${env_vars} \$HOME/.local/bin/uv run --no-progress evaluate submissions/straple/placer.py${args}"
  run_remote_detached "${run_id}" "${cmd}"
  local rc=$?
  pull_results || log "pull failed (продолжаю)"
  return "${rc}"
}

run_gpu() {
  push_code
  local env_vars
  env_vars="$(collect_env_vars)"
  local args=""
  for a in "$@"; do
    args+=" $(printf '%q' "$a")"
  done
  local run_id
  run_id="gpu_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  local cmd="${env_vars} \$HOME/.local/bin/uv run --no-progress python scripts/gpu_search.py${args}"
  run_remote_detached "${run_id}" "${cmd}"
  local rc=$?
  pull_results || log "pull failed (продолжаю)"
  return "${rc}"
}

run_sweep() {
  push_code
  local env_vars
  env_vars="$(collect_env_vars)"
  local run_id
  run_id="sweep_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  local cmd="${env_vars} \$HOME/.local/bin/uv run --no-progress python scripts/gpu_sweep.py"
  for a in "$@"; do
    cmd+=" $(printf '%q' "$a")"
  done
  run_remote_detached "${run_id}" "${cmd}"
  local rc=$?
  pull_results || log "pull failed (продолжаю)"
  return "${rc}"
}

remote_logs() {
  local last
  # shellcheck disable=SC2086
  last=$(ssh ${SSH_OPTS} "${SSH_TARGET}" "ls -1t ${REMOTE_DIR}/.remote_runs/*.log 2>/dev/null | head -1" 2>/dev/null)
  if [[ -z "${last}" ]]; then
    err "нет логов на сервере"
    return 1
  fi
  log "tail -f ${last}"
  # shellcheck disable=SC2086
  exec ssh -t ${SSH_OPTS} "${SSH_TARGET}" "tail -f ${last}"
}

clean_remote() {
  # shellcheck disable=SC2086
  ssh ${SSH_OPTS} "${SSH_TARGET}" "rm -rf ${REMOTE_DIR}/.remote_runs"
  log "cleaned ${REMOTE_DIR}/.remote_runs/"
}

usage() {
  sed -n '/^# Macro Placement/,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  push)      push_code ;;
  pull)      pull_results ;;
  bootstrap) bootstrap ;;
  ssh)       remote_shell ;;
  eval)      run_eval "$@" ;;
  gpu)       run_gpu "$@" ;;
  sweep)     run_sweep "$@" ;;
  logs)      remote_logs ;;
  clean)     clean_remote ;;
  help|-h|--help) usage ;;
  *)
    err "unknown subcommand: ${cmd}"
    usage
    exit 2
    ;;
esac
