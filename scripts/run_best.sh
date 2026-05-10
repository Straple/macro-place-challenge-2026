#!/usr/bin/env bash
# Reproduce best ibm01 result of the L-BFGS exploration session.
#
# Best of session: proxy=0.8785 (seed=43, single-shot full pipeline)
# Beats prior best Round 23 lucky 0.8856 by -0.007.
#
# Pipeline: trial9 baseline (Round 23 winning config) + L-BFGS finisher
# at step=1000 of gradient phase. Adam → L-BFGS quasi-Newton switch in
# late P3 settling smooths gradient noise that Adam alone can't escape.
#
# Wall time: ~25-28 min on T4 (K=384, 1200s gradient + legalize + CD +
# pair-swap rounds=8 + triple-cycle rounds=4).
#
# Usage on remote (89.169.181.58):
#   ./run_remote.sh push                          # sync code
#   ssh evyukhnevich@<host> 'cd macro-place &&
#     export PATH="$HOME/.local/bin:$PATH" &&
#     bash scripts/run_best.sh'
#
# Outputs (written to results/ on remote):
#   gpu_stats_ibm01.json      — final pre-CD distribution stats
#   gpu_seed_ibm01.pkl        — best placement (hard + soft positions)
#   gpu_pos_K_ibm01.npz       — full K-seed legalized positions
#   snapshots/ibm01_dump.npz  — 4-panel viz dump (if STRAPLE_BATCH_DUMP_SNAPSHOTS=1)

set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate

export STRAPLE_BATCH_EPLACE=1
export STRAPLE_BATCH_EPLACE_GRID=128
export STRAPLE_BATCH_CONG_W=10
export STRAPLE_BATCH_COHESION_START=5
export STRAPLE_BATCH_COHESION_END=0.001
export STRAPLE_BATCH_DIVERSITY=1
export STRAPLE_BATCH_OVERLAP_FORM=rect_quad
export STRAPLE_BATCH_OVERFLOW_LAMBDA=1
export STRAPLE_BATCH_OVERFLOW_TARGET=0.13
export STRAPLE_BATCH_OVERFLOW_EXP=0.7
export STRAPLE_BATCH_OVERFLOW_COEF_HI=1.5
export STRAPLE_BATCH_BLOCKAGE_W=50
export STRAPLE_BATCH_OVERLAP_W_MAX=50000
export STRAPLE_BATCH_OVERLAP_W_GROWTH=1.004

# L-BFGS finisher — winning lever.
export STRAPLE_BATCH_LBFGS_FROM_STEP=1000
export STRAPLE_BATCH_LBFGS_ALPHA=1.0
export STRAPLE_BATCH_LBFGS_CLIP=0.3

# Polish stack (Round 23 config).
export STRAPLE_BATCH_CD_POLISH=1
export STRAPLE_BATCH_CD_GPU_FILTER=1
export STRAPLE_BATCH_CD_GPU_APPROX=1
export STRAPLE_BATCH_CD_DIRS=8
export STRAPLE_BATCH_CD_ROUNDS=8
export STRAPLE_BATCH_PAIR_SWAP=1
export STRAPLE_BATCH_PAIR_SWAP_NEIGHBORS=12
export STRAPLE_BATCH_PAIR_SWAP_ROUNDS=8
export STRAPLE_BATCH_TRIPLE_CYCLE=1
export STRAPLE_BATCH_TRIPLE_CYCLE_NEIGHBORS=6
export STRAPLE_BATCH_TRIPLE_CYCLE_ROUNDS=4

# Wall time guard.
export STRAPLE_BATCH_WALL_TL=1700
export STRAPLE_BATCH_WALL_RESERVE=30

# Diagnostics.
export STRAPLE_BATCH_BREAKDOWN_LOG=1
export STRAPLE_BATCH_DUMP_SNAPSHOTS=1
export STRAPLE_BATCH_DUMP_DIR=results/snapshots
export STRAPLE_BATCH_DUMP_FRAMES_GRAD=15

# Best seed found in session.
export STRAPLE_BATCH_RUN_SEED_BASE=43

BENCH="${1:-ibm01}"

uv run python scripts/gpu_run_one.py \
    --bench "$BENCH" \
    --K 384 \
    --time-budget 1200 \
    --no-vis
