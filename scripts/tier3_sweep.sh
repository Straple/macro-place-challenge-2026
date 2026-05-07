#!/bin/bash
# Tier 3 hyperparam sweep over OVERFLOW + BLOCKAGE + COHESION on ibm01.
# Each trial: K=384, budget=600s, captures stats JSON.
# Sequential execution. Total ~2h.
set -euo pipefail

cd ~/macro-place
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
mkdir -p .remote_runs/sweep

# Trial format: name|target|exp|coef_hi|blk_w|coh_start
trials=(
    "trial1|0.10|1.0|1.5|50|5"   # current best baseline
    "trial2|0.07|1.0|1.5|50|5"   # tighter target
    "trial3|0.13|1.0|1.5|50|5"   # looser target
    "trial4|0.10|0.7|1.5|50|5"   # gentler exp
    "trial5|0.10|1.3|1.5|50|5"   # sharper exp
    "trial6|0.10|1.0|1.5|30|5"   # less blockage
    "trial7|0.10|1.0|1.5|100|5"  # more blockage
    "trial8|0.10|1.0|2.0|50|5"   # higher coef_hi (uncapped DRP)
)

start_all=$(date +%s)
for t in "${trials[@]}"; do
    IFS='|' read -r name target expv coefhi blkw cohs <<< "$t"
    echo "=== $name target=$target exp=$expv coef_hi=$coefhi blk=$blkw coh=$cohs ==="
    t0=$(date +%s)
    STRAPLE_BATCH_EPLACE=1 STRAPLE_BATCH_EPLACE_GRID=128 \
    STRAPLE_BATCH_CONG_W=10 \
    STRAPLE_BATCH_COHESION_START=$cohs STRAPLE_BATCH_COHESION_END=0.001 \
    STRAPLE_BATCH_DIVERSITY=1 STRAPLE_BATCH_OVERLAP_FORM=rect_quad \
    STRAPLE_BATCH_OVERFLOW_LAMBDA=1 \
    STRAPLE_BATCH_OVERFLOW_TARGET=$target \
    STRAPLE_BATCH_OVERFLOW_EXP=$expv \
    STRAPLE_BATCH_OVERFLOW_COEF_HI=$coefhi \
    STRAPLE_BATCH_BLOCKAGE_W=$blkw \
        uv run python scripts/gpu_run_one.py \
            --bench ibm01 --K 384 --time-budget 600 --no-vis \
            > .remote_runs/sweep/${name}.log 2>&1
    cp results/gpu_stats_ibm01.json .remote_runs/sweep/stats_${name}.json
    t1=$(date +%s)
    echo "[$name] done in $((t1-t0))s"
done

end_all=$(date +%s)
echo "ALL TRIALS DONE in $((end_all-start_all))s"
