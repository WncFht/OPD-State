#!/usr/bin/env bash
# Online student trajectories on the exact prompt order used by cached sources.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export STEPS=${STEPS:-50}
export TRAIN_FILES=${TRAIN_FILES:-[${ROOT}/runs/rollout-sources/paired-schedule/ordered.parquet]}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-$(( (STEPS + 49) / 50 ))}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-math-student-online-${STEPS}steps-eval16-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)}
export OPD_ROOT=${OPD_ROOT:-${ROOT}/runs/rollout-sources/${EXPERIMENT_NAME}}
export WANDB_JOB_TYPE=student_online WANDB_RUN_GROUP=math-rollout-sources-eval16
exec bash "$ROOT/recipe/math_baseline.sh" \
    data.shuffle=False \
    +trainer.rollout_source=student_online \
    "$@"
