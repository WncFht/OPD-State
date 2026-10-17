#!/usr/bin/env bash
# Fixed teacher/initial-student trajectories; student probabilities remain current.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:?Usage: math_cached.sh cached_teacher|cached_initial_student cache-directory [overrides]}
CACHE_DIRECTORY=${2:?A frozen cache directory is required}
shift 2
case "$SOURCE" in
    cached_teacher|cached_initial_student) ;;
    *) echo "Invalid fixed rollout source: $SOURCE" >&2; exit 2 ;;
esac
CACHE_DIRECTORY=$(realpath "$CACHE_DIRECTORY")
export STEPS=${STEPS:-50}
export TRAIN_FILES="[${ROOT}/runs/rollout-sources/paired-schedule/ordered.parquet]"
export TOTAL_EPOCHS=$(( (STEPS + 49) / 50 ))
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-math-${SOURCE}-${STEPS}steps-eval16-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)}
export OPD_ROOT=${OPD_ROOT:-${ROOT}/runs/rollout-sources/${EXPERIMENT_NAME}}
export WANDB_JOB_TYPE=$SOURCE WANDB_RUN_GROUP=math-rollout-sources-eval16
exec bash "$ROOT/recipe/math_baseline.sh" \
    data.shuffle=False \
    +trainer.fixed_rollout_cache="$CACHE_DIRECTORY" \
    +trainer.rollout_source="$SOURCE" \
    trainer.test_freq=50 \
    "$@"
