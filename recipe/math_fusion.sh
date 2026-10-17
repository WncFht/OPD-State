#!/usr/bin/env bash
# Shared-prefix logit fusion; evaluation always uses the trained student alone.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export STEPS=${STEPS:-50}
export TRAIN_FILES=${TRAIN_FILES:-[${ROOT}/runs/rollout-sources/paired-schedule/ordered.parquet]}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-$(( (STEPS + 49) / 50 ))}
export FUSION_LAMBDA=${FUSION_LAMBDA:-0.5}
export FUSION_BACKEND=${FUSION_BACKEND:-cuda_graph}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-math-fusion-${FUSION_LAMBDA}-${STEPS}steps-eval16-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)}
export OPD_ROOT=${OPD_ROOT:-${ROOT}/runs/rollout-sources/${EXPERIMENT_NAME}}
export WANDB_JOB_TYPE=logit_fusion WANDB_RUN_GROUP=math-rollout-sources-eval16
TEACHER=${TEACHER_MODEL_PATH:-${MODEL_BASE:-${ROOT}/../models}/JustRL-DeepSeek-1.5B}
exec bash "$ROOT/recipe/math_baseline.sh" \
    data.shuffle=False \
    +trainer.rollout_source=logit_fusion \
    +actor_rollout_ref.rollout.logit_fusion._target_=verl.workers.config.LogitFusionConfig \
    +actor_rollout_ref.rollout.logit_fusion.enabled=True \
    +actor_rollout_ref.rollout.logit_fusion.teacher_model_path="$TEACHER" \
    +actor_rollout_ref.rollout.logit_fusion.teacher_weight="$FUSION_LAMBDA" \
    +actor_rollout_ref.rollout.logit_fusion.backend="$FUSION_BACKEND" \
    +actor_rollout_ref.rollout.logit_fusion.completion_check_interval=16 \
    "$@"
