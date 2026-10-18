#!/usr/bin/env bash
# Prepare either direction of the same-pair alignment experiment; no implicit launch pairing.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(dirname "$ROOT")
export MODEL_BASE=${MODEL_BASE:-${WORKSPACE}/models}
case "${DIRECTION:-forward}" in
    forward)
        export ACTOR_MODEL_PATH=${MODEL_BASE}/DeepSeek-R1-Distill-Qwen-1.5B
        export TEACHER_MODEL_PATH=${MODEL_BASE}/JustRL-DeepSeek-1.5B
        ;;
    reverse)
        export ACTOR_MODEL_PATH=${MODEL_BASE}/JustRL-DeepSeek-1.5B
        export TEACHER_MODEL_PATH=${MODEL_BASE}/DeepSeek-R1-Distill-Qwen-1.5B
        ;;
    *) echo 'DIRECTION must be forward or reverse' >&2; exit 2 ;;
esac
TRAIN_PYTHON=${OPD_TRAIN_PYTHON:-${WORKSPACE}/envs/opd-state/bin/python}
VOCAB_SIZE=$("$TRAIN_PYTHON" -c 'import json,sys; t=json.load(open(sys.argv[1])); q=json.load(open(sys.argv[2])); assert t==q, "Student/teacher tokenizer files must match"; v=dict(t["model"]["vocab"]); v.update({x["content"]:x["id"] for x in t["added_tokens"]}); assert set(v.values())==set(range(len(v))); print(len(v))' "$ACTOR_MODEL_PATH/tokenizer.json" "$TEACHER_MODEL_PATH/tokenizer.json")
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
export METRIC_TOP_K=0 POLICY_LOSS_MODE=opd_topk_tail
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-7168}
export STEPS=${STEPS:-300}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-math-alignment-${DIRECTION:-forward}-k${LOG_PROB_TOP_K}-$(date +%Y%m%d-%H%M%S)}
export OPD_ROOT=${OPD_ROOT:-${ROOT}/runs/alignment/${EXPERIMENT_NAME}}
export VAL_TEMPERATURE=1.0 VAL_TOP_P=1.0
export OPD_FREEZE_SOURCE=${OPD_FREEZE_SOURCE:-1}
# CLI overrides are last so a short validation canary can change budgets.
exec bash "$ROOT/recipe/math_baseline.sh" \
    distillation.fsdp_teacher.unbiased_topk_tail=True \
    distillation.fsdp_teacher.effective_vocab_size="$VOCAB_SIZE" \
    distillation.fsdp_teacher.log_prob_top_k="$LOG_PROB_TOP_K" \
    actor_rollout_ref.actor.policy_loss.loss_mode=opd_topk_tail \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    algorithm.rollout_correction=null \
    trainer.total_epochs="$STEPS" \
    trainer.test_freq=50 trainer.val_before_train=True \
    trainer.save_freq=50 trainer.default_local_dir="$OPD_ROOT/checkpoints" \
    actor_rollout_ref.actor.checkpoint.save_contents="[hf_model]" \
    "$@"
