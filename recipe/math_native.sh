#!/usr/bin/env bash
# Flat fresh-rollout math policies. The learner is always the original student.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:-student_fresh}
if (($#)); then shift; fi
case "$SOURCE" in
    student_fresh) POLICY='{kind:single,model:current_student}' ;;
    initial_student_fresh) POLICY='{kind:single,model:initial_student}' ;;
    teacher_fresh) POLICY='{kind:single,model:teacher}' ;;
    current_request_split|initial_request_split|teacher_request_split)
        case "$SOURCE" in
            current_request_split) MODEL=current_student ;;
            initial_request_split) MODEL=initial_student ;;
            teacher_request_split) MODEL=teacher ;;
        esac
        POLICY="{kind:sequence_switch,first:${MODEL},second:${MODEL},boundary:{kind:response_tokens,value:${SWITCH_TOKENS:-2048}},keep_request_boundary:true}"
        ;;
    student_teacher_fusion|initial_teacher_fusion)
        BASE=current_student
        [[ "$SOURCE" != initial_teacher_fusion ]] || BASE=initial_student
        POLICY="{kind:logit_fusion,base:${BASE},teacher_weight:${TEACHER_WEIGHT:-0.5}}"
        ;;
    teacher_current|current_teacher|teacher_initial|initial_teacher)
        case "$SOURCE" in
            teacher_current) FIRST=teacher; SECOND=current_student ;;
            current_teacher) FIRST=current_student; SECOND=teacher ;;
            teacher_initial) FIRST=teacher; SECOND=initial_student ;;
            initial_teacher) FIRST=initial_student; SECOND=teacher ;;
        esac
        POLICY="{kind:sequence_switch,first:${FIRST},second:${SECOND},boundary:{kind:response_tokens,value:${SWITCH_TOKENS:-2048}}}"
        ;;
    batch_current|batch_initial)
        BASE=current_student
        [[ "$SOURCE" != batch_initial ]] || BASE=initial_student
        TEACHER_FRACTION=${TEACHER_FRACTION:-0.5}
        BASE_FRACTION=$(python3 -c 'import sys; print(1-float(sys.argv[1]))' "$TEACHER_FRACTION")
        POLICY="{kind:batch_mix,assignment:exact_count,routing_unit:trajectory,components:[{policy:{kind:single,model:teacher},weight:${TEACHER_FRACTION}},{policy:{kind:single,model:${BASE}},weight:${BASE_FRACTION}}]}"
        ;;
    *) echo "Unsupported native policy: $SOURCE" >&2; exit 2 ;;
esac
export STEPS=${STEPS:-50}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-math-native-${SOURCE}-${STEPS}steps-$(TZ=Asia/Shanghai date +%Y%m%d-%H%M%S)}
export OPD_ROOT=${OPD_ROOT:-${ROOT}/runs/native-rollout/${EXPERIMENT_NAME}}
export WANDB_JOB_TYPE=native_rollout WANDB_RUN_GROUP=math-native-rollout-sources
native_cli_data=false
for native_override in "$@"; do
    case "$native_override" in
        data.train_files=*|+data.train_files=*|++data.train_files=*|trainer.native_data_manifest=*|+trainer.native_data_manifest=*|++trainer.native_data_manifest=*)
            native_cli_data=true ;;
    esac
done
if [[ -z ${NATIVE_DATA_MANIFEST:-} && -z ${TRAIN_FILES:-} && $native_cli_data == false ]]; then
    NATIVE_DATA_MANIFEST="$ROOT/runs/native-data/math-full/manifest.json"
fi
if [[ -n ${NATIVE_DATA_MANIFEST:-} ]]; then
    if [[ ! -f "$NATIVE_DATA_MANIFEST" ]]; then
        echo "Native math requires a prepared full-prompt manifest: $NATIVE_DATA_MANIFEST" >&2
        echo "Prepare it with data/prep/build_fresh_math_data.py; see data/README.md#fresh-math-prompt-pool." >&2
        exit 2
    fi
    NATIVE_DATA_MANIFEST=$(realpath "$NATIVE_DATA_MANIFEST")
    export NATIVE_DATA_MANIFEST
    export TRAIN_FILES=${TRAIN_FILES:-[$(dirname "$NATIVE_DATA_MANIFEST")/ordered.parquet]}
    set -- "++trainer.native_data_manifest=${NATIVE_DATA_MANIFEST}" "$@"
fi
exec bash "$ROOT/recipe/math_baseline.sh" \
    "actor_rollout_ref.rollout.policy=${POLICY}" \
    actor_rollout_ref.rollout.generation_backend=vllm_native \
    actor_rollout_ref.rollout.collection.mode=fresh \
    ++actor_rollout_ref.rollout.logit_fusion.enabled=False \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    algorithm.rollout_correction=null data.shuffle=True \
    "$@"
