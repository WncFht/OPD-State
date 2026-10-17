#!/usr/bin/env bash
# Freeze the generator and produce one immutable teacher or initial-student cache.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${1:?Usage: prepare_math_cache.sh cached_teacher|cached_initial_student output-directory [overrides]}
OUTPUT=${2:?An output directory is required}
shift 2
case "$SOURCE" in
    cached_teacher|cached_initial_student) ;;
    *) echo "Invalid fixed rollout source: $SOURCE" >&2; exit 2 ;;
esac
cd "$ROOT"
SOURCE_COMMIT=$(git rev-parse --verify "${OPD_SOURCE_COMMIT:-HEAD}^{commit}")
SNAPSHOT="$ROOT/runs/rollout-sources/preparation-source-$SOURCE_COMMIT"
mkdir -p "$SNAPSHOT"
if [[ ! -f "$SNAPSHOT/COMMIT" ]]; then
    git archive "$SOURCE_COMMIT" verl data/prep/build_rollout_cache.py | tar -x -C "$SNAPSHOT"
    printf '%s\n' "$SOURCE_COMMIT" > "$SNAPSHOT/COMMIT"
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HOME=${HF_HOME:-${ROOT}/../cache/huggingface}
export PYTHONPATH="$SNAPSHOT/verl${PYTHONPATH:+:$PYTHONPATH}"
exec "${OPD_TRAIN_PYTHON:-${ROOT}/../envs/opd-state/bin/python}" \
    "$SNAPSHOT/data/prep/build_rollout_cache.py" generate \
    --source "$SOURCE" \
    --schedule "$ROOT/runs/rollout-sources/paired-schedule" \
    --output "$OUTPUT" \
    --student "${MODEL_BASE:-${ROOT}/../models}/DeepSeek-R1-Distill-Qwen-1.5B" \
    --teacher "${MODEL_BASE:-${ROOT}/../models}/JustRL-DeepSeek-1.5B" \
    --generator-commit "$SOURCE_COMMIT" \
    --gpus 4 --batch-size 64 --response-cap 4096 "$@"
