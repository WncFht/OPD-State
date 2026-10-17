#!/usr/bin/env bash
# Four-GPU sampled-token math baseline; three-benchmark avg@16, final-only by default.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(dirname "$ROOT")
cd "$ROOT"
TRAIN_PYTHON=${OPD_TRAIN_PYTHON:-${WORKSPACE}/envs/opd-state/bin/python}
if [[ ! -x "$TRAIN_PYTHON" ]]; then
    echo "Training Python does not exist: $TRAIN_PYTHON" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CONDA_ENV_BIN
CONDA_ENV_BIN=$(dirname "$TRAIN_PYTHON")
export MODEL_BASE=${MODEL_BASE:-${WORKSPACE}/models}
export DATA_BASE=${DATA_BASE:-${WORKSPACE}/MD-OPD/data}
unset WANDB_API_KEY SWANLAB_API_KEY
export WANDB_BASE_URL=https://api.wandb.ai
export PROJECT_NAME=OPD-State WANDB_PROJECT=OPD-State
export TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','wandb']"}
export WANDB_MODE=${WANDB_MODE:-online}
export NETRC=${NETRC:-${WORKSPACE}/.secrets/netrc}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-${WORKSPACE}/config/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${WORKSPACE}/cache/wandb}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-${WORKSPACE}/cache/wandb/staging}
export WANDB_DIR=${WANDB_DIR:-${ROOT}/runs/wandb}
export HF_HOME=${HF_HOME:-${WORKSPACE}/cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export NLTK_DATA=${NLTK_DATA:-${WORKSPACE}/cache/nltk}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.6}
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export RAY_ADDRESS=local RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export VERL_USE_EXTERNAL_PLUGINS=none
export VLLM_USE_V1=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
SOURCE_COMMIT=$(git rev-parse HEAD)
STEPS=${STEPS:-50}
export N_GPUS_PER_NODE=4 NNODES=1 DOMAIN=math MODE=full
export SAVE_FREQ=-1 TEST_FREQ=$STEPS VAL_BEFORE_TRAIN=False
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-4096}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-31744}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export TEACHER_MICRO_BSZ=${TEACHER_MICRO_BSZ:-1}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.55}
export VAL_N=${VAL_N:-16} VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.7} VAL_TOP_P=${VAL_TOP_P:-0.9}
export VAL_FILES=${VAL_FILES:-[${DATA_BASE}/math/MATH-500/test.parquet,${DATA_BASE}/math/AMC23/test.parquet,${DATA_BASE}/math/AIME25/test.parquet]}
export LOG_PROB_TOP_K=0 METRIC_TOP_K=0 N_RESPONSES=1
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-math-on-policy-50steps-$(date +%Y%m%d-%H%M%S)}
export OPD_ROOT=${OPD_ROOT:-${ROOT}/runs/math-baseline/${EXPERIMENT_NAME}}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/opd-state-ray-${UID}}
mkdir -p "$OPD_ROOT" "$WANDB_DIR" "$RAY_TMPDIR"
exec bash -o pipefail recipe/opd.sh \
    trainer.total_training_steps="$STEPS" \
    +trainer.source_commit="$SOURCE_COMMIT" \
    trainer.resume_mode=disable \
    trainer.log_val_generations=0 \
    trainer.validation_data_dir="$OPD_ROOT/validation" \
    data.seed=42 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    data.val_batch_size=32 \
    +trainer.math_eval_samples="$VAL_N" \
    +trainer.math_eval_require_full=True \
    +ray_kwargs.ray_init.num_cpus=16 \
    +ray_kwargs.ray_init.object_store_memory=4294967296 \
    "$@"
