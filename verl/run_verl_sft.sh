#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/data1/models/Qwen3.6-35B-A3B}
DATA_ROOT=${DATA_ROOT:-/data1/Qwen3.6_eval}
WORK_DIR=${WORK_DIR:-/data1/f00538480/verl_data/sft_smoke}
VERL_HOME=${VERL_HOME:-/verl}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MA_NUM_HOSTS=${MA_NUM_HOSTS:-1}
VC_TASK_INDEX=${VC_TASK_INDEX:-0}
VC_WORKER_HOSTS=${VC_WORKER_HOSTS:-127.0.0.1}
NPROC_PER_NODE=16
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NODE_RANK=${VC_TASK_INDEX}
WORLD_SIZE=$(( NPROC_PER_NODE * MA_NUM_HOSTS ))
MASTER_PORT=${MASTER_PORT:-29500}
RDZV_ID=${RDZV_ID:-qwen36_packed_smoke}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-0}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-16384}
TRAIN_STEPS=${TRAIN_STEPS:-10}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

if [ "${TRAIN_BATCH_SIZE}" = "0" ]; then
    TRAIN_BATCH_SIZE=${WORLD_SIZE}
fi

export PYTHONPATH="${VERL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_NPU_SOCKET_PORT_RANGE=26000-26200
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export VERL_SFT_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1
export VERL_DISABLE_FLASH_ATTN_CE=1

SPLIT_DIR="${WORK_DIR}/split_data"
TRAIN_FILES_LIST="${SPLIT_DIR}/train_files.hydra.txt"
VAL_FILES_LIST="${SPLIT_DIR}/val_files.hydra.txt"

if [ ! -f "${TRAIN_FILES_LIST}" ]; then
    echo "[ERROR] Split data not found. Run right-padding smoke first."
    exit 1
fi

TRAIN_FILES=$(cat "${TRAIN_FILES_LIST}")
VAL_FILES=$(cat "${VAL_FILES_LIST}")

mkdir -p "${WORK_DIR}/output"
CKPT_DIR="${WORK_DIR}/output/packed_checkpoints"
LOG_FILE="${WORK_DIR}/output/packed_train_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo " Packed path smoke (use_remove_padding=True)"
echo "   World: ${WORLD_SIZE}, GBS: ${TRAIN_BATCH_SIZE}"
echo "   Max token len: ${MAX_TOKEN_LEN}"
echo "   Steps: ${TRAIN_STEPS}"
echo "============================================================"

cd "${VERL_HOME}"

torchrun \
    --nnodes="${MA_NUM_HOSTS}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_id="${RDZV_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="" \
    data.train_max_samples=64 \
    data.messages_key=messages \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN}" \
    data.use_dynamic_bsz=True \
    data.max_length="${MAX_TOKEN_LEN}" \
    data.pad_mode=no_padding \
    data.ignore_input_ids_mismatch=True \
    data.truncation=right \
    +data.image_patch_size=16 \
    data.num_workers=0 \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=True \
    model.use_liger=False \
    +model.override_config.attn_implementation=sdpa \
    model.use_fused_kernels=False \
    model.enable_gradient_checkpointing=True \
    engine=fsdp \
    engine.strategy=fsdp2 \
    engine.model_dtype=bfloat16 \
    engine.use_torch_compile=False \
    engine.ulysses_sequence_parallel_size=1 \
    engine.fsdp_size="${WORLD_SIZE}" \
    optim.lr="${LEARNING_RATE}" \
    trainer.device=npu \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.project_name=modelarts-sft-packed \
    trainer.experiment_name=qwen36-packed-smoke \
    trainer.total_training_steps="${TRAIN_STEPS}" \
    trainer.total_epochs=100 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.nnodes="${MA_NUM_HOSTS}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    trainer.logger="['console']" \
    checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    "$@" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo "Training finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
