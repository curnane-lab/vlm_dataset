#!/usr/bin/env bash
# ============================================================================
# ModelArts Qwen3.6-35B-A3B SFT 训练启动脚本（精简版）
#
# MA 平台只指定以下路径（通过环境变量）：
#   MODEL_PATH   模型权重路径
#   DATA_PATH    数据根目录（包含 *.parquet）
#   OUTPUT_DIR   输出路径（日志/checkpoint/图片）
#
# 节点拓扑（也可由 MA 平台自动注入）：
#   NNODES         节点数，默认 1
#   NODE_RANK      当前节点 rank，默认 0
#   NPROC_PER_NODE 每节点进程/die 数，默认 16
#
# 其余训练超参均有合理默认值，可按需覆盖（见下方注释）。
# ============================================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ============================ 节点拓扑 ============================
NNODES=${NNODES:-${MA_NUM_HOSTS:-1}}
NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-0}}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
RDZV_ID=${RDZV_ID:-qwen36_sft_${NNODES}nodes}

# ============================ 平台路径 ============================
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
DATA_PATH=${DATA_PATH:?DATA_PATH is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
VERL_HOME=${VERL_HOME:-/verl}

# ============================ 训练超参 ============================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-0}      # 0 = auto (world_size)
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-8192}         # packed: max tokens/GPU; right-pad: max seq len
MAX_LENGTH=${MAX_LENGTH:-16384}              # right-padding only (ignored in packed mode)
TRAIN_STEPS=${TRAIN_STEPS:-100}
SAVE_FREQ=${SAVE_FREQ:-${TRAIN_STEPS}}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
RESUME_MODE=${RESUME_MODE:-disable}
VAL_RATIO=${VAL_RATIO:-0.1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-16}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
USE_PACKED=${USE_PACKED:-true}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-false}
NUM_WORKERS=${NUM_WORKERS:-0}
NPU_DEVICES=${NPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}

# Derived: packing mode
if [ "${USE_PACKED}" = "true" ]; then
    PAD_MODE=no_padding
    USE_REMOVE_PADDING=true
    EFFECTIVE_MAX_LEN=${MAX_TOKEN_LEN}
else
    PAD_MODE=right
    USE_REMOVE_PADDING=false
    EFFECTIVE_MAX_LEN=${MAX_LENGTH}
fi

# Derived: global batch size
WORLD_SIZE=$(( NPROC_PER_NODE * NNODES ))
if [ "${TRAIN_BATCH_SIZE}" = "0" ]; then
    TRAIN_BATCH_SIZE=${WORLD_SIZE}
fi

# ============================ 环境设置 ============================
export PYTHONPATH="${VERL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-${NPU_DEVICES}}
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:512"
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-30000-30200}
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export VERL_SFT_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

# ============================ 数据划分 ============================
SPLIT_DIR="${OUTPUT_DIR}/split_data"
TRAIN_FILES_LIST="${SPLIT_DIR}/train_files.hydra.txt"
VAL_FILES_LIST="${SPLIT_DIR}/val_files.hydra.txt"

# 清单优先级: $MANIFEST > $DATA_PATH/dataset_manifest.json >
# 不存在的 manifest.jsonl (此时切分脚本扫描 data-root 下全部 parquet)。
if [ -n "${MANIFEST:-}" ]; then
    MANIFEST_PATH="${MANIFEST}"
elif [ -f "${DATA_PATH}/dataset_manifest.json" ]; then
    MANIFEST_PATH="${DATA_PATH}/dataset_manifest.json"
else
    MANIFEST_PATH="${DATA_PATH}/manifest.jsonl"
fi

if [ ! -f "${TRAIN_FILES_LIST}" ] || [ ! -f "${VAL_FILES_LIST}" ]; then
    echo "[INFO] Splitting train/val (ratio=${VAL_RATIO})..."
    echo "[INFO] Manifest: ${MANIFEST_PATH}"
    mkdir -p "${SPLIT_DIR}"
    python3 "${SCRIPT_DIR}/split_tools/split_vlm_manifest.py" \
        --data-root "${DATA_PATH}" \
        --manifest "${MANIFEST_PATH}" \
        --output-root "${SPLIT_DIR}" \
        --val-ratio "${VAL_RATIO}" \
        --chunk-rows 25 \
        --overwrite
    for split in train val; do
        python3 - "${SPLIT_DIR}" "${split}" << 'PYEOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
split = sys.argv[2]
files = json.load(open(root / f"{split}_files.json"))
items = ",".join(f"'{p}'" for p in files)
(root / f"{split}_files.hydra.txt").write_text(f"[{items}]")
print(f"{split}: {len(files)} files")
PYEOF
    done
else
    echo "[INFO] Using cached split from ${SPLIT_DIR}"
fi

TRAIN_FILES=$(cat "${TRAIN_FILES_LIST}")
VAL_FILES=$(cat "${VAL_FILES_LIST}")

# ============================ 启动 ============================
mkdir -p "${OUTPUT_DIR}"
CKPT_DIR="${OUTPUT_DIR}/checkpoints"
LOG_FILE="${OUTPUT_DIR}/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo " ModelArts SFT"
echo "   Nodes=${NNODES}  Dies/node=${NPROC_PER_NODE}  World=${WORLD_SIZE}"
echo "   Node rank: ${NODE_RANK}  Master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "   Model:  ${MODEL_PATH}"
echo "   Data:   ${DATA_PATH}"
echo "   Output: ${OUTPUT_DIR}"
echo "   GBS=${TRAIN_BATCH_SIZE}  MBS=${MICRO_BATCH_SIZE}  MaxLen=${EFFECTIVE_MAX_LEN}"
echo "   Packed=${USE_PACKED}  Steps=${TRAIN_STEPS}  LR=${LEARNING_RATE}"
echo "============================================================"

cd "${VERL_HOME}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --rdzv_id="${RDZV_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.messages_key=messages \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    data.max_token_len_per_gpu="${EFFECTIVE_MAX_LEN}" \
    data.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    data.max_length="${EFFECTIVE_MAX_LEN}" \
    data.pad_mode="${PAD_MODE}" \
    data.ignore_input_ids_mismatch=True \
    data.truncation=right \
    data.num_workers="${NUM_WORKERS}" \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding="${USE_REMOVE_PADDING}" \
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
    trainer.project_name=modelarts-sft \
    trainer.experiment_name=qwen36-35b-a3b-sft \
    trainer.total_training_steps="${TRAIN_STEPS}" \
    trainer.total_epochs=100 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq=-1 \
    trainer.max_ckpt_to_keep=1 \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    trainer.logger="['console']" \
    checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    "$@" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

# Plot metrics
METRICS_IMG="${OUTPUT_DIR}/training_metrics.png"
python3 "${SCRIPT_DIR}/split_tools/plot_metrics.py" --log "${LOG_FILE}" --output "${METRICS_IMG}" || true

echo "============================================================"
echo " Finished: exit=${EXIT_CODE}"
echo " Checkpoint: ${CKPT_DIR}/global_step_${TRAIN_STEPS}/"
echo " Log:        ${LOG_FILE}"
echo " Metrics:    ${METRICS_IMG}"
echo "============================================================"
exit ${EXIT_CODE}
