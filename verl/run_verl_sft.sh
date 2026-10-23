#!/usr/bin/env bash
# ============================================================================
# ModelArts Qwen3.6-35B-A3B 多数据集 SFT 训练启动脚本
# 兼容单节点和多节点（A3 910C 16 die per node）
#
# ModelArts 环境变量：
#   MA_NUM_HOSTS     节点数（由平台自动注入）
#   VC_WORKER_HOSTS  所有节点 IP 列表，逗号分隔（由平台自动注入）
#   VC_TASK_INDEX    当前节点 rank（由平台自动注入）
#
# 单节点用法：
#   bash run_verl_sft.sh
# 多节点用法（ModelArts 会自动注入环境变量）：
#   bash run_verl_sft.sh
# ============================================================================
set -euo pipefail

# ============================ 可配置参数 ============================
MODEL_PATH=${MODEL_PATH:-/data1/models/Qwen3.6-35B-A3B}
DATA_ROOT=${DATA_ROOT:-/data1/Qwen3.6_eval}
WORK_DIR=${WORK_DIR:-/data1/f00538480/verl_data/sft_smoke}
VERL_HOME=${VERL_HOME:-/verl}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# 训练超参
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-0}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
TRAIN_STEPS=${TRAIN_STEPS:-100}
VAL_RATIO=${VAL_RATIO:-0.1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}

# 输出
OUTPUT_DIR=${OUTPUT_DIR:-${WORK_DIR}/output}
SPLIT_DIR=${SPLIT_DIR:-${WORK_DIR}/split_data}

# ============================ 分布式拓扑（ModelArts 环境变量） ============================
# ModelArts 自动注入 MA_NUM_HOSTS / VC_WORKER_HOSTS / VC_TASK_INDEX
# 单节点时这些变量可能不存在，需要设置默认值
MA_NUM_HOSTS=${MA_NUM_HOSTS:-1}
VC_TASK_INDEX=${VC_TASK_INDEX:-0}
VC_WORKER_HOSTS=${VC_WORKER_HOSTS:-127.0.0.1}

# 每节点 die 数必须先于 WORLD_SIZE 赋值；set -u 下后置会直接退出
NPROC_PER_NODE=16

# 解析 rendezvous 地址：取第一个节点 IP
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NODE_RANK=${VC_TASK_INDEX}
WORLD_SIZE=$(( NPROC_PER_NODE * MA_NUM_HOSTS ))
MASTER_PORT=${MASTER_PORT:-29500}
RDZV_ID=${RDZV_ID:-qwen36_sft_${MA_NUM_HOSTS}nodes}

# GBS auto: world_size
if [ "${TRAIN_BATCH_SIZE}" = "0" ]; then
    TRAIN_BATCH_SIZE=${WORLD_SIZE}
fi

# ============================ 环境设置 ============================
export PYTHONPATH="${VERL_HOME}${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1800}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-26000-26200}
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export VERL_SFT_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1
export VERL_DISABLE_FLASH_ATTN_CE=1

# ============================ 数据划分 ============================
TRAIN_FILES_LIST="${SPLIT_DIR}/train_files.hydra.txt"
VAL_FILES_LIST="${SPLIT_DIR}/val_files.hydra.txt"

if [ ! -f "${TRAIN_FILES_LIST}" ] || [ ! -f "${VAL_FILES_LIST}" ]; then
    echo "[INFO] Generating train/val split (val_ratio=${VAL_RATIO})..."
    mkdir -p "${SPLIT_DIR}"

    MANIFEST="${SPLIT_DIR}/manifest.json"
    python3 - <<PYEOF
import json, glob, os
root = "${DATA_ROOT}"
datasets = []
for f in sorted(glob.glob(os.path.join(root, "*.parquet"))):
    name = os.path.basename(f).removesuffix("_sample200.parquet")
    rel = os.path.relpath(f, root)
    datasets.append({
        "name": name,
        "schema_adapter": "qwen_vl",
        "weight": 1.0,
        "total_rows": 200,
        "num_files": 1,
        "files": [{"path": rel, "rows": 200, "schema_adapter": "qwen_vl"}],
    })
m = {"version": 1, "data_root": root, "datasets": datasets}
with open("${MANIFEST}", "w") as fh:
    json.dump(m, fh, ensure_ascii=False, indent=2)
print(f"manifest: {len(datasets)} datasets")
PYEOF

    python3 "${SCRIPT_DIR}/split_vlm_manifest.py" \
        --data-root "${DATA_ROOT}" \
        --manifest "${MANIFEST}" \
        --output-root "${SPLIT_DIR}" \
        --val-ratio "${VAL_RATIO}" \
        --chunk-rows 25 \
        --overwrite

    for split in train val; do
        python3 - <<PYEOF
import json
files = json.load(open("${SPLIT_DIR}/${split}_files.json"))
inner = ",".join("'"+f+"'" for f in files)
open("${SPLIT_DIR}/${split}_files.hydra.txt", "w").write("["+inner+"]")
print(f"${split}: {len(files)} files")
PYEOF
    done
else
    echo "[INFO] Using existing split data from ${SPLIT_DIR}"
fi

TRAIN_FILES=$(cat "${TRAIN_FILES_LIST}")
VAL_FILES=$(cat "${VAL_FILES_LIST}")

echo "[INFO] Train files: $(python3 -c "import json; print(len(json.load(open('${SPLIT_DIR}/train_files.json'))))")"
echo "[INFO] Val files: $(python3 -c "import json; print(len(json.load(open('${SPLIT_DIR}/val_files.json'))))")"

# ============================ 启动训练 ============================
mkdir -p "${OUTPUT_DIR}"
CKPT_DIR="${OUTPUT_DIR}/checkpoints"
LOG_FILE="${OUTPUT_DIR}/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo " ModelArts SFT Training"
echo "   Nodes:      ${MA_NUM_HOSTS} x A3 910C (16 die each)"
echo "   Node rank:  ${NODE_RANK}"
echo "   Master:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "   World size: ${WORLD_SIZE}"
echo "   Model:      ${MODEL_PATH}"
echo "   Data root:  ${DATA_ROOT}"
echo "   Verl:       ${VERL_HOME}"
echo "   Output:     ${OUTPUT_DIR}"
echo "   GBS:        ${TRAIN_BATCH_SIZE}, MBS: ${MICRO_BATCH_SIZE}"
echo "   Max length: ${MAX_LENGTH}"
echo "   Steps:      ${TRAIN_STEPS}"
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
    data.val_files="${VAL_FILES}" \
    data.train_max_samples=-1 \
    data.val_max_samples=16 \
    data.messages_key=messages \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    data.max_token_len_per_gpu="${MAX_LENGTH}" \
    data.use_dynamic_bsz=False \
    data.max_length="${MAX_LENGTH}" \
    data.pad_mode=right \
    data.ignore_input_ids_mismatch=True \
    data.truncation=right \
    +data.image_patch_size=16 \
    data.num_workers=0 \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=False \
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
    trainer.save_freq="${TRAIN_STEPS}" \
    trainer.test_freq=-1 \
    trainer.max_ckpt_to_keep=1 \
    trainer.resume_mode=disable \
    trainer.nnodes="${MA_NUM_HOSTS}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    trainer.logger="['console']" \
    checkpoint.save_contents="['model','optimizer','extra','hf_model']" \
    "$@" 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

echo "============================================================"
echo " Training finished with exit code: ${EXIT_CODE}"
echo " Checkpoint: ${CKPT_DIR}/global_step_${TRAIN_STEPS}/"
echo " HF weights: ${CKPT_DIR}/global_step_${TRAIN_STEPS}/huggingface/"
echo " Log:        ${LOG_FILE}"
echo "============================================================"

exit ${EXIT_CODE}
