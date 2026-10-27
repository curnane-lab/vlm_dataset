#!/bin/bash
# =====================================================================
# check_qa_compat.sh — 拉起 Qwen VL 裁判服务, 对蒸馏产物做 Q-A 匹配质检
#
# 流程: 拉起 vLLM 服务 (已在跑则复用) -> 等就绪 -> qa_compat_check.py
#       抽样裁判 -> 输出统计/JSON 报告 -> 按需关停服务
#
# 用法 (NPU 训练机, 建议用空闲卡, 避开教师池 0-6 卡):
#   MODEL_DIR=/data/models/Qwen2.5-VL-32B-Instruct \
#   DATA_ROOT=/data/omnidoc_vlm_regen/tier2 \
#   bash check_qa_compat.sh
#
# 可调环境变量:
#   MODEL_DIR (必填)   Qwen VL 模型路径 (裁判)
#   DATA_ROOT (必填)   蒸馏产物根目录 (含 dataset_manifest.json)
#   CARD=7             用哪张卡
#   PORT=31000         裁判服务端口
#   GPU_UTIL=0.85      显存比例
#   SERVED_NAME=qwen-vl
#   PER_DATASET=200    每数据集抽样行数
#   WORKERS=16         裁判并发
#   MIN_MATCH_RATE=0.9 匹配率门禁 (低于则退出码 1)
#   REPORT=<DATA_ROOT>/qa_compat_report.json
#   EXTRA_ARGS=""      透传给 vllm serve 的额外参数
#   KEEP_SERVER=0      1=检查完保留服务 (复跑省时)
# =====================================================================
set -u
cd "$(dirname "$0")"

: "${MODEL_DIR:?需要 MODEL_DIR (Qwen VL 裁判模型路径)}"
: "${DATA_ROOT:?需要 DATA_ROOT (蒸馏产物根目录)}"
CARD=${CARD:-7}
PORT=${PORT:-31000}
GPU_UTIL=${GPU_UTIL:-0.85}
SERVED_NAME=${SERVED_NAME:-qwen-vl}
PER_DATASET=${PER_DATASET:-200}
WORKERS=${WORKERS:-16}
MIN_MATCH_RATE=${MIN_MATCH_RATE:-0.9}
REPORT=${REPORT:-"$DATA_ROOT/qa_compat_report.json"}
EXTRA_ARGS=${EXTRA_ARGS:-}
KEEP_SERVER=${KEEP_SERVER:-0}
LOG=${LOG:-/tmp/qa_judge_vllm_${PORT}.log}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn

STARTED=0
cleanup() {
    if [ "$STARTED" = "1" ] && [ "$KEEP_SERVER" != "1" ]; then
        echo "[cleanup] 关停裁判服务 (端口 $PORT)"
        pkill -f "vllm serve.*--port $PORT" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=============================================="
echo "Q-A 匹配质检"
echo "裁判模型:   $MODEL_DIR ($SERVED_NAME)"
echo "数据:       $DATA_ROOT"
echo "抽样:       每数据集 $PER_DATASET 行, 并发 $WORKERS"
echo "门禁:       match >= $MIN_MATCH_RATE"
echo "=============================================="

# ---------- Step 0: 服务拉起/复用 ----------
if curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
        | grep -q "$SERVED_NAME"; then
    echo "[step0] 端口 $PORT 已有 $SERVED_NAME 服务, 复用"
else
    echo "[step0] 拉起裁判服务: NPU$CARD 端口 $PORT util $GPU_UTIL"
    # shellcheck disable=SC2086
    ASCEND_RT_VISIBLE_DEVICES=$CARD nohup vllm serve "$MODEL_DIR" \
        --served-model-name "$SERVED_NAME" \
        --host 127.0.0.1 --port "$PORT" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --trust-remote-code $EXTRA_ARGS \
        > "$LOG" 2>&1 &
    STARTED=1
    echo "[step0] 等待就绪 (日志 $LOG) ..."
    ok=0
    for _ in $(seq 1 360); do   # 最长 30 分钟 (大模型加载慢)
        if curl -s --max-time 3 "http://127.0.0.1:$PORT/v1/models" \
                2>/dev/null | grep -q "$SERVED_NAME"; then
            ok=1; break
        fi
        sleep 5
    done
    if [ "$ok" != "1" ]; then
        echo "[error] 裁判服务启动超时, 看 $LOG" >&2
        exit 1
    fi
    echo "[step0] 服务就绪"
fi

# ---------- Step 1: 抽样裁判 ----------
echo "[step1] 抽样裁判检查 ..."
python3 qa_compat_check.py \
    --data-root "$DATA_ROOT" \
    --model "$SERVED_NAME" \
    --num-instances 1 --base-port "$PORT" \
    --per-dataset "$PER_DATASET" \
    --workers "$WORKERS" \
    --min-match-rate "$MIN_MATCH_RATE" \
    --report "$REPORT"
rc=$?

echo ""
if [ $rc -eq 0 ]; then
    echo "Q-A 质检通过, 报告: $REPORT"
else
    echo "Q-A 质检存在不达标数据集, 报告: $REPORT" >&2
fi
exit $rc
