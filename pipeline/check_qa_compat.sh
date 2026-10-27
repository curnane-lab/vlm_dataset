#!/bin/bash
# =====================================================================
# check_qa_compat.sh — 拉起 Qwen VL 裁判服务, 对蒸馏产物做 Q-A 匹配质检
#
# 服务拉起机制与 start_teachers.sh 同源 (服务器已验证):
#   mineru-kit vlm-server --engine vllm + 同款环境变量 + 幂等复用 +
#   就绪轮询 + nohup 日志; 默认不传 MinerU 专属的 --hf-overrides。
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
#   LAUNCHER=mineru-kit  mineru-kit(默认, 与教师池同源) | vllm (裸 vllm serve)
#   HF_OVERRIDES=""    需要时传 '{"tie_word_embeddings":true}' 之类
#   PER_DATASET=200    每数据集抽样行数
#   WORKERS=16         裁判并发
#   MIN_MATCH_RATE=0.9 匹配率门禁 (低于则退出码 1; 仅统计模式)
#   FIX=0              1=修复模式: 全量逐行裁判, 不匹配行改写 Q 对齐 A
#                      (A 为 MinerU 蒸馏产物, 只读不动), 原地原子写回,
#                      .qafix_done 断点续跑 (仅用于蒸馏副本, 不可逆!)
#   REPORT=<DATA_ROOT>/qa_compat_report.json
#   EXTRA_ARGS=""      透传给启动命令的额外参数
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
LAUNCHER=${LAUNCHER:-mineru-kit}
HF_OVERRIDES=${HF_OVERRIDES:-}
PER_DATASET=${PER_DATASET:-200}
WORKERS=${WORKERS:-16}
MIN_MATCH_RATE=${MIN_MATCH_RATE:-0.9}
FIX=${FIX:-0}
REPORT=${REPORT:-"$DATA_ROOT/qa_compat_report.json"}
EXTRA_ARGS=${EXTRA_ARGS:-}
KEEP_SERVER=${KEEP_SERVER:-0}
LOG_DIR=${LOG_DIR:-/data/mineru}
LOG="$LOG_DIR/qa_judge_npu${CARD}.log"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn

STARTED=0
cleanup() {
    if [ "$STARTED" = "1" ] && [ "$KEEP_SERVER" != "1" ]; then
        echo "[cleanup] 关停裁判服务 (端口 $PORT)"
        pkill -f "\-\-port $PORT " 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=============================================="
echo "Q-A 匹配质检"
echo "裁判模型:   $MODEL_DIR ($SERVED_NAME, $LAUNCHER)"
echo "数据:       $DATA_ROOT"
echo "抽样:       每数据集 $PER_DATASET 行, 并发 $WORKERS"
if [ "$FIX" = "1" ]; then
    echo "模式:       修复 (全量逐行裁判, 不匹配行改写 Q 写回; A 不动)"
else
    echo "模式:       仅统计 (门禁 match >= $MIN_MATCH_RATE)"
fi
echo "=============================================="

# ---------- Step 0: 服务拉起/复用 (与 start_teachers.sh 同机制) ----------
if curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null \
        | grep -q "$SERVED_NAME"; then
    echo "[step0] 端口 $PORT 已有 $SERVED_NAME 服务, 复用"
else
    mkdir -p "$LOG_DIR"
    echo "[step0] 拉起裁判服务: NPU$CARD 端口 $PORT util $GPU_UTIL"
    if [ "$LAUNCHER" = "vllm" ]; then
        # shellcheck disable=SC2086
        ASCEND_RT_VISIBLE_DEVICES=$CARD nohup vllm serve "$MODEL_DIR" \
            --served-model-name "$SERVED_NAME" \
            --host 127.0.0.1 --port "$PORT" \
            --gpu-memory-utilization "$GPU_UTIL" \
            --trust-remote-code $EXTRA_ARGS \
            > "$LOG" 2>&1 &
    else
        # 与 start_teachers.sh 同款拉起 (服务器已验证);
        # MinerU 专属 --hf-overrides 默认不带, 需要时经 HF_OVERRIDES 传入
        HF_OV=()
        [ -n "$HF_OVERRIDES" ] && HF_OV=(--hf-overrides "$HF_OVERRIDES")
        # shellcheck disable=SC2086
        ASCEND_RT_VISIBLE_DEVICES=$CARD nohup mineru-kit vlm-server \
            --engine vllm \
            --model "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
            --host 127.0.0.1 --port "$PORT" \
            --gpu-memory-utilization "$GPU_UTIL" \
            "${HF_OV[@]}" $EXTRA_ARGS \
            > "$LOG" 2>&1 &
    fi
    STARTED=1
    echo "[step0] 等待就绪 (日志 $LOG) ..."
    ok=0
    for _ in $(seq 1 360); do   # 最长 30 分钟 (裁判模型比 1.2B 大, 加载慢)
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

# ---------- Step 1: 抽样裁判 / 修复 ----------
if [ "$FIX" = "1" ]; then
    echo "[step1] 修复模式: 全量逐行裁判 + 改写 Q ..."
    python3 qa_compat_check.py \
        --data-root "$DATA_ROOT" \
        --model "$SERVED_NAME" \
        --num-instances 1 --base-port "$PORT" \
        --workers "$WORKERS" \
        --fix --resume \
        --report "$REPORT"
else
    echo "[step1] 抽样裁判检查 ..."
    python3 qa_compat_check.py \
        --data-root "$DATA_ROOT" \
        --model "$SERVED_NAME" \
        --num-instances 1 --base-port "$PORT" \
        --per-dataset "$PER_DATASET" \
        --workers "$WORKERS" \
        --min-match-rate "$MIN_MATCH_RATE" \
        --report "$REPORT"
fi
rc=$?

echo ""
if [ $rc -eq 0 ]; then
    echo "Q-A 质检通过, 报告: $REPORT"
else
    echo "Q-A 质检存在不达标数据集, 报告: $REPORT" >&2
fi
exit $rc
