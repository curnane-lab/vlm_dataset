#!/bin/bash
# 启动 NPU 教师实例池(纯 VLM 模式, 供 distill_qa.py 使用)
#
# 用法:
#   bash start_teachers.sh                        # 默认: 7 实例, 卡 0-6, 端口 30000-30006, 显存 0.8
#   bash start_teachers.sh -n 2                   # 2 实例: 取卡列表前 2 张
#   bash start_teachers.sh --cards "0 2"          # 指定用哪几张卡(每卡 1 实例)
#   bash start_teachers.sh --utils "0.5 0.2"      # 逐实例显存比例
#   bash start_teachers.sh --base-port 31000      # 改起始端口
#   bash start_teachers.sh -n 1 --cards "0" --utils "0.6" --base-port 30000
set -u

MODEL_DIR=/data/mineru/models/MinerU2.5-Pro-2604-1.2B
BASE_PORT=30000
NUM_INSTANCES=""
CARDS="0 1 2 3 4 5 6"
UTILS=""
DEFAULT_UTIL=0.8

usage() {
    sed -n '2,10p' "$0"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--num-instances) NUM_INSTANCES="$2"; shift 2;;
        --cards)            CARDS="$2"; shift 2;;
        --utils|--util)     UTILS="$2"; shift 2;;
        --base-port)        BASE_PORT="$2"; shift 2;;
        -h|--help)          usage; exit 0;;
        *) echo "未知参数: $1" >&2; usage; exit 1;;
    esac
done

read -ra card_arr <<< "$CARDS"
if [ -n "$NUM_INSTANCES" ]; then
    case "$NUM_INSTANCES" in
        ''|*[!0-9]*) echo "--num-instances 必须是正整数: $NUM_INSTANCES" >&2; exit 1;;
    esac
    if [ "$NUM_INSTANCES" -gt "${#card_arr[@]}" ]; then
        echo "实例数($NUM_INSTANCES)超过卡数(${#card_arr[@]}: $CARDS)" >&2; exit 1
    fi
    card_arr=("${card_arr[@]:0:$NUM_INSTANCES}")
fi
N=${#card_arr[@]}
if [ "$N" -lt 1 ]; then
    echo "至少需要 1 张卡" >&2; exit 1
fi

if [ -n "$UTILS" ]; then
    read -ra util_arr <<< "$UTILS"
    if [ "${#util_arr[@]}" -ne "$N" ]; then
        echo "--utils 数量(${#util_arr[@]})须与实例数($N)一致" >&2; exit 1
    fi
else
    # 默认: 每实例统一 DEFAULT_UTIL; 按实际占用用 --utils 覆盖
    util_arr=()
    for card in "${card_arr[@]}"; do
        util_arr+=( "$DEFAULT_UTIL" )
    done
fi

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn

for i in "${!card_arr[@]}"; do
    card=${card_arr[$i]}
    port=$((BASE_PORT + i))
    if curl -s --max-time 2 http://127.0.0.1:$port/v1/models 2>/dev/null | grep -q mineru-vlm; then
        echo "实例 $i (NPU$card, 端口 $port) 已在运行, 跳过"
        continue
    fi
    echo "启动实例 $i: NPU$card, 端口 $port, gpu-memory-utilization ${util_arr[$i]}"
    ASCEND_RT_VISIBLE_DEVICES=$card nohup mineru-kit vlm-server --engine vllm \
        --model "$MODEL_DIR" --served-model-name mineru-vlm \
        --host 127.0.0.1 --port $port \
        --gpu-memory-utilization "${util_arr[$i]}" \
        --hf-overrides '{"tie_word_embeddings":true}' \
        > "/data/mineru/vlm_server_npu$card.log" 2>&1 &
done

echo "等待实例就绪 ..."
rc=0
for i in "${!card_arr[@]}"; do
    card=${card_arr[$i]}
    port=$((BASE_PORT + i))
    for _ in $(seq 1 60); do
        curl -s --max-time 3 http://127.0.0.1:$port/v1/models 2>/dev/null | grep -q mineru-vlm && break
        sleep 5
    done
    if curl -s --max-time 3 http://127.0.0.1:$port/v1/models 2>/dev/null | grep -q mineru-vlm; then
        echo "实例 $i (NPU$card, 端口 $port) 就绪"
    else
        echo "实例 $i (NPU$card, 端口 $port) 启动失败, 看 vlm_server_npu$card.log" >&2
        rc=1
    fi
done
[ $rc -eq 0 ] && echo "教师实例池全部就绪: ${N} 个实例, 端口 $BASE_PORT..$((BASE_PORT + N - 1)) (model=mineru-vlm)"
exit $rc
