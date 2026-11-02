#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/data/verl_project/output/final_merged_v3_gbs64_step500/checkpoints/global_step_500/huggingface}
GT=/data/mineru/datasets/OpenDataLab/OmniDocBench/OmniDocBench.json
IMAGES=/data/mineru/datasets/OpenDataLab/OmniDocBench/images
OUT_ROOT=${OUT_ROOT:-/data/mineru/odb_v3_sft_nonthink_4npu_eval}
BASE_PORT=32400
INSTANCE_CARDS=(10,11 12,13)
READY_TIMEOUT=${READY_TIMEOUT:-3600}
MAX_TOKENS=32768
SAVE_NAME=${SAVE_NAME:-qwen36_v3_sft_nonthink_4npu}

for required in "$MODEL_DIR/config.json" "$MODEL_DIR/model.safetensors.index.json" "$GT" "$IMAGES"; do
  test -e "$required" || { echo "missing required path: $required" >&2; exit 1; }
done

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/preds" "$OUT_ROOT/metadata" "$OUT_ROOT/cache"

source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1 || true
unset LD_PRELOAD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_BUFFSIZE=512
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_SERVER_DEV_MODE=1
export PYTHONPATH="/data/verl_project/qa_stats/engine_compat${PYTHONPATH:+:$PYTHONPATH}"
export CDM_TEXLIVE_ROOT=/data/mineru/texlive
export CDM_TEXLIVE_BIN=/data/mineru/texlive/bin/aarch64-linux
export PATH=/data/mineru/bin:/data/mineru/texlive/bin/aarch64-linux:$PATH

pids=()
servers=""
for i in "${!INSTANCE_CARDS[@]}"; do
  cards=${INSTANCE_CARDS[$i]}
  port=$((BASE_PORT+i))
  export TRITON_CACHE_DIR="$OUT_ROOT/cache/triton_$i"
  export TRITON_CACHE_HOME="$OUT_ROOT/cache/triton_$i"
  export TRITON_KERNEL_CACHE_DIR="$OUT_ROOT/cache/triton_$i/kernels"
  export XDG_CACHE_HOME="$OUT_ROOT/cache/xdg_$i"
  export MPLCONFIGDIR="$OUT_ROOT/cache/mpl_$i"
  mkdir -p "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

  ASCEND_RT_VISIBLE_DEVICES=$cards python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name qwen-vl \
    --host 127.0.0.1 \
    --port "$port" \
    --tensor-parallel-size 2 \
    --max-model-len 57344 \
    --max-num-seqs 12 \
    --max-num-batched-tokens 8192 \
    --dtype bfloat16 \
    --trust-remote-code \
    --log-error-stack \
    --mm-processor-cache-gb 0 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    > "$OUT_ROOT/logs/server_$i.log" 2>&1 &
  pids+=("$!")
  servers="${servers:+$servers,}http://127.0.0.1:${port}"
done

cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

deadline=$((SECONDS + READY_TIMEOUT))
ready=0
while [ "$SECONDS" -lt "$deadline" ]; do
  ready=0
  for port in $(seq "$BASE_PORT" "$((BASE_PORT + ${#INSTANCE_CARDS[@]} - 1))"); do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      ready=$((ready + 1))
    fi
  done
  [ "$ready" -eq "${#INSTANCE_CARDS[@]}" ] && break
  alive=0
  for pid in "${pids[@]:-}"; do kill -0 "$pid" 2>/dev/null && alive=$((alive + 1)); done
  [ "$alive" -eq "${#INSTANCE_CARDS[@]}" ] || {
    echo "a server died during startup" >&2
    tail -80 "$OUT_ROOT/logs/server_0.log" "$OUT_ROOT/logs/server_1.log" >&2 || true
    exit 1
  }
  sleep 10
done
[ "$ready" -eq "${#INSTANCE_CARDS[@]}" ] || {
  echo "servers not ready: $ready/${#INSTANCE_CARDS[@]}" >&2
  exit 1
}
echo "servers ready: $servers"

python3 /data/mineru/qwen_infer_omnidocbench_nonthink.py \
  --gt "$GT" \
  --images "$IMAGES" \
  --out "$OUT_ROOT/preds" \
  --metadata-dir "$OUT_ROOT/metadata" \
  --servers "$servers" \
  --model qwen-vl \
  --concurrency-per-instance 12 \
  --max-tokens "$MAX_TOKENS" \
  --timeout 1800 \
  --resume \
  2>&1 | tee "$OUT_ROOT/logs/infer.log"

pred_count=$(find "$OUT_ROOT/preds" -maxdepth 1 -name '*.md' | wc -l)
failed_count=$(rg -l '<!-- inference failed:' "$OUT_ROOT/preds" 2>/dev/null | wc -l || true)
truncated_count=$(rg -l '"finish_reason": "length"' "$OUT_ROOT/metadata" 2>/dev/null | wc -l || true)
think_count=$(rg -l '<think>' "$OUT_ROOT/preds" 2>/dev/null | wc -l || true)
echo "prediction count: $pred_count, failed: $failed_count, truncated: $truncated_count, think: $think_count"
[ "$pred_count" -eq 1651 ] && [ "$failed_count" -eq 0 ] || {
  echo "inference incomplete" >&2
  exit 1
}

cat > "$OUT_ROOT/eval.yaml" <<EOF
end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist, CDM]
      cdm_workers: 32
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: 32
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: $GT
    prediction:
      data_path: $OUT_ROOT/preds
    match_method: quick_match
    match_workers: 32
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
EOF

source /data/mineru/odb_venv/bin/activate
cd /data/mineru/OmniDocBench
export save_name="$SAVE_NAME"
python pdf_validation.py --config "$OUT_ROOT/eval.yaml" \
  2>&1 | tee "$OUT_ROOT/logs/eval.log"

mkdir -p "$OUT_ROOT/result"
cp "./result/${SAVE_NAME}_metric_result.json" "$OUT_ROOT/result/" || true
cp "./result/${SAVE_NAME}_run_summary.json" "$OUT_ROOT/result/" || true
cp "./result/${SAVE_NAME}_stage_execution.json" "$OUT_ROOT/result/" || true
cp "./result/${SAVE_NAME}_runtime_environment.json" "$OUT_ROOT/result/" || true

echo "evaluation completed at $(date -Is)" | tee -a "$OUT_ROOT/logs/run.log"
