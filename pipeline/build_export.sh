#!/usr/bin/env bash
set -euo pipefail
PIPELINE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$PIPELINE_DIR${PYTHONPATH:+:$PYTHONPATH}"

usage() {
  cat <<'USAGE'
Usage:
  bash build_export.sh audit    --src-root SRC --dst-root DST [--datasets ...]
  bash build_export.sh trainable --src-root SRC --dst-root DST [--datasets ...]
  bash build_export.sh split    --data-root DST --output-root SPLIT [--val-ratio 0.1]
  bash build_export.sh all      --src-root SRC --dst-root DST --output-root SPLIT [--datasets ...]

Commands:
  audit      Copy the source export and keep review rows; write rejected audit rows.
  trainable  Copy source, normalize formulas, and omit review rows.
  split      Split a normalized export into train/val using dataset_manifest.json.
  all        Run trainable export, then split it.

Common upstream paths:
  source   /data/verl_project/distill_outputs/final_merged_v1
  export   /data/verl_project/distill_outputs/<version>_trainable
  split    /data/verl_project/output/<version>_trainable_split
USAGE
}

run_mode=${1:-}
shift || true
case "$run_mode" in
  audit|trainable)
    profile=$run_mode
    python3 "$PIPELINE_DIR/normalize_formula_delims.py" --profile "$profile" "$@"
    ;;
  split)
    python3 /data/verl_project/verl/split_tools/split_vlm_manifest.py "$@"
    ;;
  all)
    parsed=()
    src=""; dst=""; split=""; val=0.1
    while [ $# -gt 0 ]; do
      case "$1" in
        --src-root) src=$2; shift 2;;
        --dst-root) dst=$2; shift 2;;
        --output-root) split=$2; shift 2;;
        --val-ratio) val=$2; shift 2;;
        *) parsed+=("$1"); shift;;
      esac
    done
    [ -n "$src" ] && [ -n "$dst" ] && [ -n "$split" ] || { usage >&2; exit 2; }
    python3 "$PIPELINE_DIR/normalize_formula_delims.py" \
      --profile trainable --src-root "$src" --dst-root "$dst" "${parsed[@]:-}"
    python3 /data/verl_project/verl/split_tools/split_vlm_manifest.py \
      --data-root "$dst" --manifest "$dst/dataset_manifest.json" \
      --output-root "$split" --val-ratio "$val" --chunk-rows 25 --overwrite
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
