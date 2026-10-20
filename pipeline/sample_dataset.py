# -*- coding: utf-8 -*-
"""
sample_dataset.py — manifest 驱动的确定性下采样预处理 (Linux)

用途: 先导试验前把大数据集缩成小数据集。例如 tier2 共 192 万行,
--sample-rate 0.05 (20 倍下采样) 后约 9.6 万行, 写到 regen 目录,
之后对该目录直接跑 distill_qa.py --in-place 完成蒸馏。

流程:
  1. 读取 manifest, 枚举 parquet 分片
  2. iter_batches 流式读源分片, crc32(分片相对路径#行号) 确定性选行
     (同分片同采样率重跑选中行完全一致, 无随机数)
  3. ParquetWriter 按源 schema 写到输出根 (相同相对目录结构)
  4. 输出根生成行数已修正的新 dataset_manifest.json, 其余字段保留
源数据全程只读。

后续蒸馏 (对采样副本原地刷写, 不碰源数据):
  python3 distill_qa.py --data-root <输出根> --in-place \
    --num-instances 7 --base-port 30000 --model mineru-vlm \
    --workers-per-instance 8 --resume

依赖: pandas, pyarrow; 复用 distill_qa.py 的 make_sampler/norm_rel_path
"""

import argparse
import json
import os
import sys

import pandas as pd  # noqa: F401  (pyarrow to_pandas 依赖)
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_qa import make_sampler, norm_rel_path  # noqa: E402


def sample_file(src_path, dst_path, sampler, rel, chunk_rows):
    """流式采样一个分片, 返回 (源行数, 采样行数)。"""
    pf = pq.ParquetFile(src_path)
    total = pf.metadata.num_rows
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    seen = kept = 0
    writer = pq.ParquetWriter(dst_path, schema=pf.schema_arrow)
    try:
        for batch in pf.iter_batches(batch_size=chunk_rows):
            chunk = batch.to_pandas()
            mask = [sampler(rel, i) for i in range(seen, seen + len(chunk))]
            seen += len(chunk)
            chunk = chunk[mask]
            if len(chunk) == 0:
                continue
            writer.write_table(
                pa.Table.from_pandas(chunk, schema=pf.schema_arrow,
                                     preserve_index=False))
            kept += len(chunk)
    finally:
        writer.close()
    return total, kept


def main():
    ap = argparse.ArgumentParser(description="VLM 数据集确定性下采样预处理")
    ap.add_argument("--data-root", default="/data/omnidoc_vlm")
    ap.add_argument("--manifest", default=None,
                    help="默认 <data-root>/dataset_manifest.json")
    ap.add_argument("--output-root", required=True,
                    help="采样输出根目录 (之后对它跑 distill_qa.py --in-place)")
    ap.add_argument("--sample-rate", type=float, required=True,
                    help="采样比例 (0,1), 如 0.05 = 20 倍下采样")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--chunk-rows", type=int, default=500)
    args = ap.parse_args()

    if not 0 < args.sample_rate < 1:
        ap.error("--sample-rate 必须在 (0,1) 区间; 不采样请直接用原数据")
    if os.path.abspath(args.output_root) == os.path.abspath(args.data_root):
        ap.error("--output-root 不能与 --data-root 相同")

    manifest_path = args.manifest or os.path.join(
        args.data_root, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        ap.error(f"manifest 不存在: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    sampler = make_sampler(args.sample_rate)
    print(f"采样率: {args.sample_rate} (约 1/{round(1 / args.sample_rate)})"
          f"  输出: {args.output_root}")

    out_manifest = {k: v for k, v in manifest.items() if k != "datasets"}
    out_manifest["data_root"] = os.path.abspath(args.output_root)
    out_manifest["datasets"] = []

    grand_src = grand_kept = 0
    for ds in manifest["datasets"]:
        name = ds["name"]
        if args.datasets and name not in args.datasets:
            continue
        ds_dir = os.path.join(args.data_root, name)
        if not os.path.isdir(ds_dir):
            print(f"=== {name}: 本地目录缺失, 跳过 ===", file=sys.stderr)
            continue
        files = ds.get("files") or []
        if args.max_files:
            files = files[:args.max_files]

        out_ds = {k: v for k, v in ds.items() if k != "files"}
        out_ds["files"] = []
        ds_src = ds_kept = 0
        for finfo in files:
            rel = norm_rel_path(finfo["path"])
            src = os.path.join(args.data_root, rel)
            dst = os.path.join(args.output_root, rel)
            if not os.path.exists(src):
                print(f"  [skip] missing {src}", file=sys.stderr)
                continue
            total, kept = sample_file(src, dst, sampler, rel,
                                      args.chunk_rows)
            ds_src += total
            ds_kept += kept
            entry = dict(finfo)
            entry["path"] = rel
            entry["rows"] = kept
            out_ds["files"].append(entry)
        out_ds["num_files"] = len(out_ds["files"])
        out_ds["total_rows"] = ds_kept
        out_manifest["datasets"].append(out_ds)
        grand_src += ds_src
        grand_kept += ds_kept
        print(f"=== {name}: {ds_src} -> {ds_kept} 行 "
              f"({len(out_ds['files'])} 分片) ===")

    mpath = os.path.join(args.output_root, "dataset_manifest.json")
    os.makedirs(args.output_root, exist_ok=True)
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(out_manifest, f, ensure_ascii=False, indent=2)
    print(f"\n完成: {grand_src} -> {grand_kept} 行 "
          f"(实际比例 {grand_kept / max(grand_src, 1):.4f})")
    print(f"新 manifest: {mpath}")
    print(f"后续: python3 {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'distill_qa.py')} "
          f"--data-root {args.output_root} --in-place --resume ...")


if __name__ == "__main__":
    main()
