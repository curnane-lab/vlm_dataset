# -*- coding: utf-8 -*-
"""
footer_check.py — manifest 驱动的 parquet 分片体检 (Linux/通用)

用途: 数据传输(拷贝/同步)之后、蒸馏/训练之前, 对数据根目录下 manifest
登记的全部分片做体检, 提前暴露传输截断/损坏/行数不符的分片, 避免任务
跑到中途才崩 (事故背景见 README 第 6 节: 2026-10-21 目标机
PubTables-1M 首分片传输截断)。

两级检查:
  默认        : 存在性 + footer 可读 (read_metadata) + 行数比对 manifest
  --full-read : 追加 iter_batches 全量流式读逐行计数 (慢, 深度校验,
                可发现 footer 完好但数据区损坏的情况)

用法:
  python3 footer_check.py --data-root /data/omnidoc_vlm
  python3 footer_check.py --data-root /data/omnidoc_vlm \
      --manifest /data/omnidoc_vlm/dataset_manifest_tier2.json \
      --report /tmp/footer_report.json
  python3 footer_check.py --data-root /data/omnidoc_vlm \
      --datasets UniMER_Dataset_vlm --full-read

退出码: 0 = 全部正常; 1 = 存在缺失/损坏/行数不符 (清单打印 stderr,
        加 --report 可同时落 JSON 供重传脚本使用)
依赖: pyarrow; 复用 distill_qa.py 的 norm_rel_path
"""

import argparse
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_qa import norm_rel_path  # noqa: E402


def check_file(path, expect_rows, full_read):
    """返回 None(正常) 或 问题描述字符串。"""
    if not os.path.exists(path):
        return "缺失"
    try:
        n = pq.read_metadata(path).num_rows
    except Exception as e:  # noqa: BLE001
        return f"footer 不可读: {type(e).__name__}"
    if expect_rows is not None and n != expect_rows:
        return f"行数 {n} != manifest {expect_rows}"
    if full_read:
        try:
            cnt = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=2000):
                cnt += batch.num_rows
        except Exception as e:  # noqa: BLE001
            return f"数据区读取失败: {type(e).__name__}"
        if cnt != n:
            return f"数据区行数 {cnt} != footer {n}"
    return None


def main():
    ap = argparse.ArgumentParser(description="parquet 分片 footer/数据体检")
    ap.add_argument("--data-root", default="/data/omnidoc_vlm")
    ap.add_argument("--manifest", default=None,
                    help="默认 <data-root>/dataset_manifest.json")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--full-read", action="store_true",
                    help="除 footer 外再全量流式读每个分片 (慢)")
    ap.add_argument("--report", default=None,
                    help="可选: 把体检结果写成 JSON 到该路径")
    args = ap.parse_args()

    manifest_path = args.manifest or os.path.join(
        args.data_root, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        ap.error(f"manifest 不存在: {manifest_path} "
                 f"(检查 --data-root 是否指向数据集根目录)")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    mode = "footer + 全量数据" if args.full_read else "footer"
    print(f"体检模式: {mode}  manifest: {manifest_path}")

    bad_files = []          # (rel, 问题)
    report_ds = []
    g_files = g_rows = 0
    for ds in manifest["datasets"]:
        name = ds["name"]
        if args.datasets and name not in args.datasets:
            continue
        files = ds.get("files") or []
        if args.max_files:
            files = files[:args.max_files]
        n_bad = ds_rows = 0
        for finfo in files:
            rel = norm_rel_path(finfo["path"])
            p = os.path.join(args.data_root, rel)
            prob = check_file(p, finfo.get("rows"), args.full_read)
            if prob:
                n_bad += 1
                bad_files.append((rel, prob))
                print(f"  [bad] {rel}: {prob}", file=sys.stderr)
            else:
                ds_rows += finfo.get("rows") or 0
        g_files += len(files)
        g_rows += ds_rows
        report_ds.append({"name": name, "files": len(files), "bad": n_bad,
                          "rows": ds_rows})
        print(f"=== {name}: {len(files)} 分片, 坏 {n_bad}, "
              f"{ds_rows:,} 行 {'OK' if not n_bad else ''} ===")

    n_bad = len(bad_files)
    print(f"\n体检完成: {g_files} 分片, {g_rows:,} 行, 异常 {n_bad}")
    if bad_files:
        print(f"!! {n_bad} 个分片异常, 请重传后复检:", file=sys.stderr)
        for rel, prob in bad_files[:50]:
            print(f"   {rel}: {prob}", file=sys.stderr)
        if n_bad > 50:
            print(f"   ... 其余 {n_bad - 50} 条见 --report JSON",
                  file=sys.stderr)

    if args.report:
        rep = {"data_root": os.path.abspath(args.data_root),
               "manifest": os.path.abspath(manifest_path),
               "full_read": args.full_read,
               "total_files": g_files, "total_rows": g_rows,
               "bad_count": n_bad,
               "bad_files": [{"path": r, "problem": p}
                             for r, p in bad_files],
               "datasets": report_ds}
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        print(f"报告: {args.report}")

    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
