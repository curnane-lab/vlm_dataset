# -*- coding: utf-8 -*-
"""
check_overlap_omnidocbench.py — 合训集与 OmniDocBench 评测集图像重合检查

动机: 训练数据若与 OmniDocBench (1651 页评测集) 存在同页/近重复,
评测分数会虚高。本脚本对指定数据集的图像与 ODB 页面图片做重合检查:
  1) 精确: 图片字节 md5
  2) 近似: 8x8 平均哈希 (aHash), 汉明距离 <= --max-hamming 判疑似重合

注意: Infinity/olmOCR 图像由源 PDF 渲染而来, 与 ODB 页面即使同源,
字节也几乎必然不同 -> md5 只作兜底, 主要靠 aHash。aHash 对缩放/重压缩
鲁棒, 但对版式微调不敏感; --max-hamming 默认 0 (最严), 放宽到 2-4 可
容忍更多渲染差异, 同时假阳性上升, 疑似对需人工复核。

用法 (在能访问 ODB 图片的机器上运行):
  python3 check_overlap_omnidocbench.py \
      --data-root F:/omnidoc_mix_s20 \
      --datasets Infinity-Doc-400K_parsing_vlm \
      --odb-images /data/mineru/datasets/OpenDataLab/OmniDocBench/images \
      --report overlap_report.json
退出码: 0 = 无重合; 1 = 发现疑似重合 (清单在报告里)。

依赖: pyarrow, Pillow; 复用 distill_qa.py 的 norm_rel_path。
"""

import argparse
import hashlib
import io
import json
import os
import sys

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_qa import norm_rel_path  # noqa: E402

AHASH_SIZE = 8


def ahash(image_bytes):
    """8x8 平均哈希, 返回 64 位 int。"""
    im = Image.open(io.BytesIO(image_bytes)).convert("L")
    im = im.resize((AHASH_SIZE, AHASH_SIZE), Image.LANCZOS)
    px = list(im.getdata())
    mean = sum(px) / len(px)
    bits = 0
    for p in px:
        bits = (bits << 1) | (1 if p >= mean else 0)
    return bits


def hamming(a, b):
    return (a ^ b).bit_count()


def main():
    ap = argparse.ArgumentParser(description="合训集 vs OmniDocBench 图像重合检查")
    ap.add_argument("--data-root", required=True, help="合训数据集根目录")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="只检查这些数据集 (默认全部)")
    ap.add_argument("--odb-images", required=True,
                    help="OmniDocBench images 目录 (1651 页)")
    ap.add_argument("--max-hamming", type=int, default=0,
                    help="aHash 汉明距离阈值, 默认 0 (精确同版式)")
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    manifest_path = args.manifest or os.path.join(
        args.data_root, "dataset_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # ---- ODB 侧建索引 ----
    odb_files = sorted(f for f in os.listdir(args.odb_images)
                       if f.lower().endswith(
                           (".png", ".jpg", ".jpeg", ".webp")))
    if not odb_files:
        ap.error(f"ODB 图片目录为空: {args.odb_images}")
    odb_md5 = {}
    odb_ahash = []  # (ahash_int, filename)
    for fn in odb_files:
        with open(os.path.join(args.odb_images, fn), "rb") as f:
            b = f.read()
        odb_md5.setdefault(hashlib.md5(b).hexdigest(), []).append(fn)
        odb_ahash.append((ahash(b), fn))
    print(f"ODB 索引: {len(odb_files)} 页 "
          f"({len(odb_md5)} 唯一 md5)", flush=True)

    # ---- 合训侧逐行比对 ----
    overlaps = []
    n_rows = n_imgs = 0
    for ds in manifest["datasets"]:
        name = ds["name"]
        if args.datasets and name not in args.datasets:
            continue
        for finfo in ds.get("files") or []:
            rel = norm_rel_path(finfo["path"])
            p = os.path.join(args.data_root, rel)
            if not os.path.exists(p):
                print(f"  [skip] missing {p}", file=sys.stderr)
                continue
            pf = pq.ParquetFile(p)
            base = 0
            for batch in pf.iter_batches(batch_size=200,
                                         columns=["images"]):
                for i, row in enumerate(batch.to_pylist()):
                    n_rows += 1
                    for im in row["images"] or []:
                        b = im["bytes"]
                        if not b:
                            continue
                        n_imgs += 1
                        h5 = hashlib.md5(b).hexdigest()
                        hit = None
                        if h5 in odb_md5:
                            hit = ("md5", 0, odb_md5[h5])
                        else:
                            ah = ahash(b)
                            close = [(d, fn) for oh, fn in odb_ahash
                                     if (d := hamming(ah, oh))
                                     <= args.max_hamming]
                            if close:
                                close.sort()
                                hit = ("ahash", close[0][0],
                                       [fn for _, fn in close])
                        if hit:
                            overlaps.append({
                                "dataset": name, "shard": rel,
                                "row_idx": base + i,
                                "method": hit[0], "hamming": hit[1],
                                "odb_files": hit[2]})
                base += batch.num_rows
            print(f"  {name}/{os.path.basename(rel)}: 累计 {n_rows} 行, "
                  f"重合 {len(overlaps)}", flush=True)

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"data_root": os.path.abspath(args.data_root),
                   "odb_images": os.path.abspath(args.odb_images),
                   "max_hamming": args.max_hamming,
                   "rows_checked": n_rows, "images_checked": n_imgs,
                   "overlap_count": len(overlaps),
                   "overlaps": overlaps}, f, ensure_ascii=False, indent=1)
    print(f"\n检查完成: {n_rows} 行 {n_imgs} 图, 疑似重合 {len(overlaps)}")
    for o in overlaps[:10]:
        print(f"  [{o['method']} h={o['hamming']}] {o['dataset']} "
              f"{os.path.basename(o['shard'])}#{o['row_idx']} <-> "
              f"{','.join(o['odb_files'][:3])}")
    print(f"报告: {args.report}")
    print("OVERLAP:", "FAIL" if overlaps else "CLEAN")
    sys.exit(1 if overlaps else 0)


if __name__ == "__main__":
    main()
