# -*- coding: utf-8 -*-
"""
convert_verl_to_hyperparallel.py — verl SFT parquet -> hyper-parallel data/omni 格式

用途: 把本仓统一 qwen_vl schema (verl.trainer.sft_trainer 直读的
messages + images{bytes,path} parquet) 转成 hyper-parallel
hyper_parallel.data.omni (OmniMappingDataset + AutoProcessorTransform)
的训练数据。两边的 messages/images 语义一致, 转换只做两件事:
  1. 契约校验 (hyper-parallel 比 verl 更严格, 违约行直接丢弃并计数):
     - messages 非空 list, >=2 条且末条 role=assistant, content 非空
     - 所有 content 必须是 str, 统计 "<image>" 占位符总数
     - 占位符数 == images 数 (hyper-parallel 会因不等直接报错)
     - 图片 bytes 按 magic 校验 (PNG/JPEG/WEBP/GIF/BMP), 可选 PIL 全解码
  2. 落盘格式适配:
     - --format jsonl (默认): 图片抽出写到 <output-root>/<--images-dir-name>/,
       行内 images 引用相对路径, sha1 去重跨分片共享
       (hyper-parallel 以数据文件所在目录锚定相对路径; 其 JSONL 读取器的
       images 只支持 path 字符串, 不能内嵌 bytes, 抽盘是必须的)
     - --format parquet: 原样保留 images{bytes,path}, 只做校验过滤直写
       (hyper-parallel 的 parquet 读取器原生支持内嵌 bytes, 无需抽盘)

依赖: pyarrow; --validate-decode 时额外需要 Pillow。

示例:
  python3 convert_verl_to_hyperparallel.py \
    --input /data/.../split_data/vlm_parquet \
    --output-root /data/.../hyperparallel_omni \
    --format jsonl --validate-decode

hyper-parallel 侧 (示例, 数据段粘进模型 yaml, 参考 examples/data/omni_data.yaml):
  dataset:
    _target_: hyper_parallel.data.omni.build_dataset.build_online_omni_mapping_dataset
    data_path: <output-root>/v2_crop_train-00000-of-00001.jsonl
    data_config: {}
    data_transform:
      _target_: hyper_parallel.data.omni.omni_transform.AutoProcessorTransform
      max_seq_len: 8192
  dataloader:
    _target_: hyper_parallel.data.omni.loader.OmniPackingLoader
    collate_fn:
      _target_: hyper_parallel.data.omni.collator.OmniCollator
"""

import argparse
import hashlib
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

# 图片 magic -> 扩展名; hyper-parallel 侧 PIL convert("RGB") 同样认这些格式
MAGIC_EXTS = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
)


def image_ext(data):
    """按 magic 判断图片扩展名, 未知格式返回 None。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:2] == b"BM":
        return ".bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


def count_image_placeholders(messages):
    """统计所有 str content 里的 <image> 占位符 (对齐 verl re.split 语义)。"""
    return sum(
        m.get("content", "").count("<image>")
        for m in messages
        if isinstance(m, dict) and isinstance(m.get("content"), str)
    )


def validate_row(row):
    """按 hyper-parallel omni 契约校验一行。

    返回 (messages, images, None) 或 (None, None, 丢弃原因)。
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, None, "messages_empty"
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        return None, None, "last_message_not_assistant"
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            return None, None, "content_not_str"
        if not m.get("content", "").strip():
            return None, None, "content_empty"

    images = row.get("images") or []
    if count_image_placeholders(messages) != len(images):
        return None, None, "placeholder_images_mismatch"

    for img in images:
        data = img.get("bytes") if isinstance(img, dict) else None
        if not isinstance(data, (bytes, bytearray)) or not data:
            return None, None, "image_bytes_missing"
        if image_ext(bytes(data[:12])) is None:
            return None, None, "image_format_unknown"
    return messages, images, None


def maybe_decode_check(images):
    """--validate-decode: PIL 全解码并转 RGB, 失败抛异常。"""
    from io import BytesIO

    from PIL import Image

    for img in images:
        with Image.open(BytesIO(img["bytes"])) as im:
            im.convert("RGB").load()


def write_image(images_dir, data, ext, dedup_cache):
    """sha1 去重落盘一张图, 返回相对数据文件目录的引用路径。"""
    data = bytes(data)
    sha = hashlib.sha1(data).hexdigest()[:20]
    rel = dedup_cache.get(sha)
    if rel is not None:
        return rel
    name = f"{sha}{ext}"
    dst = os.path.join(images_dir, name)
    if not os.path.exists(dst):
        with open(dst, "wb") as f:
            f.write(data)
    dir_name = os.path.basename(images_dir.rstrip("/\\"))
    rel = f"{dir_name}/{name}"
    dedup_cache[sha] = rel
    return rel


def convert_file(src, args, images_dir, dedup_cache, stats):
    """转换一个分片, 返回 (读行数, 保留行数)。"""
    stem = os.path.basename(src)[: -len(".parquet")]
    out_path = os.path.join(args.output_root,
                            stem + (".jsonl" if args.format == "jsonl" else ".parquet"))
    if os.path.exists(out_path) and not args.overwrite:
        print(f"  [skip] 已存在 {out_path} (--overwrite 覆盖)")
        return 0, 0

    pf = pq.ParquetFile(src)
    total = pf.metadata.num_rows

    jsonl_fp = None
    writer = None
    read_rows = kept = dropped = 0
    reasons = {}
    try:
        if args.format == "jsonl":
            jsonl_fp = open(out_path, "w", encoding="utf-8", newline="\n")
        else:
            writer = pq.ParquetWriter(out_path, pf.schema_arrow)

        for batch in pf.iter_batches(batch_size=args.chunk_rows):
            for row in batch.to_pylist():
                read_rows += 1
                messages, images, reason = validate_row(row)
                if messages is None:
                    dropped += 1
                    reasons[reason] = reasons.get(reason, 0) + 1
                    continue
                if args.validate_decode:
                    try:
                        maybe_decode_check(images)
                    except Exception:  # noqa: BLE001
                        dropped += 1
                        reasons["image_decode_failed"] = reasons.get("image_decode_failed", 0) + 1
                        continue

                if args.format == "jsonl":
                    rel_images = []
                    for img in images:
                        data = bytes(img["bytes"])
                        rel_images.append(
                            write_image(images_dir, data, image_ext(data[:12]), dedup_cache))
                    line = json.dumps(
                        {"messages": messages, "images": rel_images},
                        ensure_ascii=False, separators=(",", ":"))
                    jsonl_fp.write(line + "\n")
                else:
                    writer.write_table(pa.Table.from_pylist(
                        [{"messages": messages, "images": images}],
                        schema=pf.schema_arrow))
                kept += 1
                if args.limit and kept >= args.limit:
                    break
            if args.limit and kept >= args.limit:
                break
            if read_rows % (args.chunk_rows * 32) == 0:
                print(f"  ... {read_rows}/{total} 行 (保留 {kept})",
                      file=sys.stderr)
    finally:
        if jsonl_fp is not None:
            jsonl_fp.close()
        if writer is not None:
            writer.close()

    print(f"  {os.path.basename(src)}: {read_rows} -> {kept} 行"
          + (f" (丢弃 {dropped})" if dropped else ""))
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    丢弃原因 {reason}: {n}", file=sys.stderr)

    stats["rows"] += read_rows
    stats["kept"] += kept
    stats["dropped"] += dropped
    for reason, n in reasons.items():
        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + n
    return read_rows, kept


def main():
    ap = argparse.ArgumentParser(
        description="verl SFT parquet -> hyper-parallel data/omni 格式转换")
    ap.add_argument("--input", nargs="+", required=True,
                    help="parquet 文件或目录 (目录则递归收集 *.parquet), 可多个")
    ap.add_argument("--output-root", required=True,
                    help="输出根目录; jsonl 与图片目录都落在它下面")
    ap.add_argument("--format", choices=["jsonl", "parquet"], default="jsonl",
                    help="jsonl=抽盘图片+相对路径引用 (推荐); parquet=保留内嵌 bytes 直写")
    ap.add_argument("--images-dir-name", default="images",
                    help="jsonl 模式图片目录名 (相对 output-root, 默认 images)")
    ap.add_argument("--chunk-rows", type=int, default=128,
                    help="流式批大小; 整页大图数据建议调小 (默认 128)")
    ap.add_argument("--limit", type=int, default=None,
                    help="每个分片最多转换 N 行 (抽样调试用)")
    ap.add_argument("--validate-decode", action="store_true",
                    help="PIL 全解码校验 (需要 Pillow, 大数据较慢)")
    ap.add_argument("--overwrite", action="store_true",
                    help="覆盖已存在的输出文件")
    args = ap.parse_args()

    # 收集输入
    sources = []
    for p in args.input:
        if os.path.isfile(p):
            sources.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                sources.extend(
                    os.path.join(root, f) for f in sorted(files)
                    if f.endswith(".parquet"))
        else:
            ap.error(f"--input 不存在: {p}")
    sources = [s for s in sources if s.endswith(".parquet")]
    if not sources:
        ap.error("没有收集到任何 parquet 文件")

    if args.output_root.rstrip("/\\") in {os.path.abspath(os.path.dirname(s)).rstrip("/\\") for s in sources}:
        print("!! 警告: output-root 与输入目录相同, 源数据会被覆盖风险自负",
              file=sys.stderr)
    os.makedirs(args.output_root, exist_ok=True)
    images_dir = os.path.join(args.output_root, args.images_dir_name)
    if args.format == "jsonl":
        os.makedirs(images_dir, exist_ok=True)

    stats = {"rows": 0, "kept": 0, "dropped": 0, "reasons": {}}
    dedup_cache = {}
    for src in sorted(sources):
        print(f"=== {os.path.basename(src)} ===")
        convert_file(src, args, images_dir, dedup_cache, stats)

    print(f"\n完成: 读 {stats['rows']} 行, 保留 {stats['kept']}, "
          f"丢弃 {stats['dropped']}")
    for reason, n in sorted(stats["reasons"].items(), key=lambda kv: -kv[1]):
        print(f"  丢弃原因 {reason}: {n}")
    print(f"\nhyper-parallel data_path 即 <output-root> 下各 .jsonl/.parquet; "
          f"相对图片路径以数据文件所在目录锚定, 训练时请保持目录结构整体拷贝")


if __name__ == "__main__":
    main()
