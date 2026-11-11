# -*- coding: utf-8 -*-
"""
convert_hyperparallel_to_verl.py — hyper-parallel data/omni 格式 -> verl SFT parquet

用途: convert_verl_to_hyperparallel.py 的反向转换。把 hyper-parallel
hyper_parallel.data.omni 的训练数据 (JSONL 行内 {"messages": [...],
"images": ["images/<sha>.png", ...]}, 或其 parquet) 转回本仓统一
qwen_vl schema (verl.trainer.sft_trainer 直读的
messages: list<struct{role,content}> + images: list<struct{bytes,path}>)。

图片来源解析 (对齐 hyper-parallel 以数据文件所在目录锚定相对路径的语义):
  - jsonl: images 元素是 str 相对/绝对路径, 相对路径以 jsonl 文件所在目录解析
  - parquet: images 元素是 {path} (同上解析) 或 {bytes} (直接透传)

契约校验 (违约行丢弃并计数):
  - messages 非空 list, >=2 条且末条 role=assistant, content 为非空 str
  - "<image>" 占位符总数 == images 数
  - 图片 bytes 按 magic 校验 (PNG/JPEG/WEBP/GIF/BMP), 可选 PIL 全解码
  - jsonl 引用的图片文件不存在 -> 丢弃该行

依赖: pyarrow; --validate-decode 时额外需要 Pillow。

示例:
  python3 convert_hyperparallel_to_verl.py \
    --input /data/.../hyperparallel_omni \
    --output-root /data/.../verl_sft_parquet

verl 侧: 输出即本仓统一 schema, 直接作为 data.train_files 喂
verl.trainer.sft_trainer (data.messages_key=messages, image_key=images 默认)。
"""

import argparse
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

# 本仓统一 qwen_vl schema (与 README 数据契约一致)
VERL_SCHEMA = pa.schema([
    ("messages", pa.list_(pa.struct([
        ("role", pa.string()),
        ("content", pa.string()),
    ]))),
    ("images", pa.list_(pa.struct([
        ("bytes", pa.binary()),
        ("path", pa.string()),
    ]))),
])

MAGIC_EXTS = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
)


def image_ok(data):
    """按 magic 校验图片格式。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:3] == b"\xff\xd8\xff":
        return True
    if data[:4] in (b"GIF87a", b"GIF89a"):
        return True
    if data[:2] == b"BM":
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


def count_image_placeholders(messages):
    """统计所有 str content 里的 <image> 占位符 (对齐 verl re.split 语义)。"""
    return sum(
        m.get("content", "").count("<image>")
        for m in messages
        if isinstance(m, dict) and isinstance(m.get("content"), str)
    )


def resolve_image(entry, base_dir):
    """解析一个 images 元素, 返回 (bytes, path) 或 (None, None, 原因)。"""
    if isinstance(entry, str):
        path, inline = entry, None
    elif isinstance(entry, dict):
        path, inline = entry.get("path"), entry.get("bytes")
    else:
        return None, None, "image_entry_type"

    if inline is not None:
        if not isinstance(inline, (bytes, bytearray)) or not inline:
            return None, None, "image_bytes_missing"
        return bytes(inline), path or "", None

    if not isinstance(path, str) or not path:
        return None, None, "image_path_missing"
    full = path if os.path.isabs(path) else os.path.join(base_dir, path)
    try:
        with open(full, "rb") as f:
            data = f.read()
    except OSError:
        return None, None, "image_file_missing"
    if not data:
        return None, None, "image_bytes_missing"
    return data, path, None


def validate_row(row, base_dir):
    """校验一行并解析图片 bytes。返回 (messages, images, None) 或 (None, None, 原因)。"""
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

    raw_images = row.get("images") or []
    if count_image_placeholders(messages) != len(raw_images):
        return None, None, "placeholder_images_mismatch"

    images = []
    for entry in raw_images:
        data, path, reason = resolve_image(entry, base_dir)
        if reason is not None:
            return None, None, reason
        if not image_ok(data):
            return None, None, "image_format_unknown"
        images.append({"bytes": data, "path": path})
    return messages, images, None


def maybe_decode_check(images):
    """--validate-decode: PIL 全解码并转 RGB, 失败抛异常。"""
    from io import BytesIO

    from PIL import Image

    for img in images:
        with Image.open(BytesIO(img["bytes"])) as im:
            im.convert("RGB").load()


def iter_rows(src, batch_rows=128):
    """按扩展名产出 (row, base_dir)。jsonl 逐行, parquet 流式限批防大图 OOM。"""
    base_dir = os.path.dirname(os.path.abspath(src))
    if src.endswith(".jsonl"):
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line), base_dir
    elif src.endswith(".parquet"):
        pf = pq.ParquetFile(src)
        for batch in pf.iter_batches(batch_size=batch_rows):
            for row in batch.to_pylist():
                yield row, base_dir
    else:
        raise ValueError(f"不支持的输入类型: {src}")


def convert_file(src, args, stats):
    """转换一个输入文件, 返回 (读行数, 保留行数)。"""
    stem = os.path.basename(src)
    for suf in (".jsonl", ".parquet"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    out_path = os.path.join(args.output_root, stem + ".parquet")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"  [skip] 已存在 {out_path} (--overwrite 覆盖)")
        return 0, 0

    read_rows = kept = dropped = 0
    reasons = {}
    buffer = []
    writer = pq.ParquetWriter(out_path, VERL_SCHEMA)
    try:
        for row, base_dir in iter_rows(src, args.chunk_rows):
            read_rows += 1
            messages, images, reason = validate_row(row, base_dir)
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
            buffer.append({"messages": messages, "images": images})
            kept += 1
            if len(buffer) >= args.chunk_rows:
                writer.write_table(pa.Table.from_pylist(buffer, schema=VERL_SCHEMA))
                buffer.clear()
            if args.limit and kept >= args.limit:
                break
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=VERL_SCHEMA))
    finally:
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
        description="hyper-parallel data/omni 格式 -> verl SFT parquet 转换")
    ap.add_argument("--input", nargs="+", required=True,
                    help="hyper-parallel 的 .jsonl/.parquet 文件或目录 (目录递归收集), 可多个")
    ap.add_argument("--output-root", required=True,
                    help="输出根目录, 每个输入文件对应一个同名 .parquet")
    ap.add_argument("--chunk-rows", type=int, default=128,
                    help="parquet 写入批大小 (默认 128)")
    ap.add_argument("--limit", type=int, default=None,
                    help="每个输入文件最多转换 N 行 (抽样调试用)")
    ap.add_argument("--validate-decode", action="store_true",
                    help="PIL 全解码校验 (需要 Pillow, 大数据较慢)")
    ap.add_argument("--overwrite", action="store_true",
                    help="覆盖已存在的输出文件")
    args = ap.parse_args()

    sources = []
    for p in args.input:
        if os.path.isfile(p):
            sources.append(p)
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                sources.extend(
                    os.path.join(root, f) for f in sorted(files)
                    if f.endswith((".jsonl", ".parquet")))
        else:
            ap.error(f"--input 不存在: {p}")
    sources = [s for s in sources if s.endswith((".jsonl", ".parquet"))]
    if not sources:
        ap.error("没有收集到任何 .jsonl/.parquet 文件")

    if args.output_root.rstrip("/\\") in {
            os.path.abspath(os.path.dirname(s)).rstrip("/\\") for s in sources}:
        print("!! 警告: output-root 与输入目录相同, 源数据有被覆盖风险",
              file=sys.stderr)
    os.makedirs(args.output_root, exist_ok=True)

    stats = {"rows": 0, "kept": 0, "dropped": 0, "reasons": {}}
    for src in sorted(sources):
        print(f"=== {os.path.basename(src)} ===")
        convert_file(src, args, stats)

    print(f"\n完成: 读 {stats['rows']} 行, 保留 {stats['kept']}, "
          f"丢弃 {stats['dropped']}")
    for reason, n in sorted(stats["reasons"].items(), key=lambda kv: -kv[1]):
        print(f"  丢弃原因 {reason}: {n}")
    print("\n输出即本仓统一 qwen_vl schema, 直接作为 verl data.train_files 使用")


if __name__ == "__main__":
    main()
