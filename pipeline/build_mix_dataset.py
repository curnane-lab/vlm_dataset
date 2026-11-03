# -*- coding: utf-8 -*-
"""
build_mix_dataset.py — 以 tier2 采样副本为底, 混入 Infinity 含表整页子集,
构造合训数据集 (针对"整页漏表"问题的 SFT 混训)。

策略 (与评审结论对齐):
  - tier2 部分: 整体硬链接进输出根 (同盘零拷贝, 失败退化为 copy2)
  - Infinity 部分: 跨分片均布单遍扫描 (分片 stride 顺序, 无随机数),
    只选"整页 MD prompt + 答案含 MD 管道表"的整页行 (含表页面监督),
    另按配额保留正文/公式页维持转写能力
  - 选中行写出为新的 Infinity 子集分片; 答案后续由 MinerU2.5-Pro 蒸馏
    刷新 (Infinity 子集走 markdown 路由, 与 tier2 同一管线)

质检 (每行入选前): 占位符数==图片数, 答案非空且长度在 [--min-len,
--max-len], 图片字节 PNG/JPEG 魔数合法; 表格行必须命中 MD 管道表正则。

用法:
  python3 build_mix_dataset.py \
      --tier2-root F:/omnidoc_tier2_s20 \
      --data-root F:/omnidoc_vlm \
      --output-root F:/omnidoc_mix_s20 \
      --n-table 4000 --n-plain 1000
产物: 输出根含合并 manifest (tier2 五数据集 + Infinity 子集), 之后可直接:
  distill_qa.py 刷新答案 / qc_distilled.py / qa_compat_check.py / 训练。

依赖: pyarrow。
"""

import argparse
import json
import os
import re
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

INFINITY = "Infinity-Doc-400K_parsing_vlm"
# 整页 MD prompt 池 (vlm_maps.PROMPTS_MD) 均含 markdown 字样;
# 表格 crop 池 (PROMPTS_TABLE_HTML) 含表格/table 字样, 借此区分
MD_PROMPT_KEY = "markdown"
# MD 管道表: "| ... |" 行 + 次日 |---| 分隔行
MD_TABLE_RE = re.compile(r"\n\|[^\n]*\|\n\|[\s:|-]+\|")
FLUSH_EVERY = 500  # 每收满 500 行落一次盘, 控制内存


def is_page_md_prompt(q):
    return MD_PROMPT_KEY in q.replace("<image>", "").lower()


def last_qa(row):
    msgs = row["messages"] or []
    q = next((m["content"] for m in reversed(msgs) if m["role"] == "user"),
             "")
    a = next((m["content"] for m in reversed(msgs)
              if m["role"] == "assistant"), "")
    return q, a


def row_ok(row, min_len, max_len):
    """入选校验。返回 (是否可用, "table"/"plain"/丢弃原因)。"""
    msgs = row["messages"] or []
    imgs = row["images"] or []
    q, a = last_qa(row)
    n_ph = sum(m["content"].count("<image>") for m in msgs
               if m["role"] == "user")
    if n_ph != len(imgs) or not imgs:
        return False, "placeholder_img_mismatch"
    if not a.strip() or not (min_len <= len(a) <= max_len):
        return False, "len_or_empty"
    if not is_page_md_prompt(q):
        return False, "not_page_prompt"
    for im in imgs:
        b = im["bytes"]
        if not b or len(b) < 8:
            return False, "bad_img"
        if not (b[:4] == b"\x89PNG" or b[:3] == b"\xff\xd8\xff"):
            return False, "bad_img"
    return True, "table" if MD_TABLE_RE.search(a) else "plain"


def link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description="构造 tier2+Infinity 合训数据集")
    ap.add_argument("--tier2-root", required=True)
    ap.add_argument("--data-root", required=True, help="原始数据集根目录")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--n-table", type=int, default=4000,
                    help="含表整页目标数 (默认 4000)")
    ap.add_argument("--n-plain", type=int, default=1000,
                    help="正文/公式整页目标数 (默认 1000, 维持转写能力)")
    ap.add_argument("--min-len", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=20000)
    args = ap.parse_args()

    for bad in (args.tier2_root, args.data_root):
        if os.path.abspath(args.output_root) == os.path.abspath(bad):
            ap.error("--output-root 不能与输入根相同")

    with open(os.path.join(args.tier2_root, "dataset_manifest.json"),
              encoding="utf-8") as f:
        t2 = json.load(f)
    with open(os.path.join(args.data_root, "dataset_manifest.json"),
              encoding="utf-8") as f:
        src_mani = json.load(f)

    os.makedirs(args.output_root, exist_ok=True)
    out_mani = {k: v for k, v in t2.items() if k != "datasets"}
    out_mani["data_root"] = os.path.abspath(args.output_root)
    out_mani["datasets"] = []

    # ---- 1) tier2 硬链接进输出根 ----
    for ds in t2["datasets"]:
        for finfo in ds["files"]:
            rel = finfo["path"].replace("\\", "/")
            link_or_copy(os.path.join(args.tier2_root, rel),
                         os.path.join(args.output_root, rel))
        out_mani["datasets"].append(ds)
        print(f"[tier2] {ds['name']}: {ds['num_files']} 分片 "
              f"{ds['total_rows']:,} 行 已链接", flush=True)

    # ---- 2) Infinity 单遍流式选取 (分片 stride 均布, 确定性) ----
    inf = next(d for d in src_mani["datasets"] if d["name"] == INFINITY)
    files = [f for f in inf["files"]
             if os.path.exists(os.path.join(
                 args.data_root, f["path"].replace("\\", "/")))]
    n_shards = len(files)
    stride = max(1, n_shards // 200)

    rel_out = f"{INFINITY}/vlm_parquet/train-00000-of-00001.parquet"
    dst = os.path.join(args.output_root, rel_out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src0 = pq.ParquetFile(os.path.join(
        args.data_root, files[0]["path"].replace("\\", "/")))
    schema = src0.schema_arrow

    n_table = n_plain = scanned = 0
    drop = {}
    pending = []
    done = False
    with pq.ParquetWriter(dst, schema=schema) as writer:
        for start in range(stride):
            if done:
                break
            for si in range(start, n_shards, stride):
                rel = files[si]["path"].replace("\\", "/")
                pf = pq.ParquetFile(os.path.join(args.data_root, rel))
                for batch in pf.iter_batches(batch_size=500):
                    for row in batch.to_pylist():
                        scanned += 1
                        ok, why = row_ok(row, args.min_len, args.max_len)
                        if not ok:
                            drop[why] = drop.get(why, 0) + 1
                            continue
                        if why == "table":
                            if n_table >= args.n_table:
                                continue
                            n_table += 1
                        else:
                            if n_plain >= args.n_plain:
                                continue
                            n_plain += 1
                        pending.append({"messages": row["messages"],
                                        "images": row["images"]})
                        if len(pending) >= FLUSH_EVERY:
                            writer.write_table(pa.Table.from_pylist(
                                pending, schema=schema))
                            pending.clear()
                print(f"[scan] 分片 {si + 1}/{n_shards}: 含表 "
                      f"{n_table}/{args.n_table}, 正文 "
                      f"{n_plain}/{args.n_plain}", flush=True)
                if n_table >= args.n_table and n_plain >= args.n_plain:
                    done = True
                    break
        if pending:
            writer.write_table(pa.Table.from_pylist(pending,
                                                    schema=schema))
            pending.clear()

    total = n_table + n_plain
    print(f"[infinity] 扫描 {scanned:,} 行, 选中 {total} "
          f"(含表 {n_table}, 正文/公式 {n_plain}); 过滤 {drop}")
    if n_table < args.n_table:
        print(f"[warn] 含表整页不足配额 ({n_table} < {args.n_table}), "
              f"按实际数量写入", file=sys.stderr)

    out_mani["datasets"].append({
        "name": INFINITY, "schema_adapter": "qwen_vl", "weight": 1.0,
        "total_rows": total, "num_files": 1,
        "files": [{"path": rel_out, "rows": total}]})
    mpath = os.path.join(args.output_root, "dataset_manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(out_mani, f, ensure_ascii=False, indent=1)
    print(f"\n完成: 合训根 {args.output_root}")
    print(f"manifest: {mpath}")
    print(f"总计: {len(out_mani['datasets'])} 数据集, "
          f"{sum(d['total_rows'] for d in out_mani['datasets']):,} 行")
    print("后续: distill_qa.py 刷新答案 (Infinity 子集走 markdown 路由) "
          "-> qc_distilled.py -> check_qa_compat.sh -> 训练")


if __name__ == "__main__":
    main()
