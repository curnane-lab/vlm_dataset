# -*- coding: utf-8 -*-
"""
report_table_density.py — 数据集整页"密集表格"覆盖度统计

对指定数据集 (默认整页类) 统计其答案中的表格监督强度:
  - 含表页占比 / 多表页占比
  - 表格行数、列数分布 (p50/p90/max), 长表(>=20行)/宽表(>=10列)占比
  - 合并单元格率 (HTML colspan/rowspan; MD 管道表天然没有, 会标 0)
同时支持两种表格形态: HTML <table> 与 Markdown 管道表。

用途: 检验"整页密集表格"覆盖情况; 可对蒸馏前(原生 GT)与蒸馏后(标签)
各跑一次, 判断结构监督是否被刷新管线抹平。

用法:
  python3 report_table_density.py --data-root /data/omnidoc_mix_s20 \
      --datasets Infinity-Doc-400K_parsing_vlm
  # hyper-parallel jsonl (images 为路径) 也支持:
  python3 report_table_density.py --jsonl /data/vl_train_121k.jsonl
依赖: pyarrow (仅 parquet 模式)
"""

import argparse
import json
import os
import re
import sys

import pyarrow.parquet as pq

MD_TABLE_RE = re.compile(r"\n\|[^\n]*\|\n\|[\s:|-]+\|")


def parse_md_tables(a):
    lines = a.split("\n")
    tabs = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and i + 1 < len(lines) \
                and re.match(r"^\|[\s:|-]+\|?\s*$",
                             lines[i + 1].strip() or " "):
            ncol = lines[i].count("|") - 1
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            tabs.append((j - i - 1, ncol))
            i = j
        else:
            i += 1
    return tabs


def parse_html_tables(a):
    """返回 [(行数<tr>, 列数(首行<td>), 含colspan, 含rowspan)]"""
    out = []
    for tb in re.findall(r"<table.*?</table>", a, re.S):
        nrow = len(re.findall(r"<tr[\s>]", tb))
        m0 = re.search(r"<tr[\s>].*?</tr>", tb, re.S)
        ncol = len(re.findall(r"<t[dh][\s>]", m0.group(0))) if m0 else 0
        out.append((nrow, ncol,
                    bool(re.search(r"colspan\s*=", tb)),
                    bool(re.search(r"rowspan\s*=", tb))))
    return out


class Stats:
    def __init__(self):
        self.rows = self.page_with_table = self.multi_table = 0
        self.nrow, self.ncol = [], []
        self.n_colspan = self.n_rowspan = 0
        self.pages_with_merge = 0
        self.alens = []

    def add(self, a):
        self.rows += 1
        self.alens.append(len(a))
        tabs = []
        for nr, nc in parse_md_tables(a):
            tabs.append((nr, nc, False, False))
        tabs.extend(parse_html_tables(a))
        if not tabs:
            return
        self.page_with_table += 1
        if len(tabs) >= 2:
            self.multi_table += 1
        merge = False
        for nr, nc, cs, rs in tabs:
            self.nrow.append(nr)
            self.ncol.append(nc)
            self.n_colspan += int(cs)
            self.n_rowspan += int(rs)
            merge = merge or cs or rs
        if merge:
            self.pages_with_merge += 1

    def report(self, title):
        def pct(v, q):
            v = sorted(v)
            return v[min(len(v) - 1, int(q * len(v)))] if v else 0
        n_tab = len(self.nrow)
        print(f"\n== {title} ==")
        print(f"行数: {self.rows:,}; 含表页: {self.page_with_table:,} "
              f"({self.page_with_table / max(self.rows, 1):.1%}); "
              f"多表页: {self.multi_table:,} "
              f"({self.multi_table / max(self.page_with_table, 1):.1%})")
        if not n_tab:
            print("无表格样本")
            return
        print(f"表格总数: {n_tab:,}")
        print(f"  行数/表: p50={pct(self.nrow, .5)} p90={pct(self.nrow, .9)} "
              f"max={max(self.nrow)}; >=20行 {sum(1 for x in self.nrow if x >= 20) / n_tab:.1%}, "
              f">=50行 {sum(1 for x in self.nrow if x >= 50) / n_tab:.1%}")
        print(f"  列数/表: p50={pct(self.ncol, .5)} p90={pct(self.ncol, .9)} "
              f"max={max(self.ncol)}; >=10列 {sum(1 for x in self.ncol if x >= 10) / n_tab:.1%}")
        print(f"  合并单元格: colspan {self.n_colspan / n_tab:.1%}, "
              f"rowspan {self.n_rowspan / n_tab:.1%}; "
              f"含合并单元格页 {self.pages_with_merge:,} "
              f"({self.pages_with_merge / max(self.page_with_table, 1):.1%} 的含表页)")
        print(f"  答案长度: p50={pct(self.alens, .5)} "
              f"p90={pct(self.alens, .9)} max={max(self.alens):,}")


def main():
    ap = argparse.ArgumentParser(description="整页密集表格覆盖度统计")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--jsonl", default=None,
                    help="hyper-parallel 训练 jsonl (messages+images 路径)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="每分片最多读 N 行 (快速抽查)")
    args = ap.parse_args()
    if not args.data_root and not args.jsonl:
        ap.error("需要 --data-root 或 --jsonl")

    st = Stats()
    if args.jsonl:
        with open(args.jsonl, encoding="utf-8") as f:
            n = 0
            for line in f:
                d = json.loads(line)
                msgs = d["messages"]
                q = next((m["content"] for m in msgs if m["role"] == "user"),
                         "")
                if "markdown" not in q.replace("<image>", "").lower():
                    continue  # 只统计整页 MD 任务行
                a = next((m["content"] for m in reversed(msgs)
                          if m["role"] == "assistant"), "")
                st.add(a)
                n += 1
                if args.max_rows and n >= args.max_rows:
                    break
        st.report(os.path.basename(args.jsonl) + " (整页 MD 行)")
        return

    manifest_path = args.manifest or os.path.join(
        args.data_root, "dataset_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    for ds in manifest["datasets"]:
        name = ds["name"]
        if args.datasets and name not in args.datasets:
            continue
        for finfo in ds.get("files") or []:
            rel = finfo["path"].replace("\\", "/")
            p = os.path.join(args.data_root, rel)
            if not os.path.exists(p):
                print(f"  [skip] missing {p}", file=sys.stderr)
                continue
            pf = pq.ParquetFile(p)
            seen = 0
            for batch in pf.iter_batches(batch_size=1000,
                                         columns=["messages"]):
                for r in batch.to_pylist():
                    a = next((m["content"] for m in reversed(r["messages"])
                              if m["role"] == "assistant"), "")
                    st.add(a)
                seen += batch.num_rows
                if args.max_rows and seen >= args.max_rows:
                    break
        print(f"[scan] {name} 完成, 累计 {st.rows:,} 行", flush=True)
    st.report(os.path.basename(args.data_root))


if __name__ == "__main__":
    main()
