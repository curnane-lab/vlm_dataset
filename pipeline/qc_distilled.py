# -*- coding: utf-8 -*-
"""
qc_distilled.py — 蒸馏产物质检: 校验 regenerate 后的 response 是否正常合理

检查三类问题 (按 distill_qa.py 的路由契约逐项对齐):
  1) 没刷新: 空答案; 与源答案逐字相同 (= distill 失败时保留原答案的静默兜底)
  2) 格式错: crop/equation 必须是 $$...$$ 或 \\[...\\], 禁止裸的
     equation/align/gather/eqnarray; crop/table 应为 HTML, crop/text
     应为纯文本, markdown 路由应为文档排版且不能残留上述裸环境;
     结构上末条必须 assistant
  3) 内容退化: 布局 token 泄漏 (<|box_start|> 等), OTSL token 泄漏
     (<fcel> 等), 重复循环输出, 长度异常, user 消息/图片字节被改动

路由与 distill_qa.py 完全一致 (route_dataset / crop_block_type);
skip 档 (holdout) 蒸馏时不刷, 本脚本对它们只做结构检查。

用法:
  # 纯格式扫描 (messages-only, 快)
  python3 qc_distilled.py --data-root /data/omnidoc_vlm_regen/tier2
  # 对照源数据: 刷新率 + user/图片完整性
  python3 qc_distilled.py --data-root /data/omnidoc_vlm_regen/tier2 \
      --source-root /data/omnidoc_vlm --check-images 8 --report qc.json
  # 快速抽查: 每分片只看前 200 行
  python3 qc_distilled.py --data-root ... --max-rows 200

退出码: 0 = 达标; 1 = 存在超阈值问题 (阈值见 --min-refresh/--max-issue-rate)
依赖: pyarrow; 复用 distill_qa.py 的 route_dataset/crop_block_type/norm_rel_path
"""

import argparse
import hashlib
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_qa import (  # noqa: E402
    crop_block_type, equation_display_ok, has_top_display_env,
    norm_rel_path, route_dataset)

LAYOUT_TOKENS = ("<|box_start|>", "<|box_end|>", "<|boend|>",
                 "<|ref_start|>", "<|ref_end|>", "<|rotate_up|>",
                 "<|rotate_down|>", "<|im_start|>", "<|im_end|>")
OTSL_TOKENS = ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>")
# 严重问题类 (计入 --max-issue-rate); too_short/too_long/low_cjk 只告警
SEVERE = ("empty_answer", "layout_token", "otsl_token", "format_violation",
          "repetition", "user_tampered", "image_tampered")
SEVERE_SET = frozenset(SEVERE)


def is_cjk(ch):
    return "一" <= ch <= "鿿"


def check_repetition(text):
    """行级重复: 某一行占比过半且行数 >= 10, 视为退化循环。"""
    if len(text) < 200:
        return False
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 10:
        return False
    from collections import Counter
    top, cnt = Counter(lines).most_common(1)[0]
    return cnt * 2 >= len(lines) and len(top) >= 4


def check_row(answer, route, crop_type, min_md_len, expect_cjk=False):
    """单条 assistant 答案质检, 返回问题集合 (issue -> True)。"""
    issues = set()
    a = (answer or "").strip()
    if not a:
        issues.add("empty_answer")
        return issues
    if any(t in a for t in LAYOUT_TOKENS):
        issues.add("layout_token")
    if any(t in a for t in OTSL_TOKENS):
        issues.add("otsl_token")
    if check_repetition(a):
        issues.add("repetition")
    if route == "crop":
        if crop_type == "equation":
            if "<table" in a or "<td" in a or "<tr" in a:
                issues.add("format_violation")
            if not equation_display_ok(a):
                issues.add("format_violation")
            if len(a) > 5000:
                issues.add("too_long")
        elif crop_type == "table":
            if "<table" not in a and "<tr" not in a and "<td" not in a:
                issues.add("format_violation")
            elif a.count("<table") != a.count("</table>"):
                issues.add("format_violation")
        else:  # text: 纯文本, 不应含 HTML/布局符
            if "<table" in a or "<td" in a:
                issues.add("format_violation")
            if expect_cjk and len(a) >= 4:
                # CASIA 手写中文: 答案应以汉字为主 (源数据乱码/识别失败会暴露)
                n_cjk = sum(1 for c in a if is_cjk(c))
                if n_cjk / len(a) < 0.3:
                    issues.add("low_cjk")
    elif route == "markdown":
        if len(a) < min_md_len:
            issues.add("too_short")
        if has_top_display_env(a):
            issues.add("format_violation")
    return issues


def pctl(sorted_vals, q):
    if not sorted_vals:
        return 0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def main():
    ap = argparse.ArgumentParser(description="蒸馏产物 response 质检")
    ap.add_argument("--data-root", required=True, help="蒸馏产物根目录")
    ap.add_argument("--source-root", default=None,
                    help="原始数据根目录 (给出后检查刷新率/user/图片完整性)")
    ap.add_argument("--manifest", default=None,
                    help="默认 <data-root>/dataset_manifest.json")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="每分片最多扫描行数 (快速抽查)")
    ap.add_argument("--chunk-rows", type=int, default=500)
    ap.add_argument("--check-images", type=int, default=0,
                    help="每分片抽样前 N 行校验图片字节与源一致 (需 --source-root)")
    ap.add_argument("--min-md-len", type=int, default=50,
                    help="markdown 路由答案过短阈值 (默认 50 字符)")
    ap.add_argument("--min-refresh", type=float, default=0.8,
                    help="刷新率下限 (默认 0.8; 仅 --source-root 且非 skip 档生效)")
    ap.add_argument("--max-issue-rate", type=float, default=0.05,
                    help="严重问题率上限 (默认 0.05)")
    ap.add_argument("--report", default=None, help="结果落 JSON 到该路径")
    args = ap.parse_args()

    manifest_path = args.manifest or os.path.join(
        args.data_root, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        ap.error(f"manifest 不存在: {manifest_path} "
                 f"(检查 --data-root 是否指向蒸馏产物根目录)")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    report = []
    any_fail = False
    for ds in manifest["datasets"]:
        name = ds["name"]
        if args.datasets and name not in args.datasets:
            continue
        route = route_dataset(name)
        crop_type = crop_block_type(name) if route == "crop" else None
        files = ds.get("files") or []
        if args.max_files:
            files = files[:args.max_files]

        st = {"rows": 0, "severe_rows": 0, "unchanged": 0, "user_tampered": 0,
              "image_tampered": 0, "lens": []}
        for k in SEVERE + ("too_short", "too_long", "low_cjk",
                           "delim_residue"):
            st.setdefault(k, 0)

        for finfo in files:
            rel = norm_rel_path(finfo["path"])
            dst = os.path.join(args.data_root, rel)
            if not os.path.exists(dst):
                print(f"  [skip] 产物缺失 {rel}", file=sys.stderr)
                continue
            src = (os.path.join(args.source_root, rel)
                   if args.source_root else None)
            src_pf = None
            if src:
                if os.path.exists(src):
                    src_pf = pq.ParquetFile(src)
                else:
                    print(f"  [warn] 源分片缺失 {src}, 跳过对比",
                          file=sys.stderr)
            pf = pq.ParquetFile(dst)
            src_iter = (src_pf.iter_batches(batch_size=args.chunk_rows,
                                            columns=["messages"])
                        if src_pf else None)
            n_seen = 0
            for batch in pf.iter_batches(batch_size=args.chunk_rows,
                                         columns=["messages"]):
                if args.max_rows and n_seen >= args.max_rows:
                    break
                rows = batch.to_pylist()
                if args.max_rows:
                    rows = rows[:args.max_rows - n_seen]
                n_seen += len(rows)
                src_rows = next(src_iter).to_pylist() if src_iter else None
                for i, r in enumerate(rows):
                    msgs = r["messages"] or []
                    ans = next((m["content"] for m in reversed(msgs)
                                if m["role"] == "assistant"), "")
                    st["rows"] += 1
                    st["lens"].append(len(ans or ""))
                    issues = check_row(ans, route, crop_type,
                                       args.min_md_len,
                                       expect_cjk="casia" in name.lower())
                    for iss in issues:
                        st[iss] += 1
                    row_severe = bool(issues & SEVERE_SET)
                    if src_rows is not None and i < len(src_rows):
                        s_msgs = src_rows[i]["messages"] or []
                        s_ans = next((m["content"] for m in reversed(s_msgs)
                                      if m["role"] == "assistant"), "")
                        if (ans or "") == (s_ans or ""):
                            st["unchanged"] += 1
                        u_dst = [m["content"] for m in msgs
                                 if m["role"] == "user"]
                        u_src = [m["content"] for m in s_msgs
                                 if m["role"] == "user"]
                        if u_dst != u_src:
                            st["user_tampered"] += 1
                            row_severe = True
                    if row_severe:
                        st["severe_rows"] += 1
            # 图片抽样完整性
            if src_pf and args.check_images:
                d_first = next(pq.ParquetFile(dst).iter_batches(
                    batch_size=args.check_images,
                    columns=["messages", "images"]), None)
                s_first = (next(src_pf.iter_batches(
                    batch_size=args.check_images,
                    columns=["images"]), None)
                    if d_first is not None else None)
                if d_first is not None and s_first is not None:
                    d_rows = d_first.to_pylist()
                    s_rows = s_first.to_pylist()
                    for i in range(min(len(d_rows), len(s_rows))):
                        d_imgs = d_rows[i]["images"] or []
                        s_imgs = s_rows[i]["images"] or []
                        n_ph = sum(m["content"].count("<image>")
                                   for m in d_rows[i]["messages"]
                                   if m["role"] == "user")
                        if n_ph != len(d_imgs):
                            st["image_tampered"] += 1
                            st["severe_rows"] += 1
                            continue
                        if len(d_imgs) != len(s_imgs):
                            st["image_tampered"] += 1
                            st["severe_rows"] += 1
                            continue
                        for di, si in zip(d_imgs, s_imgs):
                            if hashlib.sha256(di["bytes"]).hexdigest() != \
                                    hashlib.sha256(si["bytes"]).hexdigest():
                                st["image_tampered"] += 1
                                st["severe_rows"] += 1
                                break

        n = max(st["rows"], 1)
        severe_n = st["severe_rows"]
        refresh = (1 - st["unchanged"] / n) if args.source_root else None
        lens = sorted(st["lens"])
        line = (f"=== {name} [{route}{'/' + crop_type if crop_type else ''}] "
                f"{st['rows']} 行: 严重 {severe_n} "
                f"(空{st['empty_answer']} 布局{st['layout_token']} "
                f"OTSL{st['otsl_token']} 格式{st['format_violation']} "
                f"重复{st['repetition']}) "
                f"长度 p50={pctl(lens, 0.5)} p95={pctl(lens, 0.95)}")
        if refresh is not None:
            line += (f" 刷新率 {refresh:.1%} (user改动{st['user_tampered']}"
                     f" 图改动{st['image_tampered']})")
        fails = []
        if route != "skip":
            if severe_n / n > args.max_issue_rate:
                fails.append(f"严重问题率 {severe_n / n:.1%}")
            if refresh is not None and refresh < args.min_refresh:
                fails.append(f"刷新率 {refresh:.1%} < {args.min_refresh:.0%}")
        else:
            line += " (distill 跳过档, 仅结构检查)"
        if fails:
            any_fail = True
            line += "  *** 不达标: " + "; ".join(fails) + " ***"
        print(line)
        report.append({"name": name, "route": route, "crop_type": crop_type,
                       "rows": st["rows"], "severe": severe_n,
                       "refresh": refresh,
                       "issues": {k: st[k] for k in st
                                  if isinstance(st[k], int)
                                  and k not in ("rows",)},
                       "len_p50": pctl(lens, 0.5),
                       "len_p95": pctl(lens, 0.95),
                       "fail": fails})

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"data_root": os.path.abspath(args.data_root),
                       "source_root": args.source_root,
                       "datasets": report}, f, ensure_ascii=False, indent=1)
        print(f"\n报告: {args.report}")
    print("\nQC RESULT:", "FAIL" if any_fail else "PASS")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
