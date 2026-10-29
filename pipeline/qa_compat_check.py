# -*- coding: utf-8 -*-
"""
qa_compat_check.py — 蒸馏产物 Q-A 匹配质检 (通用 VLM 裁判, OpenAI 兼容服务)

动机: 蒸馏只刷新答案 A, prompt Q 不动; 两者可能脱节 (如 Q 说"纯文本"
而 A 是 markdown; 或裁判认为 A 与图片内容不符/臆造)。本脚本用一个通用
VLM (如 Qwen2.5-VL/Qwen3-VL) 看图判定每对 (Q, A) 是否匹配, 输出统计。

抽样: 确定性跨分片均布抽样 — 先用 messages-only 快扫定位行号, 再按
row-group 粒度只取目标行 (避免为抽 200 行全量读图); 同参数重跑选中行
完全一致, 无随机数。

裁判输入: 图片 (原 bytes 直转 data_url, 零转码) + Q + A (A 过长截首尾);
裁判输出一行 JSON: {"match": bool, "issues": [...], "confidence": 0..1}
空 A 不送裁判, 直接判 mismatch (empty_answer)。

用法:
  # 只统计 (默认)
  python3 qa_compat_check.py --data-root /data/omnidoc_vlm_regen/tier2 \
      --model qwen-vl --num-instances 1 --base-port 31000 \
      --per-dataset 200 --workers 16 --report qa_compat.json
  # 修复模式: 全量逐行裁判, 不匹配行由裁判改写 Q 对齐 A
  # (A 为 MinerU 蒸馏产物, 只读不动), parquet 原子写回, .qafix_done 断点续跑
  python3 qa_compat_check.py --data-root ... --fix --resume --workers 16
通常经 check_qa_compat.sh 一键拉起服务+检查 (FIX=1 开修复)。

退出码: 统计模式 0 = 全部数据集达标; 1 = 有数据集低于 --min-match-rate。
        修复模式 0 = 全部修复成功; 1 = 存在修复失败/裁判失败率过高。
依赖: pyarrow, requests; 复用 distill_qa.py 的 TeacherPool /
build_api_messages / norm_rel_path
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from distill_qa import (  # noqa: E402
    TeacherPool, build_api_messages, norm_rel_path)

JUDGE_TMPL = """# 任务
判断"答案 A"是否恰当地回答了"问题 Q"。图片是原始输入, Q 和 A 是针对它的问答。

# 问题 Q
{q}

# 答案 A
{a}

# 判定标准
1. 相关性: A 针对 Q 提出的任务作答, 内容与图片实际内容相符, 不臆造
2. 格式一致性: 若 Q 指定输出格式 (如"纯文本"/"HTML"/"LaTeX"/"Markdown"),
   A 必须符合; Q 未指定格式时, 不因 A 的格式选择判负
3. 完整性: A 非空、未截断、无重复退化

# 输出
只输出一行 JSON, 不要输出任何其他内容:
{{"match": true, "issues": [], "confidence": 0.95}}"""

VERDICT_RE = re.compile(r"\{.*\}", re.DOTALL)
MAX_IMAGES_PER_ROW = 4  # 裁判每行最多看图数, 控制请求体积

REWRITE_TMPL = """# 任务
"答案 A"是正确的标准答案, 但"原问题 Q"与 A 不匹配 (原因: {issues})。
请结合图片改写 Q, 使其成为能自然引出 A 的提问/指令。
要求: 保持 Q 的原意与语言; 修正与 A 冲突的要求 (如输出格式约定);
不得引入 A 无法满足的新要求; 不要包含 "<image>" 标记; 只输出改写后的 Q 文本。

# 原问题 Q
{q}

# 答案 A
{a}"""


def parse_verdict(text):
    """从裁判输出提取 JSON verdict; 失败返回 None。"""
    m = VERDICT_RE.search(text or "")
    if not m:
        return None
    try:
        v = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    match = v.get("match")
    if isinstance(match, str):
        match = match.strip().lower() in ("true", "yes", "1")
    if not isinstance(match, bool):
        return None
    issues = v.get("issues")
    if not isinstance(issues, list):
        issues = [str(issues)] if issues else []
    return {"match": match, "issues": [str(i) for i in issues],
            "confidence": v.get("confidence")}


def plan_sample(files, root, per_dataset):
    """确定性均布抽样计划: {rel: [行号...]} (按 footer 实际行数)。"""
    shards = []
    for finfo in files:
        rel = norm_rel_path(finfo["path"])
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            print(f"  [skip] missing {p}", file=sys.stderr)
            continue
        shards.append((rel, p, pq.read_metadata(p).num_rows))
    total = sum(n for _, _, n in shards)
    if total == 0:
        return {}, 0
    n_target = min(per_dataset, total)
    # 全局均匀取 n_target 个目标行号 (等距, 含首行)
    targets = sorted({min(total - 1, round(j * total / n_target))
                      for j in range(n_target)})
    plan = {}
    gi = 0
    ti = 0
    for rel, _, n in shards:
        sel = []
        while ti < len(targets) and targets[ti] < gi + n:
            sel.append(targets[ti] - gi)
            ti += 1
        if sel:
            plan[rel] = sel
        gi += n
    return plan, total


def fetch_rows(path, row_indices):
    """按行号取行 (messages+images), 只在命中的 row-group 粒度读。"""
    pf = pq.ParquetFile(path)
    out = {}
    want = sorted(row_indices)
    base = 0
    for rgi in range(pf.metadata.num_row_groups):
        rg_rows = pf.metadata.row_group(rgi).num_rows
        lo, hi = base, base + rg_rows
        hit = [i for i in want if lo <= i < hi]
        if hit:
            t = pf.read_row_group(rgi, columns=["messages", "images"])
            rows = t.to_pylist()
            for i in hit:
                out[i] = rows[i - lo]
        base = hi
        if len(out) == len(want):
            break
    return [(i, out[i]) for i in want if i in out]


def last_pair(messages):
    """取最后一轮 user/assistant 对 (蒸馏刷的是末条 assistant)。"""
    q = a = None
    for m in messages or []:
        if m["role"] == "user":
            q = m["content"]
        elif m["role"] == "assistant":
            a = m["content"]
    return q or "", a or ""


def judge_row(row, pool, model_caps):
    """返回 (verdict_dict | None, auto_issue | None)。None=裁判解析失败。"""
    q, a = last_pair(row["messages"])
    if not (a or "").strip():
        return ({"match": False, "issues": ["empty_answer"],
                 "confidence": 1.0}, "empty_answer")
    q_cap, a_cap = model_caps
    q_clean = q.replace("<image>", " ").strip()  # 裁判展示剥掉占位符
    q_disp = q_clean if len(q_clean) <= q_cap else q_clean[:q_cap] + " …(截断)"
    if len(a) <= a_cap:
        a_disp = a
    else:
        h = a_cap * 3 // 4
        a_disp = a[:h] + "\n…(中部截断)…\n" + a[-(a_cap - h):]
    images = (row["images"] or [])[:MAX_IMAGES_PER_ROW]
    fake_msgs = [{"role": "user",
                  "content": "<image>\n" * len(images)
                             + JUDGE_TMPL.format(q=q_disp, a=a_disp)}]
    api_msgs = build_api_messages(fake_msgs, images)
    message = pool.chat(
        api_msgs,
        return_message=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    verdict = parse_verdict(message.get("content") or "")
    if verdict is None:
        verdict = parse_verdict(message.get("reasoning_content") or "")
    return verdict, None


def rewrite_q(row, q, a, issues, pool, args):
    """裁判改写 Q 对齐 A (A 不动)。失败返回 None。
    强制保证 <image> 占位符数量与原 Q 一致 (不足则按原数补前缀)。"""
    q_clean = q.replace("<image>", " ").strip()
    if len(a) <= args.a_cap:
        a_disp = a
    else:
        h = args.a_cap * 3 // 4
        a_disp = a[:h] + "\n…(中部截断)…\n" + a[-(args.a_cap - h):]
    images = (row["images"] or [])[:MAX_IMAGES_PER_ROW]
    fake = [{"role": "user",
             "content": "<image>\n" * len(images)
                        + REWRITE_TMPL.format(
                            q=q_clean, a=a_disp,
                            issues=", ".join(issues) or "不匹配")}]
    try:
        new = pool.chat(build_api_messages(fake, images)).strip()
    except Exception:  # noqa: BLE001
        return None
    if not new:
        return None
    # 轻校验: 拒绝空/过短/把裁判 verdict JSON 当成 Q 的伪改写
    if len(new) < 4 or len(new) > 2000:
        return None
    if new.startswith("{") and '"match"' in new:
        return None
    need = q.count("<image>")
    if new.count("<image>") != need:
        new = new.replace("<image>", "").strip()
        new = ("<image>\n" * need) + new
    return new


def fix_shard(path, pool, args):
    """修复模式单分片: 全量逐行裁判, 不匹配行改写 Q 后原子写回。
    A/图片/其余行均不动。返回统计 dict。"""
    st = {"rows": 0, "match": 0, "mismatch": 0, "auto_empty": 0,
          "judge_error": 0, "fixed": 0, "fix_failed": 0,
          "fix_skipped_multi": 0}
    fix_map = {}
    pf = pq.ParquetFile(path)
    seen = 0
    capped = False
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for batch in pf.iter_batches(batch_size=args.chunk_rows):
            rows = batch.to_pylist()
            if args.max_rows:
                rows = rows[:max(0, args.max_rows - seen)]
            if not rows:
                capped = True
                break

            def judge_and_rewrite(r):
                verdict, auto = judge_row(r, pool, (args.q_cap, args.a_cap))
                if auto or verdict is None or verdict["match"]:
                    return verdict, auto, None, False
                q, a = last_pair(r["messages"])
                n_user = sum(1 for m in r["messages"] or []
                             if m["role"] == "user")
                if n_user != 1:
                    return verdict, auto, None, True
                new_q = rewrite_q(r, q, a, verdict["issues"], pool, args)
                return verdict, auto, new_q, False

            futs = {ex.submit(judge_and_rewrite, r): seen + i
                    for i, r in enumerate(rows)}
            for f, idx in futs.items():
                verdict, auto, new_q, multi = f.result()
                st["rows"] += 1
                if auto:
                    st["auto_empty"] += 1
                    st["mismatch"] += 1
                    continue
                if verdict is None:
                    st["judge_error"] += 1
                    continue
                if verdict["match"]:
                    st["match"] += 1
                    continue
                st["mismatch"] += 1
                if multi:
                    st["fix_skipped_multi"] += 1
                elif new_q:
                    fix_map[idx] = new_q
                    st["fixed"] += 1
                else:
                    st["fix_failed"] += 1
            seen += len(rows)
            if args.max_rows and seen >= args.max_rows:
                capped = True
                break

    if fix_map:
        tmp = path + ".qafix_tmp"
        seen2 = 0
        pf2 = pq.ParquetFile(path)
        with pq.ParquetWriter(tmp, schema=pf2.schema_arrow) as writer:
            for batch in pf2.iter_batches(batch_size=args.chunk_rows):
                rows = batch.to_pylist()
                for i, r in enumerate(rows):
                    gi = seen2 + i
                    if gi in fix_map:
                        msgs = [dict(m) for m in r["messages"]]
                        for m in reversed(msgs):
                            if m["role"] == "user":
                                m["content"] = fix_map[gi]
                                break
                        r["messages"] = msgs
                writer.write_table(
                    pa.Table.from_pylist(rows, schema=pf2.schema_arrow))
                seen2 += len(rows)
        # Windows 下 os.replace 要求源文件句柄先释放 (Linux 无此限制)
        del pf, pf2
        import gc
        gc.collect()
        os.replace(tmp, path)  # 原子替换
    if not capped:
        # 完整处理过的分片才打标记; --max-rows 截断的不标记以便续跑
        with open(path + ".qafix_done", "w", encoding="utf-8") as f:
            f.write(f"fixed={st['fixed']} rows={st['rows']}\n")
    return st


def main():
    ap = argparse.ArgumentParser(description="蒸馏产物 Q-A 匹配质检 (VLM 裁判)")
    ap.add_argument("--data-root", required=True, help="蒸馏产物根目录")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="修复模式: 每分片最多裁判前 N 行 (调试用)")
    ap.add_argument("--per-dataset", type=int, default=200,
                    help="每数据集抽样行数 (默认 200, 确定性均布)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-rows", type=int, default=500,
                    help="修复模式流式读写的行块大小 (默认 500)")
    ap.add_argument("--model", default="qwen-vl")
    ap.add_argument("--num-instances", type=int, default=1)
    ap.add_argument("--base-port", type=int, default=31000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--judge-max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--q-cap", type=int, default=1000)
    ap.add_argument("--a-cap", type=int, default=4000)
    ap.add_argument("--min-match-rate", type=float, default=0.9,
                    help="匹配率下限 (默认 0.9), 低于则该数据集不达标")
    ap.add_argument("--report", default=None)
    ap.add_argument("--fix", action="store_true",
                    help="修复模式: 全量逐行裁判, 不匹配行改写 Q 对齐 A "
                         "(A 不动), 原地原子写回 (不可逆, 只用于蒸馏副本)")
    ap.add_argument("--resume", action="store_true",
                    help="修复模式: 跳过带 .qafix_done 标记的分片")
    args = ap.parse_args()

    if args.fix:
        print("*** 修复模式: 不匹配行的 Q 将被改写并直接写回源 parquet, "
              "不可逆! 请确认 --data-root 是蒸馏副本而非原始数据 ***")

    manifest_path = args.manifest or os.path.join(
        args.data_root, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        ap.error(f"manifest 不存在: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    base_urls = [f"http://{args.host}:{args.base_port + i}"
                 for i in range(args.num_instances)]
    pool = TeacherPool(base_urls, args.model, args.judge_max_tokens,
                       0.0, timeout=args.timeout, retries=args.retries)

    report = []
    any_fail = False
    for ds in manifest["datasets"]:
        name = ds["name"]
        if args.datasets and name not in args.datasets:
            continue
        files = ds.get("files") or []
        if args.max_files:
            files = files[:args.max_files]
        if args.fix:
            # ---- 修复模式: 全量逐行裁判 + 改写 Q 原子写回 ----
            st = {"rows": 0, "match": 0, "mismatch": 0, "auto_empty": 0,
                  "judge_error": 0, "fixed": 0, "fix_failed": 0,
                  "fix_skipped_multi": 0}
            n_shards = 0
            for finfo in files:
                rel = norm_rel_path(finfo["path"])
                p = os.path.join(args.data_root, rel)
                if not os.path.exists(p):
                    print(f"  [skip] missing {p}", file=sys.stderr)
                    continue
                if args.resume and os.path.exists(p + ".qafix_done"):
                    continue
                fst = fix_shard(p, pool, args)
                n_shards += 1
                for k in st:
                    st[k] += fst[k]
                print(f"  {rel}: {fst['rows']} 行, 修复 {fst['fixed']}, "
                      f"修复失败 {fst['fix_failed']}, match {fst['match']}")
            denom = max(st["match"] + st["mismatch"], 1)
            rate = st["match"] / denom
            fail = (st["fix_failed"] > 0
                    or st["judge_error"] > 0.1 * max(st["rows"], 1))
            any_fail = any_fail or fail
            line = (f"=== {name} [fix] {st['rows']} 行 {n_shards} 分片: "
                    f"match {rate:.1%}, 修复 {st['fixed']}, "
                    f"修复失败 {st['fix_failed']}, 多轮跳过 "
                    f"{st['fix_skipped_multi']}, 空A(不修) "
                    f"{st['auto_empty']}, 裁判失败 {st['judge_error']}")
            if fail:
                line += "  *** 修复不完全 ***"
            print(line)
            report.append({"name": name, "mode": "fix",
                           "shards": n_shards, **st,
                           "match_rate": round(rate, 4), "fail": fail})
            continue
        plan, total = plan_sample(files, args.data_root, args.per_dataset)
        rows = []
        for rel, idxs in plan.items():
            rows.extend(fetch_rows(os.path.join(args.data_root, rel), idxs))

        st = {"sampled": len(rows), "match": 0, "mismatch": 0,
              "auto_empty": 0, "judge_error": 0}
        issue_counter = {}

        def work(r):
            return judge_row(r, pool, (args.q_cap, args.a_cap))

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for verdict, auto in ex.map(work, [r for _, r in rows]):
                if auto:
                    st["auto_empty"] += 1
                    st["mismatch"] += 1
                    issue_counter[auto] = issue_counter.get(auto, 0) + 1
                    continue
                if verdict is None:
                    st["judge_error"] += 1
                    continue
                if verdict["match"]:
                    st["match"] += 1
                else:
                    st["mismatch"] += 1
                for iss in verdict["issues"]:
                    issue_counter[iss] = issue_counter.get(iss, 0) + 1

        denom = max(st["match"] + st["mismatch"], 1)
        rate = st["match"] / denom
        fail = rate < args.min_match_rate
        any_fail = any_fail or fail
        top_issues = sorted(issue_counter.items(), key=lambda kv: -kv[1])[:5]
        line = (f"=== {name}: 抽 {st['sampled']}/{total} 行, "
                f"match {rate:.1%} ({st['match']}/{denom}), "
                f"空A {st['auto_empty']}, 裁判失败 {st['judge_error']}")
        if top_issues:
            line += "  issues: " + ", ".join(f"{k}x{v}" for k, v in top_issues)
        if fail:
            line += f"  *** 不达标 (< {args.min_match_rate:.0%}) ***"
        print(line)
        report.append({"name": name, "total_rows": total, **st,
                       "match_rate": round(rate, 4), "fail": fail,
                       "issues": dict(top_issues)})

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"data_root": os.path.abspath(args.data_root),
                       "judge_model": args.model,
                       "min_match_rate": args.min_match_rate,
                       "datasets": report}, f, ensure_ascii=False, indent=1)
        print(f"\n报告: {args.report}")
    print("\nQA-COMPAT RESULT:", "FAIL" if any_fail else "PASS")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
