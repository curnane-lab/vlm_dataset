# -*- coding: utf-8 -*-
"""
distill_qa.py — 基于 vLLM 教师模型的 VLM QA 数据蒸馏脚本 (Linux)

流程:
  1. 读取 dataset_manifest.json, 枚举所有 parquet 分片
  2. 流式读取分片 (pyarrow iter_batches 按 --chunk-rows 行块读取,
     ParquetWriter 增量写出; 不整片 read_table, UniMER 单片 ~33 万行
     带图也不会撑爆内存)
  3. 按教师模式生成 response 替换每行的 assistant 内容
  4. 输出两种模式:
     备份模式(不加 --in-place): 蒸馏结果写到 --output-root, 目录结构与
       原数据一致; 处理失败的分片直接拷贝原始分片兜底, 原始 parquet
       全程只读。
     原地模式(--in-place): 蒸馏结果直接刷回源 parquet (临时文件 +
       os.replace 原子替换, 失败行保留原始回答), 每个成功分片旁写
       <分片>.distill_done 标记, --resume 按标记跳过。不可逆, 慎用!

并行模型: NUM_INSTANCES 个 vLLM 实例, 行级任务按 round-robin 分发到
各实例, 线程池并发 (每实例 WORKERS_PER_INSTANCE 个并发请求)。

教师模式 (--teacher-mode):
  two-step : 默认。走 MinerU 官方两阶段链路 (mineru_vl_utils.MinerUClient.
             aio_batch_two_step_extract): 第一阶段版面检测, 第二阶段按 bbox
             裁块并用内置分类型 prompt 识别, 与 OmniDocBench 评测同源。
             忽略数据集自带的 user prompt, 按数据集类型路由序列化方式:
               markdown  : 整页解析类 (Vary/Infinity-Doc/HRD parsing),
                           块列表渲染为 Markdown (title→#、equation→$$、
                           table→HTML、paratext 丢弃)
               grounding : HRD grounding, 序列化为 JSON [{class,text,bbox}]
               crop      : 识别级 crop (UniMER/MathWriting/CASIA/FinTabNet/
                           PubTables/场景文字识别), 整图跳过版面检测, 直送
                           分类型识别 prompt (equation/table/text), 空结果
                           时回退 two-step 裁块路径
               skip      : holdout 档 (tinychart + FineVision QA/KIE +
                           HRD grounding), 本轮跳过不蒸馏
             线程模型: 每个工作线程绑定一个实例 (round-robin), 持有持久的
             asyncio 事件循环 + MinerUClient (http-client), 连接复用。
  single   : 把对话历史+图片直接发给 /v1/chat/completions, 单步出答案。

支持断点续跑: 备份模式按输出分片行数判断, 原地模式按 .distill_done 标记。

依赖: pandas, pyarrow, requests, tqdm; two-step 模式另需 mineru_vl_utils, Pillow
"""

import argparse
import asyncio
import base64
import itertools
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import pyarrow.parquet as pq
import requests
from tqdm import tqdm

# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------

def norm_rel_path(p: str) -> str:
    """manifest 里的分片相对路径归一化为 Linux 形式。
    真实 manifest 在 Windows 上生成, 可能带反斜杠或 ./ 前缀。"""
    p = p.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    return p


def make_sampler(sample_rate: float):
    """确定性下采样器: keep(i) = crc32(分片路径#行号) 命中 sample_rate 比例。
    同分片同采样率重跑选中行完全一致。sample_dataset.py 预处理用。"""
    if sample_rate >= 1.0:
        return None
    import zlib
    threshold = int(sample_rate * 1_000_000)

    def keep(rel: str, i: int) -> bool:
        return zlib.crc32(f"{rel}#{i}".encode()) % 1_000_000 < threshold

    return keep


# ----------------------------------------------------------------------
# 教师模型调用
# ----------------------------------------------------------------------

def guess_mime(image_bytes: bytes) -> str:
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def image_to_data_url(image_bytes: bytes) -> str:
    return f"data:{guess_mime(image_bytes)};base64," + base64.b64encode(
        image_bytes).decode("ascii")


def build_api_messages(messages, images):
    """把 qwen_vl 风格的 messages 转成 OpenAI vision chat 格式。
    <image> 占位符替换为 image_url part; 图片按出现顺序取。"""
    api_msgs = []
    img_iter = iter(range(len(images)))
    img_cache = []

    def next_image_part():
        # 惰性编码, 同一张图在多轮中复用同一个 data_url
        try:
            idx = next(img_iter)
        except StopIteration:
            return None
        while len(img_cache) <= idx:
            img_cache.append(image_to_data_url(images[len(img_cache)]["bytes"]))
        return {"type": "image_url",
                "image_url": {"url": img_cache[idx]}}

    for m in messages:
        role, content = m["role"], m["content"]
        if role == "user" and "<image>" in content:
            parts = []
            # 支持一条消息里多个 <image>
            segs = content.split("<image>")
            for j, seg in enumerate(segs):
                if j > 0:
                    p = next_image_part()
                    if p is not None:
                        parts.append(p)
                if seg.strip():
                    parts.append({"type": "text", "text": seg})
            api_msgs.append({"role": role, "content": parts})
        else:
            api_msgs.append({"role": role, "content": content})
    return api_msgs


class TeacherPool:
    """round-robin 分发到多个 vLLM 实例, 带重试。"""

    def __init__(self, base_urls, model, max_tokens, temperature,
                 timeout=300, retries=3):
        self.base_urls = base_urls
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self._cycle = itertools.cycle(base_urls)
        self._lock = threading.Lock()
        self._sessions = threading.local()

    def _next_url(self):
        with self._lock:
            return next(self._cycle)

    def _session(self):
        if not hasattr(self._sessions, "s"):
            self._sessions.s = requests.Session()
        return self._sessions.s

    def chat(self, api_messages):
        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        last_err = None
        for attempt in range(self.retries):
            url = self._next_url() + "/v1/chat/completions"
            try:
                r = self._session().post(url, json=payload,
                                         timeout=self.timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"teacher call failed after {self.retries} "
                           f"retries: {last_err}")


# ----------------------------------------------------------------------
# two-step 教师: MinerU 官方两阶段链路 (版面检测 -> 裁块识别)
# ----------------------------------------------------------------------

def route_dataset(name: str) -> str:
    """two-step 模式的数据集路由: markdown / grounding / crop / skip。
    skip 名单与 vlm_dataset 仓 README 四档分类 (holdout) 对齐。"""
    n = name.lower()
    if "tinychart" in n:
        return "skip"          # 图表 QA: two-step 答非所问
    if "finevision" in n:
        # FineVision 仅场景文字识别 (iiit5k) 是 OCR 任务 (tier3), 其余 QA/KIE holdout
        return "crop" if "scene_text_recognition" in n else "skip"
    if "grounding" in n:
        return "skip"          # HRD grounding label 体系与 MinerU 版面类别不符
    if any(k in n for k in ("unimer", "mathwriting", "casia",
                            "fintabnet", "pubtables", "formula")):
        return "crop"
    return "markdown"


def crop_block_type(name: str) -> str:
    """crop 路由数据集的整图识别类型 (MinerU 内置分类型 prompt)。"""
    n = name.lower()
    if any(k in n for k in ("unimer", "mathwriting", "formula")):
        return "equation"
    if any(k in n for k in ("fintabnet", "pubtables")):
        return "table"
    return "text"  # casia 手写行 / scene_text_recognition 等


_MD_SKIP_TYPES = {"header", "footer", "page_number", "page_footnote",
                  "aside_text", "image", "chart", "image_block"}


def strip_math_delims(content: str) -> str:
    """剥掉公式识别结果外层的 $$..$$ / \\[..\\] 包装, 避免渲染时重复嵌套。"""
    c = content.strip()
    for left, right in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)")):
        if c.startswith(left) and c.endswith(right) and len(c) > 2 * len(left):
            return c[len(left):-len(right)].strip()
    return c


def render_blocks_markdown(blocks) -> str:
    """块列表 -> Markdown, 与 HRD 数据集构造规则一致的确定性排版。"""
    parts = []
    for b in blocks:
        t, c = b["type"], (b.get("content") or "").strip()
        if t in _MD_SKIP_TYPES or not c:
            continue
        if t in ("doc_title", "title"):
            parts.append(f"# {c}")
        elif t == "paragraph_title":
            parts.append(f"## {c}")
        elif t == "equation":
            parts.append(f"$$\n{strip_math_delims(c)}\n$$")
        elif t in ("code", "algorithm"):
            parts.append(f"```\n{c}\n```")
        else:
            parts.append(c)
    return "\n\n".join(parts)


def render_blocks_grounding(blocks) -> str:
    """块列表 -> JSON [{class, text, bbox}], bbox 升回到 0-1000 整数刻度。"""
    items = [{"class": b["type"],
              "text": b.get("content") or "",
              "bbox": [round(x * 1000) for x in b["bbox"]]}
             for b in blocks]
    return json.dumps(items, ensure_ascii=False)


def render_blocks_crop(blocks) -> str:
    """识别级 crop: 按阅读序拼接块内容 (LaTeX / 文本 / HTML), 丢弃页眉页脚等噪声。
    公式块的 \\[..\\] 包装剥掉, 与 UniMER 等裸 LaTeX 目标对齐。"""
    skip = {"header", "footer", "page_number", "page_footnote", "aside_text"}
    out = []
    for b in blocks:
        if b["type"] in skip:
            continue
        c = (b.get("content") or "").strip()
        if not c:
            continue
        out.append(strip_math_delims(c) if b["type"] == "equation" else c)
    return "\n".join(out)


_RENDERERS = {"markdown": render_blocks_markdown,
              "grounding": render_blocks_grounding,
              "crop": render_blocks_crop}


class TwoStepTeacherPool:
    """MinerUClient two-step 教师池。
    每个工作线程固定绑定一个实例 (round-robin), 持有持久事件循环和客户端,
    避免跨 loop 复用 httpx.AsyncClient 以及每行重建连接的开销。"""

    def __init__(self, base_urls, model, timeout=600):
        self.base_urls = base_urls
        self.model = model
        self.timeout = timeout
        self._tls = threading.local()
        self._assign = itertools.count()

    def _thread_ctx(self):
        if not hasattr(self._tls, "client"):
            from mineru_vl_utils import MinerUClient
            url = self.base_urls[next(self._assign) % len(self.base_urls)]
            loop = asyncio.new_event_loop()
            client = MinerUClient(
                backend="http-client",
                server_url=url,
                model_name=self.model,
                http_timeout=self.timeout,
                use_tqdm=False,
            )
            self._tls.loop, self._tls.client = loop, client
        return self._tls.loop, self._tls.client

    def extract(self, pil_images):
        """对一组 PIL 图跑两阶段抽取, 返回 list[ExtractResult] (与图一一对应)。"""
        loop, client = self._thread_ctx()
        return loop.run_until_complete(
            client.aio_batch_two_step_extract(pil_images))

    def extract_typed(self, pil_images, block_type):
        """crop 直连: 整图当整块, 跳过版面检测, 直送分类型识别 prompt。
        返回 list[ExtractStr|None]。小 crop 图做版面检测是离分布用法
        (实测乱出框/臆造类型), 直连又快又稳。"""
        loop, client = self._thread_ctx()
        return loop.run_until_complete(
            client.aio_batch_content_extract(pil_images, block_type))


def distill_row_two_step(row, pool: TwoStepTeacherPool, route: str,
                         crop_type: str = "text"):
    """two-step 版单行蒸馏: 忽略对话 prompt, 对全行图片跑抽取,
    按路由序列化后替换所有 assistant 轮次。失败时返回原始 messages。
    crop 路由走整图直连识别 (含 two-step 兜底)。"""
    import io
    from PIL import Image

    messages = [dict(m) for m in row["messages"]]
    raw_images = list(row["images"]) if row["images"] is not None else []
    try:
        pil_images = [Image.open(io.BytesIO(bytes(im["bytes"]))).convert("RGB")
                      for im in raw_images]
        if not pil_images:
            raise ValueError("row has no images")
        if route == "crop":
            outs = pool.extract_typed(pil_images, crop_type)
            parts = []
            for o in outs:
                c = (str(o) if o else "").strip()
                if c:
                    parts.append(strip_math_delims(c)
                                 if crop_type == "equation" else c)
            answer = "\n".join(parts).strip()
            if not answer:
                # 直连偶发空结果 -> 回退 two-step 裁块路径再试一次
                results = pool.extract(pil_images)
                answer = "\n\n".join(render_blocks_crop(r)
                                     for r in results).strip()
        else:
            results = pool.extract(pil_images)
            render = _RENDERERS[route]
            answer = "\n\n".join(render(r) for r in results).strip()
        if not answer:
            raise ValueError("two-step produced empty answer")
        for i, m in enumerate(messages):
            if m["role"] == "assistant":
                m["content"] = answer
        return messages, True, None
    except Exception as e:  # noqa: BLE001
        return row["messages"], False, str(e)


# ----------------------------------------------------------------------
# 单行蒸馏: 重写所有 assistant 轮次
# ----------------------------------------------------------------------

def distill_row(row, pool: TeacherPool):
    """返回 (new_messages, ok, err)。失败时返回原始 messages。"""
    messages = [dict(m) for m in row["messages"]]
    images = list(row["images"]) if row["images"] is not None else []
    try:
        # 找出所有 assistant 轮次的下标
        asst_idx = [i for i, m in enumerate(messages)
                    if m["role"] == "assistant"]
        for ai in asst_idx:
            prefix = build_api_messages(messages[:ai], images)
            answer = pool.chat(prefix)
            messages[ai]["content"] = answer.strip() if answer else ""
        return messages, True, None
    except Exception as e:  # noqa: BLE001
        return row["messages"], False, str(e)


# ----------------------------------------------------------------------
# 文件级处理 (pandas 读写 parquet)
# ----------------------------------------------------------------------

def process_file(src_path, dst_path, pool, max_rows=None,
                 teacher_mode="single", route=None, in_place=False,
                 chunk_rows=500, crop_type="text"):
    """流式处理一个分片: iter_batches 按块读取 -> 线程池蒸馏 ->
    ParquetWriter 增量写出。避免 UniMER 单分片 ~33 万行带图一次性进内存
    (数据集文档明确禁止 read_table 整片读取), 同时把在途 future 数
    约束在 chunk_rows 量级。"""
    import pyarrow as pa

    if teacher_mode == "two-step":
        row_fn = lambda row: distill_row_two_step(row, pool, route,  # noqa: E731
                                                  crop_type)
    else:
        row_fn = lambda row: distill_row(row, pool)  # noqa: E731

    pf = pq.ParquetFile(src_path)
    total_rows = pf.metadata.num_rows
    if max_rows:
        total_rows = min(total_rows, max_rows)

    out_path = src_path + ".tmp" if in_place else dst_path
    if not in_place:
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    seen = 0      # 已读源行数
    n = fail = 0  # n = 实际写出行数
    writer = pq.ParquetWriter(out_path, schema=pf.schema_arrow)
    try:
        with ThreadPoolExecutor(max_workers=pool.workers) as ex:
            # total_rows == 0 时 iter_batches 对 0 row-group 文件会抛异常, 短路
            batches = pf.iter_batches(batch_size=chunk_rows) if total_rows else []
            for batch in batches:
                if seen >= total_rows:
                    break
                chunk = batch.to_pandas()
                if seen + len(chunk) > total_rows:
                    chunk = chunk.head(total_rows - seen)
                seen += len(chunk)
                if len(chunk) == 0:
                    continue
                records = chunk.to_dict("records")
                results_msgs = [None] * len(records)
                futs = {ex.submit(row_fn, row): i
                        for i, row in enumerate(records)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    msgs, ok, err = fut.result()
                    results_msgs[i] = msgs
                    if not ok:
                        fail += 1
                        print(f"  [warn] row {n + i} failed: {err}",
                              file=sys.stderr)
                chunk["messages"] = pd.Series(
                    [[{"role": m["role"], "content": m["content"]}
                      for m in msgs] for msgs in results_msgs],
                    index=chunk.index)
                table = pa.Table.from_pandas(chunk, schema=pf.schema_arrow,
                                             preserve_index=False)
                writer.write_table(table)
                n += len(chunk)
    finally:
        writer.close()

    if in_place:
        # 原地模式: 临时文件 + 原子替换, 防止中途崩溃损坏源 parquet
        os.replace(out_path, src_path)
        with open(src_path + ".distill_done", "w") as f:
            f.write(f"rows={n} fail={fail}\n")
    return n, fail


def same_shape_done(dst_path, expected_rows):
    """断点续跑: 输出存在且行数 >= 期望行数视为已完成。"""
    if not os.path.exists(dst_path):
        return False
    try:
        return pq.read_metadata(dst_path).num_rows >= expected_rows
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="VLM QA 数据蒸馏 (vLLM 教师)")
    ap.add_argument("--data-root", default="/data/omnidoc_vlm",
                    help="数据集根目录 (Linux 上的路径, 对应 F:\\omnidoc_vlm)")
    ap.add_argument("--manifest", default=None,
                    help="dataset_manifest.json 路径, 默认 <data-root>/"
                         "dataset_manifest.json")
    ap.add_argument("--output-root", default="/data/omnidoc_vlm_distill")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="只处理这些数据集名, 默认全部")
    ap.add_argument("--max-files", type=int, default=None,
                    help="每个数据集最多处理多少个分片 (调试用)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="每个分片最多处理多少行 (调试用)")
    ap.add_argument("--num-instances", type=int, default=7)
    ap.add_argument("--base-port", type=int, default=8015)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="minerU")
    ap.add_argument("--teacher-mode", choices=["single", "two-step"],
                    default="two-step",
                    help="two-step(默认): MinerU 两阶段链路 (版面检测+裁块识别), "
                         "按数据集类型路由序列化; single: 对话直接发给教师")
    ap.add_argument("--in-place", action="store_true",
                    help="原地模式: 蒸馏结果直接刷回源 parquet (原子替换 + "
                         ".distill_done 标记), 不可逆!")
    ap.add_argument("--chunk-rows", type=int, default=500,
                    help="流式读取/写出的行块大小 (默认 500), 控制内存峰值")
    ap.add_argument("--workers-per-instance", type=int, default=16,
                    help="每个 vLLM 实例的并发请求数")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--resume", action="store_true",
                    help="跳过已完成的输出分片")
    ap.add_argument("--skip-footer-check", action="store_true",
                    help="跳过预检中的 parquet footer 可读性/行数体检 "
                         "(默认开启; 传输截断的文件存在但 footer 损坏)")
    args = ap.parse_args()

    # 安全检查: 备份模式下输出目录必须与数据目录不同, 防止覆盖原始 parquet
    if args.in_place:
        if args.max_rows:
            ap.error("--in-place 不能配 --max-rows (只蒸馏前 N 行再整体刷回"
                     "会丢掉剩余行)")
        print("*** 原地模式: 蒸馏结果将直接刷回源 parquet, 不可逆! ***")
    elif os.path.abspath(args.output_root) == os.path.abspath(args.data_root):
        ap.error("--output-root 不能与 --data-root 相同 (会覆盖原始数据); "
                 "如确需原地写回, 请显式加 --in-place")

    manifest_path = args.manifest or os.path.join(
        args.data_root, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        ap.error(f"manifest 不存在: {manifest_path} "
                 f"(检查 --data-root 是否指向数据集根目录)")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # ---- manifest 预检: 统计登记分片的存在性, 提前暴露路径分隔符/
    #      盘符等可移植性问题, 避免跑完全程才发现全部 missing ----
    pre_total = pre_missing = 0
    pre_missing_examples = []
    pre_datasets = 0
    for ds in manifest["datasets"]:
        if args.datasets and ds["name"] not in args.datasets:
            continue
        pre_datasets += 1
        files = ds.get("files") or []
        if args.max_files:
            files = files[:args.max_files]
        for finfo in files:
            pre_total += 1
            rel = norm_rel_path(finfo["path"])
            if ":" in rel.split("/", 1)[0] or rel.startswith("/"):
                pre_missing += 1
                pre_missing_examples.append(finfo["path"])
            elif not os.path.exists(os.path.join(args.data_root, rel)):
                pre_missing += 1
                pre_missing_examples.append(finfo["path"])
    print(f"manifest 预检: {pre_datasets} 个数据集, {pre_total} 个分片, "
          f"缺失 {pre_missing}")
    if pre_missing:
        print(f"  缺失样例: {pre_missing_examples[:3]}", file=sys.stderr)
        if pre_missing == pre_total:
            print("  *** 全部缺失! 检查 --data-root 是否正确, 或 manifest "
                  "路径是否含盘符/反斜杠 ***", file=sys.stderr)

    # ---- footer 体检: 存在不代表可读。传输截断的文件存在但 footer 损坏,
    #      跑几小时到该分片才崩; 逐分片 read_metadata + 行数比对, 提前暴露 ----
    if not args.skip_footer_check:
        pre_corrupt = []
        for ds in manifest["datasets"]:
            if args.datasets and ds["name"] not in args.datasets:
                continue
            files = ds.get("files") or []
            if args.max_files:
                files = files[:args.max_files]
            for finfo in files:
                rel = norm_rel_path(finfo["path"])
                p = os.path.join(args.data_root, rel)
                if not os.path.exists(p):
                    continue  # 缺失已在上面统计
                try:
                    n = pq.read_metadata(p).num_rows
                    if finfo.get("rows") is not None and n != finfo["rows"]:
                        pre_corrupt.append(
                            (rel, f"行数 {n} != manifest {finfo['rows']}"))
                except Exception as e:  # noqa: BLE001
                    pre_corrupt.append(
                        (rel, f"footer 不可读: {type(e).__name__}"))
        print(f"footer 体检: {pre_total - pre_missing} 个已存在分片, "
              f"损坏 {len(pre_corrupt)}")
        if pre_corrupt:
            for rel, why in pre_corrupt[:10]:
                print(f"  [corrupt] {rel}: {why}", file=sys.stderr)
            sys.exit("*** 存在损坏分片 (多为传输截断), 请重传这些文件后再跑; "
                     "确需跳过体检用 --skip-footer-check ***")

    base_urls = [f"http://{args.host}:{args.base_port + i}"
                 for i in range(args.num_instances)]
    if args.teacher_mode == "two-step":
        # mineru_vl_utils 的 loguru DEBUG 日志量大, 压到 WARNING
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
        pool = TwoStepTeacherPool(base_urls, args.model,
                                  timeout=max(args.timeout, 600))
    else:
        pool = TeacherPool(base_urls, args.model, args.max_tokens,
                           args.temperature, timeout=args.timeout,
                           retries=args.retries)
    pool.workers = args.num_instances * args.workers_per_instance

    print(f"教师实例: {base_urls}")
    print(f"教师模式: {args.teacher_mode}, 并发度: {pool.workers} 线程")

    total_files = total_rows = total_fail = 0
    skipped_datasets, skipped_files, copied_files = [], [], []
    for ds in manifest["datasets"]:
        name = ds["name"]
        if args.datasets and name not in args.datasets:
            continue
        route = None
        crop_type = "text"
        if args.teacher_mode == "two-step":
            route = route_dataset(name)
            if route == "skip":
                print(f"\n=== {name}: holdout 档数据集, 本轮跳过 ===")
                skipped_datasets.append(name)
                continue
            if route == "crop":
                crop_type = crop_block_type(name)
            print(f"\n=== {name}: 路由 {route}"
                  f"{f' (整图 {crop_type} 直连)' if route == 'crop' else ''} ===")
        files = ds.get("files") or []
        if args.max_files:
            files = files[:args.max_files]

        # 数据集子文件夹整体缺失 -> 直接跳过
        ds_dir = os.path.join(args.data_root, name)
        if not os.path.isdir(ds_dir):
            print(f"\n=== {name}: 本地目录缺失, 跳过 ({ds_dir}) ===",
                  file=sys.stderr)
            skipped_datasets.append(name)
            continue

        if args.teacher_mode != "two-step":
            print(f"\n=== {name} ({len(files)} 个分片) ===")
        else:
            print(f"  {len(files)} 个分片")
        for finfo in tqdm(files, desc=name):
            rel = norm_rel_path(finfo["path"])
            src = os.path.join(args.data_root, rel)
            dst = src if args.in_place else os.path.join(args.output_root, rel)
            expected = args.max_rows or finfo.get("rows", 0)
            if args.resume:
                if args.in_place:
                    # 原地模式按 .distill_done 标记判断 (源文件行数恒 >= 期望)
                    if os.path.exists(src + ".distill_done"):
                        continue
                elif same_shape_done(dst, expected):
                    continue
            if not os.path.exists(src):
                tqdm.write(f"  [skip] missing {src}")
                skipped_files.append(rel)
                continue
            t0 = time.time()
            try:
                n, fail = process_file(src, dst, pool, args.max_rows,
                                       args.teacher_mode, route,
                                       in_place=args.in_place,
                                       chunk_rows=args.chunk_rows,
                                       crop_type=crop_type)
            except Exception as e:  # noqa: BLE001
                if args.in_place:
                    # 原地模式不动源文件, 直接记录跳过
                    tqdm.write(f"  [skip] {rel} 处理失败: {e}, 源文件未修改")
                    skipped_files.append(rel)
                    continue
                # 源分片自身不可读(损坏/传输截断)时不做兜底拷贝,
                # 否则坏数据混进输出目录会被下游当成正常分片
                try:
                    pq.read_metadata(src)
                    src_readable = True
                except Exception:  # noqa: BLE001
                    src_readable = False
                if not src_readable:
                    tqdm.write(f"  [skip] {rel} 源分片损坏(footer 不可读), "
                               f"未拷贝进输出, 请重传源文件后 --resume")
                    skipped_files.append(rel)
                    continue
                # 教师/网络等处理失败但源完好 -> 拷贝原始分片兜底,
                # 保证输出目录完整, 不中断整体流程
                tqdm.write(f"  [skip] {rel} 处理失败: {e}, 拷贝原始分片兜底")
                try:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    copied_files.append(rel)
                except OSError as ce:
                    tqdm.write(f"  [skip] {rel} 兜底拷贝也失败: {ce}")
                    skipped_files.append(rel)
                continue
            dt = time.time() - t0
            total_files += 1
            total_rows += n
            total_fail += fail
            tqdm.write(f"  {rel}: {n} 行, 失败 {fail}, "
                       f"{dt:.1f}s ({n / max(dt, 1e-6):.1f} 行/s)")

    print(f"\n完成: {total_files} 个文件, {total_rows} 行, "
          f"失败 {total_fail} 行")
    if skipped_datasets:
        print(f"跳过的数据集 ({len(skipped_datasets)}): "
              f"{', '.join(skipped_datasets)}")
    if skipped_files:
        print(f"缺失/跳过的分片 ({len(skipped_files)}): "
              f"{', '.join(skipped_files[:10])}"
              f"{' ...' if len(skipped_files) > 10 else ''}")
    if copied_files:
        print(f"失败分片已拷贝原始文件兜底 ({len(copied_files)}): "
              f"{', '.join(copied_files[:10])}"
              f"{' ...' if len(copied_files) > 10 else ''}")
    if args.in_place:
        print(f"原地模式: 结果已刷回源数据目录 {args.data_root}")
    else:
        print(f"输出目录: {args.output_root} (原始数据 {args.data_root} 未被修改)")


if __name__ == "__main__":
    main()
