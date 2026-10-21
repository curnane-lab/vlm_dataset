# OmniDocBench NPU 评测环境配置指南

> 平台: openEuler aarch64 + Ascend 910B3 NPU + MinerU2.5-Pro-2604-1.2B (pure VLM two-step)
> 归档日期: 2026-10-21 · 评测仓版本: OmniDocBench v1.6
> 本文记录在一台**离线 NPU 服务器**上复现 OmniDocBench 端到端评测的完整环境配置。
> 注意: 评测仓副本与预测目录已在宿主机删除, 文中补丁细节来自会话记录, 重建时需重新打补丁。

## 1. 复测成绩 (快照)

| 指标 | 本环境 (NPU) | 论文报告 |
|---|---:|---:|
| Full Overall | **95.24** | 95.69 |
| Text Edit-dist ↓ | **0.0347** | 0.036 |
| Formula CDM ↑ | 95.76 | 97.29 |
| Table TEDS ↑ | 93.43 | 93.42 |
| Table TEDS-S ↑ | 95.97 | — |
| ReadOrder Edit-dist ↓ | 0.1253 | — |
| Base / Hard split | 95.67 / 93.48 | — |

结论: NPU 平台整体达到论文水平; CDM 差距主要归因于渲染栈差异 (见第 5 节)。

## 2. 推理侧 (被测对象)

模型服务 (pure VLM, 无小模型):

```bash
# 每卡一个实例, 关键: --hf-overrides 修复官方 config.json 顶层缺失
# tie_word_embeddings 导致的 lm_head 不绑定 (不修则输出全 '!')
ASCEND_RT_VISIBLE_DEVICES=0 mineru-kit vlm-server --engine vllm \
  --model /data/mineru/models/MinerU2.5-Pro-2604-1.2B \
  --served-model-name mineru-vlm --host 127.0.0.1 --port 30000 \
  --gpu-memory-utilization 0.8 \
  --hf-overrides '{"tie_word_embeddings":true}'
# 环境变量: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
# 多卡批量拉起见 pipeline/start_teachers.sh
```

批量推理 (页面图 → 每页 markdown, 断点续跑, 每实例并发 8):

```bash
python3 omnidocbench_infer.py \
  --gt /data/mineru/datasets/OpenDataLab/OmniDocBench/OmniDocBench.json \
  --images /data/mineru/datasets/OpenDataLab/OmniDocBench/images \
  --out /data/mineru/odb_preds \
  --servers http://127.0.0.1:30000/v1[,30001...] --batch-per-instance 8
# 内部: MinerUClient(backend=http-client).two_step_extract(image) -> json2md -> <stem>.md
```

数据: 1651 页页面图片 + `OmniDocBench.json` GT, 存 `datasets/OpenDataLab/OmniDocBench/`。

## 3. 评测环境 odb_venv (Python 3.10)

官方评测只支持 Python 3.10, 且老版本依赖与系统环境 (py3.12 + numpy 1.26.4) 冲突,
故建独立 venv, **与推理/蒸馏链路完全隔离**:

```bash
/usr/bin/python3.10 -m venv /data/mineru/odb_venv   # 或 uv venv --python 3.10
source /data/mineru/odb_venv/bin/activate
# 离线/镜像: pip -i https://mirrors.aliyun.com/pypi/simple/  (uv 需 UV_CACHE_DIR=/tmp/uv-cache)
git clone <gh-proxy>/opendatalab/OmniDocBench && cd OmniDocBench   # v1.6
pip install -e .                    # 装评测仓及大部分依赖
pip install datasets==2.20.0 huggingface_hub==0.24.7 numpy==1.24.4  # 关键钉版
```

关键钉版理由:

| 包 | 版本 | 原因 |
|---|---|---|
| datasets | 2.20.0 | 新版 datasets/hub 组合无法加载 GT 数据集脚本 |
| huggingface_hub | 0.24.7 | 同上, 配对钉版 |
| numpy | 1.24.4 | venv 内独立; 不影响系统 1.26.4 (torch_npu ABI 要求) |

其余依赖由 `pip install -e .` 带入 (参考版本: evaluate 0.4.3, apted 1.0.3,
pylatexenc 2.10, func_timeout 4.3.5, Levenshtein 0.25.1, rapidfuzz 3.14.5,
pandas 2.0.3, pillow 10.4.0, scipy 1.10.1, matplotlib 3.7.5, lxml 4.9.1,
beautifulsoup4 4.11.1, multiprocess 0.70.16, nltk 3.9.1, PyYAML 6.0.2)。

## 4. 评测仓补丁 (必须重打, 否则评测死锁)

评测在多线程页面匹配中再 `multiprocessing.Process` fork 子进程, **线程池 + fork
组合会死锁** (fork 继承锁状态)。两处修改 (会话记录, 仓已删, 重建时重新定位):

- `src/dataset/text_postprocess.py`
- `src/core/cal_metric.py` (文件名以仓内实际为准)

改法: 所有 `multiprocessing.Process/Pool` 改为 spawn 上下文:

```python
# 改法示例
ctx = multiprocessing.get_context("spawn")
p = ctx.Process(target=..., args=...)   # 原 multiprocessing.Process(...)
```

症状对照: 评测跑到匹配/指标阶段**完全卡死、CPU 占用归零**就是这个问题的表象。

## 5. CDM 渲染栈 (与官方最大的环境差异)

CDM 指标需要 pdflatex 把公式渲染成 PDF 再栅格化成 PNG 比对。官方验证栈:
TeX Live **2025** + ImageMagick **7.1.1-47** + Ghostscript **9.55.0**。

本环境:

- **TeX Live 2026** (`/usr/local/texlive`, pdflatex 横幅 2026; CTAN 镜像 mirrors.aliyun.com 装不了 TeX, 走的是官方 tlnet 镜像站)
- **`magick` 兼容 shim** (`/usr/local/bin/magick`, Python + PyMuPDF 实现),
  **不装 ImageMagick/Ghostscript**。shim 只实现 CDM 用到的两种调用:

```
magick -density 200 -quality 100 file.pdf[0] out.png        # 首页 200DPI 栅格化
magick -density 200 -quality 100 file.pdf -append out.png   # 全部页纵向拼接
```

渲染栈差异是 CDM 95.76 vs 论文 97.29 的主要归因 (字体度量/栅格化细节不同)。
如需对齐论文 CDM, 按官方 README Option B 编译 IM7 + 装 gs 9.55 并删除 shim。

## 6. 评测配置与入口

入口命令 (评测仓根目录, 官方 README 约定):

```bash
cd /data/mineru/OmniDocBench && source /data/mineru/odb_venv/bin/activate
python pdf_validation.py --config /data/mineru/odb_eval.yaml
# 产物: ./result/<run_name>_run_summary.json 等
```

本环境使用的 `odb_eval.yaml` 全文 (end2end, 四组指标, 32 并发匹配):

```yaml
end2end_eval:
  metrics:
    text_block:
      metric: [Edit_dist]
    display_formula:
      metric: [Edit_dist, CDM]
      cdm_workers: 32
    table:
      metric: [TEDS, Edit_dist]
      teds_workers: 32
    reading_order:
      metric: [Edit_dist]
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: /data/mineru/datasets/OpenDataLab/OmniDocBench/OmniDocBench.json
    prediction:
      data_path: /data/mineru/odb_preds
    match_method: quick_match
    match_workers: 32
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
```

## 7. 完整复跑清单

```bash
# 0. 数据就位: datasets/OpenDataLab/OmniDocBench/{OmniDocBench.json, images/}
# 1. 拉起教师实例池
bash /data/mineru/start_teachers.sh            # 7 卡 x 0.8 显存
# 2. 推理 (~1651 页, 7 实例约 20-40 分钟)
python3 /data/mineru/omnidocbench_infer.py --gt .../OmniDocBench.json \
  --images .../images --out /data/mineru/odb_preds \
  --servers http://127.0.0.1:30000/v1,http://127.0.0.1:30001/v1,...
# 3. 评测 (见第 6 节; 评测仓需按第 4 节打补丁)
```

## 8. 已知坑清单

| 坑 | 现象 | 处置 |
|---|---|---|
| 官方 config.json `tie_word_embeddings` 只写在内层 text_config | vLLM 输出全 `!` | 启动加 `--hf-overrides '{"tie_word_embeddings":true}'`, 不改模型文件 |
| 线程池 + fork 死锁 | 评测卡死 CPU 归零 | 第 4 节 spawn 补丁 |
| quick_match 超时 | 个别大页匹配超 300s | 配置 fallback (penalty 0.10); 本次全量 2 页超时 |
| 超长中文文件名 | tmp 文件超 ext4 255B 上限 | infer 脚本的 tmp 命名已含 pid+线程 id, 勿再追加长前缀 |
| editable install 悬空 | 删仓后 venv 不能复跑 | 重新 clone 到原路径 `/data/mineru/OmniDocBench` 再 `pip install -e .` |
| 系统 numpy | torch_npu/scipy ABI 依赖 1.26.4 | 系统环境钉 1.26.4; odb_venv 内 1.24.4 互不影响 |
| 离线网络 | github/pypi 直连不稳 | pip 走 aliyun 镜像; GitHub 走 gh-proxy 或 REST API |
