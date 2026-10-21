# vlm_dataset — OmniDoc VLM 数据集归档

> 归档日期: 2026-10-20
> 数据位置: `F:\omnidoc_vlm`(移动硬盘)
> 校验状态: 全量只读校验 PASS(逐分片 parquet footer 行数核对,0 缺失 0 损坏)

## 1. 总量概览

| 指标 | 数量 |
|---|---:|
| 登记数据集 | **18 个** |
| parquet 分片 | **959 个** |
| 总行数 | **4,478,676 行** |
| schema | 统一 `qwen_vl`(全部 `schema_adapter=qwen_vl`, `weight=1.0`) |

另有 **1 个未登记数据集**: `olmOCR-mix-1025_parsing_vlm`(64 分片 / 261,988 行 / ~280 GiB),
不在 `dataset_manifest.json` 中,目录结构不同(无 `vlm_parquet/` 子层,含 `parts/` 残留),
所有按 manifest 驱动的脚本默认忽略它;启用前需先登记。

## 2. 存储布局与 manifest 结构

```
F:\omnidoc_vlm\
├── dataset_manifest.json            # 总目录
├── dataset_manifest_{tier1,tier2,tier3,smoke,holdout}.json
└── <数据集名>\                         # 如 UniMER_Dataset_vlm\
    └── vlm_parquet\
        └── train-XXXXX-of-NNNNN.parquet
```

manifest schema:

```json
{
  "version": 1,
  "data_root": "F:\\omnidoc_vlm",
  "datasets": [{
    "name": "UniMER_Dataset_vlm",
    "schema_adapter": "qwen_vl",
    "weight": 1.0,
    "total_rows": 986122,
    "num_files": 3,
    "files": [{"path": "UniMER_Dataset_vlm/vlm_parquet/train-00000-of-00003.parquet",
               "rows": 328708, "schema_adapter": "qwen_vl"}, ...]
  }]
}
```

## 3. 行 schema(全部数据集统一)

每行两列:

| 列 | 类型 | 说明 |
|---|---|---|
| `messages` | `list<struct<role: string, content: string>>` | 多轮对话, role 仅 `user`/`assistant`, 末条必为 assistant 且非空 |
| `images` | `list<struct<bytes: binary, path: string>>` | 图片原始字节(PNG/JPEG)内嵌 + 原路径 |

约束(转换期 enforced + 抽样验证):

- user 消息中 `<image>` 占位符数量 **严格等于** `images` 长度,按序一一对应
- 绝大多数为单图单轮;FineVision 系 / tinychart 存在多轮 QA 行
- prompt 从任务池确定性选取: `pool[crc32(stable_key) % len(pool)]`,同行重算必同 prompt

**读取注意**: 图片字节内嵌导致单分片常为 GB 级,务必列裁剪
(`columns=["messages"]` 时 280GB 的 olmOCR 只读 ~2GB)并使用流式 `iter_batches`。

## 4. 四档分类(按 MinerU2.5-Pro 蒸馏适配度)

MinerU2.5-Pro 是文档解析专用 VLM(整页/块图 → MD/HTML/LaTeX,不支持自由问答)。
据此把 18 个数据集分为四档,tier manifest 见本仓 `manifests/`(数据盘 `F:\omnidoc_vlm\` 下有相同文件,
由 `make_tier_manifests.py` 从主 manifest 生成)。

| 档位 | 定位 | 数据集 | 分片 | 行数 | 占比 |
|---|---|---|---:|---:|---:|
| **tier1** | 文档页面解析(MinerU 本职) | 4 | 775 | **890,447** | 19.9% |
| **tier2** | 结构化识别: 公式→LaTeX / 表格→HTML | 5 | 88 | **1,922,405** | 42.9% |
| **tier3** | OCR 但特殊情况(手写行/场景文字) | 2 | 2 | **35,391** | 0.8% |
| **holdout** | 本轮不刷: QA 类 + grounding | 7 | 94 | **1,630,433** | 36.4% |
| 合计 | | 18 | 959 | **4,478,676** | 100% |

### tier1 — 文档页面解析(890,447 行)

| 数据集 | 分片 | 行数 | 说明 |
|---|---:|---:|---|
| Vary-600k_parsing_vlm | 38 | 480,115 | 整页 OCR → 纯文本(中英双语) |
| Infinity-Doc-400K_parsing_vlm | 729 | 386,781 | 文档解析 → MD/HTML |
| HRDH_parsing_vlm | 4 | 14,365 | 版面解析 → MD(原答案含 `[脚注]` 等自定义模板,刷后对齐全 MinerU 风格) |
| HRDS_parsing_vlm | 4 | 9,186 | 同上(英文) |

### tier2 — 结构化识别(1,922,405 行)

| 数据集 | 分片 | 行数 | 说明 |
|---|---:|---:|---|
| UniMER_Dataset_vlm | 3 | 986,122 | 公式 → LaTeX |
| PubTables-1M_OTSL-v1.1_HTML_vlm | 48 | 522,874 | 表格 → HTML |
| MathWriting-human_vlm | 3 | 229,864 | 手写公式 → LaTeX(需抽样验证手写质量) |
| PubTables-v2_cropped_table2html_vlm | 23 | 108,710 | 裁剪表格 → HTML |
| FinTabNet_OTSL_HTML_vlm | 11 | 74,835 | 表格 → HTML(原答案为 OTSL 拼装风格) |

### tier3 — OCR 特殊情况(35,391 行)

| 数据集 | 分片 | 行数 | 说明 |
|---|---:|---:|---|
| CASIA-HMDB2-line_vlm | 1 | 33,401 | 手写中文行。**原答案编码损坏(乱码)**,regenerate 可修复;需先冒烟验证 MinerU 手写行效果 |
| FineVision_scene_text_recognition_vlm | 1 | 1,990 | iiit5k 场景裁字识别,自然场景域外,优先级最低 |

### holdout — 本轮不刷(1,630,433 行)

| 数据集 | 分片 | 行数 | 原因 |
|---|---:|---:|---|
| tinychart_train_vlm | 10 | 1,520,677 | 图表 QA,需图表理解+推理 |
| FineVision_kie_vlm | 1 | 33,616 | KIE 字段抽取(SROIE) |
| FineVision_scene_text_vqa_vlm | 5 | 21,943 | 场景文字 VQA(TextVQA) |
| FineVision_handwritten_qa_vlm | 44 | 20,457 | 手写问答(hw_squad) |
| HRDH_grounding_vlm | 4 | 14,365 | grounding JSON 的 label 体系与 MinerU 版面类别不符 |
| FineVision_document_qa_vlm | 26 | 10,189 | 文档问答(DocVQA) |
| HRDS_grounding_vlm | 4 | 9,186 | 同 HRDH_grounding |

QA 类如后续要刷,需走通用教师 VLM 重答方案(distill_qa.py),不能用 MinerU。

## 5. 渐进式验证路径(regenerate → 训练 → ablation)

prompt 确定性选取、regenerate 只替换末条 assistant,各档间唯一变量是答案来源,
可做干净对照:

```bash
# ========== 前置: 拉起 7 卡教师池 (NPU 训练机, Ascend 910B + vLLM) ==========
bash /data/mineru/start_teachers.sh

DATA=/data/omnidoc_vlm          # F 盘挂载点(按实际改)
REGEN=/data/omnidoc_vlm_regen   # 再生输出根目录(大盘, 按实际改)
COMMON="--num-instances 7 --base-port 30000 --model mineru-vlm --workers-per-instance 8"

# Step 0: 全链路冒烟(11 数据集首分片 x 20 行, 备份模式不动源数据)
python3 /data/mineru/distill/distill_qa.py --data-root $DATA \
  --manifest $DATA/dataset_manifest_smoke.json \
  --output-root /tmp/distill_smoke --max-rows 20 $COMMON
# -> 抽查 /tmp/distill_smoke 各数据集答案质量, 重点: CASIA 手写行(tier3 唯一风险点)

# Step 0.5 (可选): 先导试验用下采样副本, 如 tier2 192 万行 -> 约 9.6 万行
# 采样只读源数据, 产物是独立小数据集; 蒸馏用 --in-place 刷副本即可(不碰源)
python3 /data/mineru/distill/sample_dataset.py --data-root $DATA \
  --manifest $DATA/dataset_manifest_tier2.json \
  --output-root $REGEN/tier2_s20 --sample-rate 0.05
python3 /data/mineru/distill/distill_qa.py --data-root $REGEN/tier2_s20 \
  --in-place $COMMON --resume

# Step 1: tier1 全量 -> 训练 -> 文档解析指标 (775 分片 / 89.0 万行)
python3 /data/mineru/distill/distill_qa.py --data-root $DATA \
  --manifest $DATA/dataset_manifest_tier1.json \
  --output-root $REGEN/tier1 $COMMON --resume

# Step 2: tier2 全量 -> 训练 -> 对比 tier1-only vs tier1+2 (88 分片 / 192.2 万行)
python3 /data/mineru/distill/distill_qa.py --data-root $DATA \
  --manifest $DATA/dataset_manifest_tier2.json \
  --output-root $REGEN/tier12 $COMMON --resume

# Step 3: tier3 先人工抽查 MinerU 手写/场景文字质量,行则刷 (2 分片 / 3.5 万行)
python3 /data/mineru/distill/distill_qa.py --data-root $DATA \
  --manifest $DATA/dataset_manifest_tier3.json \
  --output-root $REGEN/tier123 $COMMON --resume
```

- manifest 的 `files[].path` 为相对路径;regenerate 输出保持相同相对结构,
  训练时把 tier manifest 拷到备份根目录(或改 `data_root`)即可在
  原始 / tier1 刷后 / tier1+2 刷后之间切换做 ablation。
- regenerate 管线(`pipeline/distill_qa.py`, NPU 训练机侧): parquet 流式
  读取(`iter_batches` 块读, 不整片进内存) → 图片内存解码直送教师池
  (MinerU2.5-Pro two-step 两阶段: 版面检测 + 裁块识别, 与 OmniDocBench
  评测同源) → 重写备份 parquet。**无任何中间产物落盘**(不生成 PDF/中间
  markdown/html 文件);数据集按类型自动路由序列化(tier1→markdown,
  tier2/3→crop 块内容直拼, holdout 自动 skip);支持断点续跑(`--resume`)、
  缺失目录自动 skip;`--in-place` 可改原地刷回(原子替换, 牺牲 ablation 换磁盘)。

## 6. 已知问题

| 问题 | 影响 | 处置 |
|---|---|---|
| olmOCR-mix-1025_parsing_vlm 未登记进 manifest | 脚本默认忽略(64 分片 / 261,988 行 / ~280 GiB) | 启用前补登记;另有 222 个源行因超长页渲染失败被确定性 drop(见 render_fail_ids.json) |
| CASIA-HMDB2-line assistant 中文乱码 | 答案不可用 | 源数据即如此;tier3 regenerate 可修复 |
| 数据在移动硬盘 F: | 盘符随机器变化 | 训练机上以 `data_root` 参数/manifest 字段为准 |

## 7. 仓库内容

本仓归档统计信息、四档分类(本文档)与全部 manifest 文件(`manifests/`,共 ~332KB),不含 parquet 数据。
数据本体位于数据盘 `F:\omnidoc_vlm\`:

```
manifests/
├── dataset_manifest.json            # 主 manifest (18 数据集 / 959 分片 / 4,478,676 行)
├── dataset_manifest_tier1.json      # 文档页面解析   775 分片 / 890,447 行
├── dataset_manifest_tier2.json      # 公式+表格       88 分片 / 1,922,405 行
├── dataset_manifest_tier3.json      # 手写行/场景文字   2 分片 / 35,391 行
├── dataset_manifest_smoke.json      # 冒烟: 11 数据集各首分片(须配 --max-rows)
└── dataset_manifest_holdout.json    # QA+grounding    94 分片 / 1,630,433 行(不刷)
```

`pipeline/` 为 NPU 训练机侧的蒸馏管线脚本(Ascend 910B + vLLM,教师 MinerU2.5-Pro):

```
pipeline/
├── start_teachers.sh      # 拉起教师实例池(参数化卡数/端口/显存, 幂等; 默认 7 卡 0.8)
├── distill_qa.py          # 蒸馏主脚本: manifest 驱动, two-step(默认, MinerU
│                          # 两阶段) / single 两种教师模式, 流式读写 parquet,
│                          # 支持原地刷回(--in-place)与备份两种输出, 断点续跑
├── sample_dataset.py      # 确定性下采样预处理(crc32 哈希选行): 源数据只读,
│                          # 产物为独立小数据集 + 行数修正后的 manifest, 先导试验用
└── make_mini_dataset.py   # 生成 qwen_vl 冒烟小数据集(全链路自检用)
```

冒烟示例: `python3 pipeline/distill_qa.py --data-root <数据根> --manifest manifests/dataset_manifest_smoke.json --output-root /tmp/smoke --max-rows 20 --num-instances 3 --base-port 30000 --model mineru-vlm`

评测环境文档:

- `docs/omnidocbench_npu_eval_guide.md` — OmniDocBench v1.6 在 NPU 上的端到端
  评测环境配置指南 (推理侧 vlm-server + two-step, 评测侧 odb_venv/py3.10,
  CDM 渲染栈 shim 方案, 评测仓 spawn 补丁, 复测成绩 95.24 vs 论文 95.69)
