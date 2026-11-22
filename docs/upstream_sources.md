# 上游开源数据集出处对照

> 归档日期: 2026-11-05
> 用途: 回连原始 ID/URL 做训练/评测去重、按原始属性(含表页/文档类型)定向选样、
> 追溯标签来源。转换映射(源字段 → messages/images)的权威实现记录见
> `pipeline/` 构建脚本与本仓 README 第 2/4 节。

## 注意: 盘上现状

- `F:\omnidoc_vlm\<数据集>\` 目前**只保留转换后的 `vlm_parquet/`**; 当时下载的
  源 parquet (`parquet/` 子目录) 已被清理。要做**字段级回溯**(Infinity 的
  `id`/`attributes`、olmOCR 的 `url`/`is_table` 等) 需重新下载上游原始包。
- FinTabNet 与 5 个 FineVision 目录当前不在 F 盘 (可能被移走); manifest
  仍登记, 脚本按缺失自动 skip。

## 对照表

| 本地数据集 | 上游出处 | 本地行数 | 转换时使用的源字段 / 备注 |
|---|---|---:|---|
| UniMER_Dataset_vlm | [HF wanderkid/UniMER_Dataset](https://huggingface.co/datasets/wanderkid/UniMER_Dataset) · [OpenDataLab UniMER-Dataset](https://opendatalab.com/OpenDataLab/UniMER-Dataset) · [GitHub UniMERNet](https://github.com/opendatalab/UniMERNet) (arXiv 2404.15254) | 986,122 | UniMER-1M 子集; `filename`/`formula`/`image` |
| MathWriting-human_vlm | [GitHub google-research/mathwriting](https://github.com/google-research/google-research/tree/master/mathwriting) (arXiv 2404.10690) | 229,864 | 230K human-written 全量 (另有 400K synthetic 未用); `latex`/`image`/`sample_id` |
| CASIA-HMDB2-line_vlm | [CASIA-HWDB (中科院 NLPR)](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html) | 33,401 | **实为 CASIA-HWDB2.0~2.2 脱机手写中文文本行** (52,230 行的子集); `text`/图像路径。目录名 HMDB2 易误解, 内容非数学公式库 |
| Vary-600k_parsing_vlm | [GitHub Ucas-HaoranWei/Vary-tiny-600k](https://github.com/Ucas-HaoranWei/Vary-tiny-600k) | 480,115 | 中英整页 OCR; `lang`/`caption`/`image_bytes` |
| FinTabNet_OTSL_HTML_vlm | [HF ds4sd/FinTabNet_OTSL](https://huggingface.co/datasets/ds4sd/FinTabNet_OTSL) (原始 FinTabNet by Zheng et al.) | 74,835 | OTSL 骨架 `html_restored` + `cells` tokens 拼完整 HTML |
| PubTables-1M_OTSL-v1.1_HTML_vlm | [GitHub microsoft/table-transformer](https://github.com/microsoft/table-transformer) (PubTables-1M, CVPR 2022) | 522,874 | `html_with_text` 直接 join (OTSL v1.1 变体) |
| PubTables-v2_cropped_table2html_vlm | PubTables-v2 (Smock et al., arXiv 2512.10888) | 108,710 | `table_json` → html |
| Infinity-Doc-400K_parsing_vlm | [HF infly/Infinity-Doc-400K](https://huggingface.co/datasets/infly/Infinity-Doc-400K) | 386,781 | `id`/`gt`(MD/HTML)/`attributes`(`task`=table_parsing 等); **回连 ID 可做 ODB 去重** |
| olmOCR-mix-1025_parsing_vlm | [HF allenai/olmOCR-mix-1025](https://huggingface.co/datasets/allenai/olmOCR-mix-1025) | 261,988 (未登记 manifest) | `id`/`pdf`/`natural_text`; 上游另有 `url`/`page_number`/`is_table` 可筛含表页; 另有 222 源行渲染失败被 drop (见 HANDOFF) |
| FineVision_document_qa_vlm | [HF HuggingFaceM4/FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) 子集 DocVQA | 10,189 | `texts` 多轮原样保留 + 多图 |
| FineVision_handwritten_qa_vlm | 同上, 子集 HW-SQuAD | 20,457 | 同上 |
| FineVision_kie_vlm | 同上, 子集 SROIE | 33,616 | 同上 |
| FineVision_scene_text_recognition_vlm | 同上, 子集 IIIT5K | 1,990 | 同上 |
| FineVision_scene_text_vqa_vlm | 同上, 子集 TextVQA | 21,943 | 同上 |
| tinychart_train_vlm | [GitHub X-PLUG/mPLUG-DocOwl](https://github.com/X-PLUG/mPLUG-DocOwl) (TinyChart, arXiv 2404.16635) | 1,520,677 | `conversations` JSON, 一图多 QA 拆成多行; 图表 QA, holdout |
| HRDH_parsing_vlm / HRDH_grounding_vlm | [GitHub jfma-ustc/HRDoc](https://github.com/jfma-ustc/HRDoc) (AAAI 2023) — **HRDoc-Hard** (1,500 文档) | 各 14,365 | `annotations`(class/text/box) 按阅读序重排为 MD / grounding JSON |
| HRDS_parsing_vlm / HRDS_grounding_vlm | 同上 — **HRDoc-Simple** (1,000 文档, 规则解析自 ACL PDF) | 各 9,186 | 同上; 上游规则明确移除图表元素, 不含表格监督, 勿当表格增强源 |

## 对去重/选样的直接建议

- **与 OmniDocBench 去重**: 优先用 Infinity `id`、olmOCR `url` 回连源文档;
  图像级近重复用 `pipeline/check_overlap_omnidocbench.py` (md5+aHash)。
- **筛含表整页**: Infinity 用 `attributes.task`, olmOCR 用 `is_table` 字段,
  比转换后靠答案正则反推更准。
- **HRDS/HRDH**: 是语义层级/阅读结构数据, 不是表格增强源 (HRDS 规则明确
  移除图表元素)。
- **CASIA-HMDB2-line**: 是手写中文文本行 (CASIA-HWDB2.x), 与公式库无关。

## 其他开源复杂表格数据集调研 (2026-11-05)

目标: 补"整页上下文 + 单元格级结构 (含合并单元格)"的监督。

### 先用盘上现成的: olmOCR 实测 (首分片前 3000 行)

| 指标 | 实测值 | 对比 |
|---|---|---|
| 含表页占比 | 16.2% | 按内容自然分布 (可用 is_table 筛) |
| 多表页 | 24.8% | — |
| 行数/表 | p50=7, p90=24, max=110; >=20行 14.4%, >=50行 1.5% | 长表比 Infinity 多 |
| 列数/表 | p50=4; >=10列 4.1% | 优于 Infinity (1.8%) |
| **colspan / rowspan** | **27.5% / 14.7%** | **高于 hp_omni (16.8%/8.8%)** |
| 含合并单元格的含表页 | **36.3%** | 复杂结构覆盖优秀 |

结论: olmOCR (261,988 行已在盘上) 是**最省事的复杂表格源**,
先补 manifest 登记 + 用 `is_table` 筛含表页即可, 不必急着下新数据集。

### 候选新数据集 (按优先级)

| 优先级 | 数据集 | 出处 | 对本需求的适配点 |
|---|---|---|---|
| **P0** | **PubTables-v2 (full-pages + full-documents 集合)** | [arXiv 2512.10888](https://arxiv.org/html/2512.10888v1) · Kensho/S&P Global | 50 万+表格三级上下文: cropped(已用)/整页/全文; **9,172 篇全文各含至少一个跨页多部分表格 (最长跨 13 页)**, 是"整页+跨页表格"独一份的监督; 与现有 PubTables-v2 cropped 同族易接入 |
| P1 | DocGenome | [GitHub Alpha-Innovator/DocGenome](https://github.com/Alpha-Innovator/DocGenome) (arXiv 2406.11633) | 50 万篇 arXiv 论文 (650 万页), 13 类组件属性+6 类逻辑关系, 整页结构化标注, 量大, 适合整页解析主粮 |
| P1 | PubTables-1M detection 集合 | [microsoft/table-transformer](https://github.com/microsoft/table-transformer) | 46 万整页表格 bbox, 可与 structure 集合的 crop 联动合成整页解析样本 (需 join, 工程量比 PubTables-v2 大) |
| P2 | cTDaR 2019 Track B | [Zenodo 2649217](https://zenodo.org/records/2649217) | 现代+历史整页各 600 训练图, XML 单元格结构含跨行跨列; 历史手写表格来自 23 个机构, 适合作"复杂结构 booster" (量小) |
| P3 | MARMOT | Rastogi et al. | 2000 中英整页, 表格区域/列级标注 (layout 级, 单元格结构弱), 仅补充版面 |

### 不适合本需求的常见数据集 (避免误选)

- **HiTab** (microsoft/HiTab): 层级表格 QA/NLG, 非结构转写; 合并单元格只有
  core-cell 记录, 不是 HTML colspan 标签
- **PubLayNet / DocLayNet**: 版面检测, 无单元格结构
- **WikiTableQuestions / TAT-QA / FinQA / HiTab**: QA 任务, 非解析监督
- **ICDAR 2013 table competition**: 156 表, 无训练数据
- **TableBank / SciTSR / SynthTabNet**: 都是 crop 级, 不解决整页缺口
  (可作 crop 多样性补充)

### 建议动作

1. olmOCR 补 manifest 登记, 按 `is_table` 筛含表页入合训 (零下载);
2. 需要"整页+跨页多部分表格"再上 PubTables-v2 full-pages/full-documents;
3. 需要更大量整页主粮上 DocGenome; cTDaR 作小规模复杂结构补强。
