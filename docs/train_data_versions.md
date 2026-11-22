# 训练数据版本

这里按时间记录训练数据相对上一版的变化。新版本加在表的最后一行。已经写过的行不再改。

源数据 18 个数据集的行数仍在 README 第 4 节。`final_merged_v1` 的分片明细在 `manifests/final_merged_v1.json`。

跟踪表里写出版本号的只有 v1 和 v3.2。其余步骤按表从左到右排列，不补版本号。每格里的「新增」只表示该步自己写的增量，这里不做累加。

| 顺序 | 版本 | 相对上一版的变化 |
|---:|---|---|
| 1 | v1 `final_merged_v1` | 公式与表格 crop，88 分片 / 96,092 行。UniMER 49,269，MathWriting 11,462，FinTabNet 3,796，PubTables-1M 26,083，PubTables-v2 5,482。其中公式 60,731 行，表格 35,361 行。 |
| 2 |  | 修复蒸馏，改用蒸馏训练数据的公式格式。 |
| 3 |  | 1) 去重与超长过滤，约 13,042 行。2) 公式归一化：`align` 转为 `aligned`，`gather` 转为 `gathered`，避免裸的 `&` 和 `\\`。 |
| 4 | v3.2 | 表格样本总数 665。有效 HTML 表格预测 567，平均 TEDS 90.06。 |
| 5 |  | 新增 5K 整页数据（Infinity-Doc-400K）+ 2.1K 蒸馏标签。 |
| 6 |  | 新增 5K 整页数据（Infinity-Doc-400K）+ 5K 蒸馏标签。 |
| 7 |  | 新增 10K 整页数据（Infinity-Doc-400K）+ 10K 蒸馏标签。 |
| 8 |  | 清理 response 中泄露的 base64 `<img>`。 |

## v1 `final_merged_v1`

`final_merged_v1.json` 记录训练机 `/data/verl_project/distill_outputs/final_merged_v1` 的蒸馏产物，供 Q-A 兼容检查。分片文件名与源 tier2 相同，下面的行数是该归档里实际保留的行，源数据行数仍以 README 第 4 节为准。

| 数据集 | 分片 | 行数 | 内容 |
|---|---:|---:|---|
| UniMER_Dataset_vlm | 3 | 49,269 | 公式 → LaTeX |
| MathWriting-human_vlm | 3 | 11,462 | 手写公式 → LaTeX |
| FinTabNet_OTSL_HTML_vlm | 11 | 3,796 | 表格 → HTML |
| PubTables-1M_OTSL-v1.1_HTML_vlm | 48 | 26,083 | 表格 → HTML |
| PubTables-v2_cropped_table2html_vlm | 23 | 5,482 | 裁剪表格 → HTML |
| 合计 | 88 | **96,092** | 公式 60,731 + 表格 35,361 |

追加时写清这一版相对上一版改了什么。样本数和指标只记已经测到的数字。跟踪表没有版本号就留空。
