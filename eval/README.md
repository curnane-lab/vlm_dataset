# eval — SFT 后模型精度

这里只放 **SFT 之后** 在 OmniDocBench 上的成绩摘要。教师 MinerU 的环境复测成绩仍在 `docs/omnidocbench_npu_eval_guide.md` 第 1 节。评测怎么跑也以那份指南为准。

每次评测一个子目录，目录名用 `YYYYMMDD_模型_训练数据`，例如 `20261031_qwen3.6-35b-a3b_final_merged_v1`。

```
eval/
├── README.md
├── meta.template.json
└── 20261031_qwen3.6-35b-a3b_final_merged_v1/
    ├── meta.json          # 从 meta.template.json 复制后填
    └── run_summary.json   # 评测仓 ./result/<run_name>_run_summary.json 原样拷贝
```

`run_summary.json` 保持评测仓原文，不手改数字。`meta.json` 写清模型路径、checkpoint、训练数据、行数和预测目录，方便以后对照是哪一次 SFT。

不要提交页面预测 markdown、页面图片，以及 CDM 渲染出的 PDF/PNG。那些留在训练机的评测目录。
