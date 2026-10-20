# -*- coding: utf-8 -*-
"""生成符合 qwen_vl schema 的迷你蒸馏测试数据集 + manifest。"""
import io
import json
import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = "/data/mineru/distill/mini_vlm"


def make_image(lines, size=(800, 200)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.load_default(size=24)
    y = 20
    for ln in lines:
        d.text((30, y), ln, fill="black", font=f)
        y += 40
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def row(user, answer, image_bytes, extra_rounds=None):
    """qwen_vl 行: messages + images。"""
    msgs = [{"role": "user", "content": user},
            {"role": "assistant", "content": answer}]
    if extra_rounds:
        msgs.extend(extra_rounds)
    return {"messages": msgs,
            "images": [{"bytes": image_bytes, "path": "synthetic.png"}]}


formula_img = make_image(["E = mc^2"])
text_img = make_image(["The quick brown fox", "jumps over the lazy dog."])
table_img = make_image(["Name  Age", "Tom   5", "Amy   7"])

ds1 = [
    row("<image>\n请把图片中的公式转成 LaTeX。", "PLACEHOLDER", formula_img),
    row("<image>\nTranscribe the text in the image.", "PLACEHOLDER", text_img),
    row("<image>\nConvert the table in the image to HTML.", "PLACEHOLDER", table_img),
]
ds2 = [
    row("<image>\nWhat equation is shown?", "PLACEHOLDER", formula_img,
        extra_rounds=[{"role": "user", "content": "What does the c stand for?"},
                      {"role": "assistant", "content": "PLACEHOLDER2"}]),
    row("<image>\nRead the first line only.", "PLACEHOLDER", text_img),
]

specs = {"mini_formula_vlm": ds1, "mini_docqa_vlm": ds2}
manifest = {"datasets": []}
for name, rows in specs.items():
    sub = os.path.join(ROOT, f"{name}", "vlm_parquet")
    os.makedirs(sub, exist_ok=True)
    # 每个数据集 2 个分片, 每片 2 行
    files = []
    for si in range(2):
        part = rows[si * 1: si * 1 + 2] if si == 0 else rows[2:3] or rows[:1]
        if not part:
            part = rows[:1]
        rel = f"{name}/vlm_parquet/train-0000{si}-of-00002.parquet"
        pd.DataFrame(part).to_parquet(os.path.join(ROOT, rel), index=False)
        files.append({"path": rel, "rows": len(part)})
    manifest["datasets"].append(
        {"name": name, "num_files": len(files), "files": files})

with open(os.path.join(ROOT, "dataset_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print("mini 数据集已生成:", ROOT)
for ds in manifest["datasets"]:
    print(" ", ds["name"], sum(f["rows"] for f in ds["files"]), "行")
