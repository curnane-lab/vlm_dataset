#!/usr/bin/env python3
"""Parse training log and plot loss/grad_norm/lr/mfu curves using Pillow."""

import argparse
import math
import re
import sys

from PIL import Image, ImageDraw, ImageFont


def parse_log(log_path):
    metrics = {}
    pattern = re.compile(
        r"step:(\d+).*?"
        r"train/loss:([\d.eE+-]+).*?"
        r"train/grad_norm:([\d.eE+-]+).*?"
        r"train/lr:([\d.eE+-]+).*?"
        r"train/mfu:([\d.eE+-]+)"
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                step = int(m.group(1))
                metrics[step] = {
                    "loss": float(m.group(2)),
                    "grad_norm": float(m.group(3)),
                    "lr": float(m.group(4)),
                    "mfu": float(m.group(5)),
                }
    return dict(sorted(metrics.items()))


def get_font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_curve(draw, data, x0, y0, w, h, color, label, font, log_scale=False):
    if not data:
        return
    values = [v for _, v in data]
    if log_scale:
        values = [math.log10(max(v, 1e-10)) for v in values]
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-12:
        margin = max(abs(vmin) * 0.1, 1e-12)
        vmin -= margin
        vmax += margin
    steps = [s for s, _ in data]
    smin, smax = min(steps), max(steps)
    if smax - smin < 1:
        smax = smin + 1

    # Axes
    draw.line([(x0, y0), (x0, y0 + h), (x0 + w, y0 + h)], fill=(200, 200, 200), width=1)

    # Grid lines (4 horizontal)
    for i in range(1, 5):
        gy = y0 + h * i // 5
        draw.line([(x0, gy), (x0 + w, gy)], fill=(230, 230, 230), width=1)

    # Curve
    points = []
    for i, (step, _) in enumerate(data):
        px = x0 + int(w * (step - smin) / (smax - smin))
        py = y0 + h - int(h * (values[i] - vmin) / (vmax - vmin))
        points.append((px, py))
    if len(points) > 1:
        draw.line(points, fill=color, width=2)
    for px, py in points:
        draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=color)

    # Y-axis labels
    for i in range(5):
        val = vmin + (vmax - vmin) * (5 - i) / 5
        gy = y0 + h * i // 5
        if log_scale:
            label_text = f"{10**val:.2e}"
        else:
            if abs(val) < 1e-3:
                label_text = f"{val:.2e}"
            elif abs(val) < 1:
                label_text = f"{val:.4f}"
            else:
                label_text = f"{val:.2f}"
        draw.text((x0 - 8 - len(label_text) * 7, gy - 7), label_text, fill=(100, 100, 100), font=font)

    # X-axis labels
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        step = int(smin + (smax - smin) * frac)
        px = x0 + int(w * frac)
        draw.text((px - 10, y0 + h + 5), str(step), fill=(100, 100, 100), font=font)

    # Title
    draw.text((x0 + w // 2 - len(label) * 4, y0 - 20), label, fill=(0, 0, 0), font=font)


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from verl SFT log")
    parser.add_argument("--log", required=True, help="Path to training log file")
    parser.add_argument("--output", required=True, help="Output image path (PNG)")
    args = parser.parse_args()

    metrics = parse_log(args.log)
    if not metrics:
        print("No metrics found in log")
        sys.exit(1)

    print(f"Parsed {len(metrics)} steps from {args.log}")

    steps = sorted(metrics.keys())
    curves = {
        "loss": [(s, metrics[s]["loss"]) for s in steps],
        "grad_norm": [(s, metrics[s]["grad_norm"]) for s in steps],
        "lr": [(s, metrics[s]["lr"]) for s in steps],
        "mfu": [(s, metrics[s]["mfu"]) for s in steps],
    }

    W, H = 1200, 900
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = get_font(13)
    title_font = get_font(16)

    draw.text((W // 2 - 150, 10), f"Training Metrics ({len(steps)} steps)", fill=(0, 0, 0), font=title_font)

    margin_x, margin_y = 80, 50
    panel_w = (W - 3 * margin_x) // 2
    panel_h = (H - 3 * margin_y) // 2 - 20

    configs = [
        ("loss", (255, 60, 60), False),
        ("grad_norm", (60, 60, 255), False),
        ("lr", (60, 160, 60), False),
        ("mfu", (200, 120, 0), False),
    ]

    positions = [
        (margin_x, margin_y),
        (margin_x * 2 + panel_w, margin_y),
        (margin_x, margin_y * 2 + panel_h),
        (margin_x * 2 + panel_w, margin_y * 2 + panel_h),
    ]

    for (name, color, log_scale), (x0, y0) in zip(configs, positions):
        draw_curve(draw, curves[name], x0, y0, panel_w, panel_h, color, f"train/{name}", font, log_scale)

    img.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
