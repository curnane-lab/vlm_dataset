#!/usr/bin/env python3
import argparse
import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq


ENV_NAMES = ("aligned", "align", "align*", "equation", "equation*",
             "gather", "gather*", "array")
SAFE_ENV_RE = r"(?:aligned|array)"
ENV_RE = r"(?:aligned|align\*?|equation\*?|gather\*?|array)"
SEGMENT_RE = re.compile(
    rf"^[ \t]*\\begin{{({ENV_RE})}}\s*(\{{.*?\}})?(.*)\\end{{\1}}[ \t]*$",
    re.DOTALL,
)
OPEN_RE = re.compile(rf"\\begin{{({ENV_RE})}}")


def strip_environment(content, env_name):
    content = content.strip()
    if content.startswith("{") and content.endswith("}"):
        body = content[1:-1]
    elif content.startswith("[") and content.endswith("]"):
        body = content[1:-1]
    else:
        body = content
    return body


def normalize_environment(content, env_name):
    """Preserve alignment structure while producing a delimitable body."""
    content = strip_environment(content, env_name)
    base = env_name.rstrip("*")
    if base == "align":
        return f"\\begin{{aligned}}{content}\\end{{aligned}}"
    if base == "gather":
        return f"\\begin{{gathered}}{content}\\end{{gathered}}"
    return content


def has_complete_environment_delimiters(text):
    for match in OPEN_RE.finditer(text):
        env_name = match.group(1)
        if text.count(f"\\begin{{{env_name}}}") != text.count(
                f"\\end{{{env_name}}}"):
            return False
    return True


def normalize_paragraph(paragraph):
    match = SEGMENT_RE.fullmatch(paragraph)
    if not match:
        return paragraph, False
    env_name = match.group(1)
    content = match.group(3)
    if env_name in ("align", "align*", "equation", "equation*",
                    "gather", "gather*"):
        content = normalize_environment(content, env_name)
    if env_name.rstrip("*") not in ("align", "gather") and (
            "\\begin{" in content or "\\end{" in content):
        return paragraph, False
    return f"\n$$\n{content.strip()}\n$$\n", True


def normalize_answer(answer):
    text = answer.strip()
    if not text:
        return answer, {"changed": False, "dropped": False}
    if not has_complete_environment_delimiters(text):
        return "", {"changed": False, "dropped": True}
    paragraphs = re.split(r"(\n\s*\n)", text)
    changed = False
    for index in range(0, len(paragraphs), 2):
        paragraphs[index], result = normalize_paragraph(paragraphs[index])
        changed = changed or result
    if changed:
        text = "".join(paragraphs)
    return text, {"changed": changed, "dropped": False}


def infer_route(dataset_name):
    name = dataset_name.lower()
    if any(token in name for token in ("unimer", "mathwriting", "formula")):
        return "formula"
    return "other"


def process_parquet(source_path, destination_path, route):
    source = pq.ParquetFile(source_path)
    stats = {"rows": 0, "changed": 0, "dropped": 0, "kept": 0}
    with pq.ParquetWriter(destination_path, source.schema_arrow) as writer:
        for batch in source.iter_batches(batch_size=100):
            rows = batch.to_pylist()
            kept_rows = []
            for row in rows:
                stats["rows"] += 1
                messages = [dict(item) for item in row["messages"]]
                if not messages or messages[-1]["role"] != "assistant":
                    stats["kept"] += 1
                    kept_rows.append(row)
                    continue
                if route != "formula":
                    stats["kept"] += 1
                    kept_rows.append(row)
                    continue
                answer, result = normalize_answer(messages[-1]["content"])
                if result["dropped"]:
                    stats["dropped"] += 1
                    continue
                if result["changed"]:
                    stats["changed"] += 1
                    messages[-1]["content"] = answer
                    row["messages"] = messages
                stats["kept"] += 1
                kept_rows.append(row)
            if kept_rows:
                writer.write_table(pa.Table.from_pylist(
                    kept_rows, schema=source.schema_arrow))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    with open(os.path.join(args.src_root, "dataset_manifest.json"),
              encoding="utf-8") as handle:
        manifest = json.load(handle)

    output_manifest = {key: value for key, value in manifest.items()
                       if key != "datasets"}
    output_manifest["data_root"] = args.dst_root
    output_manifest["comment"] = (
        "final_merged_v1 copy with independent complete LaTeX environments "
        "wrapped in display delimiters; truncated formula rows dropped"
    )
    output_manifest["datasets"] = []
    report = []

    for dataset in manifest["datasets"]:
        name = dataset["name"]
        if args.datasets and name not in args.datasets:
            continue
        route = infer_route(name)
        files = []
        total = {"rows": 0, "changed": 0, "dropped": 0, "kept": 0}
        for file_info in dataset["files"]:
            relative = file_info["path"].replace("\\", "/")
            if relative.startswith("./"):
                relative = relative[2:]
            source = os.path.join(args.src_root, relative)
            destination = os.path.join(args.dst_root, relative)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            stats = process_parquet(source, destination, route)
            for key in total:
                total[key] += stats[key]
            files.append({**file_info, "path": relative,
                          "rows": stats["kept"]})
        output_manifest["datasets"].append({
            **dataset,
            "total_rows": sum(item["rows"] for item in files),
            "num_files": len(files),
            "data_root": args.dst_root,
            "files": files,
        })
        report.append({"dataset": name, "route": route, **total})

    with open(os.path.join(args.dst_root, "dataset_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(output_manifest, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.dst_root, "normalize_report.json"), "w",
              encoding="utf-8") as handle:
        json.dump({"source": args.src_root, "output": args.dst_root,
                   "datasets": report}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
