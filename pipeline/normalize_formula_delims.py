#!/usr/bin/env python3
"""Copy a merged export and normalize formula answers.

Run this on answers from before delimiter normalization (v1 or the teacher
output). Do not run it on v2/v3 rows whose alignment shells were already
removed: ``$$x&=1\\\\y&=2$$`` no longer contains the environment name, and
this script will not guess one.

aligned/array keep the environment and its argument. align becomes aligned.
gather becomes gathered. equation drops only its shell. Any other structure
is left unchanged and listed in normalize_rejected.jsonl. Rows are never
deleted. ``changed`` counts rewrites, not confirmed pollution.
"""

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from formula_display import normalize_math
except ImportError:  # imported as pipeline.normalize_formula_delims
    from pipeline.formula_display import normalize_math


def infer_route(dataset_name):
    name = dataset_name.lower()
    if any(token in name for token in ("unimer", "mathwriting", "formula")):
        return "formula"
    return "other"


def _image_paths(row):
    paths = []
    for image in row.get("images") or []:
        if isinstance(image, dict) and image.get("path"):
            paths.append(image["path"])
    return paths


def _record(relative, row_index, row, original, result):
    return {
        "sample_id": "%s#%d" % (relative, row_index),
        "source": relative,
        "row_index": row_index,
        "image_paths": _image_paths(row),
        "original": original,
        "new": None if result.action == "review" else result.text,
        "change_type": result.change_type,
        "reason": result.reason,
    }


def process_parquet(source_path, destination_path, route, relative,
                    changes_fp, rejected_fp):
    source = pq.ParquetFile(source_path)
    stats = {
        "rows": 0,
        "unchanged": 0,
        "rewritten": 0,
        "rejected": 0,
        "by_change_type": {},
        "by_reason": {},
    }
    with pq.ParquetWriter(destination_path, source.schema_arrow) as writer:
        row_index = 0
        for batch in source.iter_batches(batch_size=100):
            rows = batch.to_pylist()
            for row in rows:
                stats["rows"] += 1
                messages = [dict(item) for item in (row.get("messages") or [])]
                if (route != "formula" or not messages
                        or messages[-1].get("role") != "assistant"):
                    stats["unchanged"] += 1
                    row_index += 1
                    continue
                original = messages[-1].get("content") or ""
                result = normalize_math(original)
                record = _record(relative, row_index, row, original, result)
                if result.action == "rewritten":
                    stats["rewritten"] += 1
                    stats["by_change_type"][result.change_type] = (
                        stats["by_change_type"].get(result.change_type, 0) + 1)
                    messages[-1]["content"] = result.text
                    row["messages"] = messages
                    changes_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                elif result.action == "review":
                    stats["rejected"] += 1
                    stats["by_reason"][result.reason] = (
                        stats["by_reason"].get(result.reason, 0) + 1)
                    rejected_fp.write(
                        json.dumps(record, ensure_ascii=False) + "\n")
                else:
                    stats["unchanged"] += 1
                row_index += 1
            if rows:
                writer.write_table(pa.Table.from_pylist(
                    rows, schema=source.schema_arrow))
    return stats


def _add_counts(total, stats):
    for key in ("rows", "unchanged", "rewritten", "rejected"):
        total[key] += stats[key]
    for bucket, count in stats["by_change_type"].items():
        total["by_change_type"][bucket] = (
            total["by_change_type"].get(bucket, 0) + count)
    for reason, count in stats["by_reason"].items():
        total["by_reason"][reason] = total["by_reason"].get(reason, 0) + count


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
        "Formula answers normalized without dropping rows. "
        "aligned/array keep their environment and arguments; "
        "align becomes aligned; gather becomes gathered; "
        "equation drops only its shell. "
        "Unbalanced, unknown, or already-stripped rows are unchanged and "
        "listed in normalize_rejected.jsonl. "
        "Run on pre-normalization answers only."
    )
    output_manifest["datasets"] = []
    os.makedirs(args.dst_root, exist_ok=True)
    changes_path = os.path.join(args.dst_root, "normalize_changes.jsonl")
    rejected_path = os.path.join(args.dst_root, "normalize_rejected.jsonl")
    report = []

    with open(changes_path, "w", encoding="utf-8") as changes_fp, \
            open(rejected_path, "w", encoding="utf-8") as rejected_fp:
        for dataset in manifest["datasets"]:
            name = dataset["name"]
            if args.datasets and name not in args.datasets:
                continue
            route = infer_route(name)
            files = []
            total = {
                "rows": 0, "unchanged": 0, "rewritten": 0, "rejected": 0,
                "by_change_type": {}, "by_reason": {},
            }
            for file_info in dataset["files"]:
                relative = file_info["path"].replace("\\", "/")
                if relative.startswith("./"):
                    relative = relative[2:]
                source = os.path.join(args.src_root, relative)
                destination = os.path.join(args.dst_root, relative)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                stats = process_parquet(
                    source, destination, route, relative,
                    changes_fp, rejected_fp)
                _add_counts(total, stats)
                files.append({**file_info, "path": relative,
                              "rows": stats["rows"]})
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
    summary = {
        "source": args.src_root,
        "output": args.dst_root,
        "rows_deleted": 0,
        "note": (
            "rewritten is not a pollution count. "
            "rejected rows stay in the parquet with their original answer. "
            "Join normalize_rejected.jsonl and normalize_changes.jsonl "
            "to earlier exports by sample_id or image_paths."
        ),
        "datasets": report,
    }
    with open(os.path.join(args.dst_root, "normalize_report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
