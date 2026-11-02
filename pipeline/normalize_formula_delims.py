#!/usr/bin/env python3
"""Copy a merged export and normalize formula answers.

Run this on answers from before delimiter normalization (v1 or the teacher
output). Do not run it on v2/v3 rows whose alignment shells were already
removed: ``$$x&=1\\\\y&=2$$`` no longer contains the environment name, and
this script will not guess one.

aligned/array keep the environment and its argument. align becomes aligned.
gather becomes gathered. equation drops only its shell. A formula that is
one ``\\(...\\)`` or ``$...$`` pair is unwrapped, then normalized.
``normalize_rejected.jsonl`` means automatic repair failed. It does not mean
those rows were removed from training input.

--profile audit (default) keeps every row. --profile trainable omits review
rows from the parquet and manifest. Source and destination roots must differ;
each output shard is written to a temporary file and then renamed.
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


def output_roots_collide(src_root, dst_root):
    """True when both arguments are the same directory after resolving links."""
    return os.path.realpath(os.path.abspath(src_root)) == os.path.realpath(
        os.path.abspath(dst_root))


def include_in_output(action, profile):
    """Audit keeps review rows. Trainable omits them."""
    if profile == "trainable" and action == "review":
        return False
    return True


def process_parquet(source_path, destination_path, route, relative,
                    changes_fp, rejected_fp, profile):
    if output_roots_collide(source_path, destination_path):
        raise SystemExit("refusing to write %s onto the source file" % source_path)
    source = pq.ParquetFile(source_path)
    stats = {
        "rows": 0,
        "unchanged": 0,
        "rewritten": 0,
        "rejected": 0,
        "excluded": 0,
        "written": 0,
        "by_change_type": {},
        "by_reason": {},
    }
    temporary = destination_path + ".normalize_tmp"
    try:
        with pq.ParquetWriter(temporary, source.schema_arrow) as writer:
            row_index = 0
            for batch in source.iter_batches(batch_size=100):
                rows = batch.to_pylist()
                kept = []
                for row in rows:
                    stats["rows"] += 1
                    messages = [dict(item) for item in (row.get("messages") or [])]
                    if (route != "formula" or not messages
                            or messages[-1].get("role") != "assistant"):
                        stats["unchanged"] += 1
                        stats["written"] += 1
                        kept.append(row)
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
                        changes_fp.write(
                            json.dumps(record, ensure_ascii=False) + "\n")
                    elif result.action == "review":
                        stats["rejected"] += 1
                        stats["by_reason"][result.reason] = (
                            stats["by_reason"].get(result.reason, 0) + 1)
                        rejected_fp.write(
                            json.dumps(record, ensure_ascii=False) + "\n")
                        if not include_in_output(result.action, profile):
                            stats["excluded"] += 1
                            row_index += 1
                            continue
                    else:
                        stats["unchanged"] += 1
                    stats["written"] += 1
                    kept.append(row)
                    row_index += 1
                if kept:
                    writer.write_table(pa.Table.from_pylist(
                        kept, schema=source.schema_arrow))
        os.replace(temporary, destination_path)
    except Exception:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return stats


def _add_counts(total, stats):
    for key in ("rows", "unchanged", "rewritten", "rejected", "excluded", "written"):
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
    parser.add_argument(
        "--profile", choices=("audit", "trainable"), default="audit",
        help="audit keeps review rows in the parquet; trainable omits them. "
             "normalize_rejected.jsonl is written for both.")
    args = parser.parse_args()
    if output_roots_collide(args.src_root, args.dst_root):
        parser.error("--dst-root must be a different directory from --src-root")

    with open(os.path.join(args.src_root, "dataset_manifest.json"),
              encoding="utf-8") as handle:
        manifest = json.load(handle)

    output_manifest = {key: value for key, value in manifest.items()
                       if key != "datasets"}
    output_manifest["data_root"] = args.dst_root
    output_manifest["profile"] = args.profile
    if args.profile == "trainable":
        profile_note = (
            "Trainable export: review rows are omitted from these parquet "
            "files and from this manifest. They are listed in "
            "normalize_rejected.jsonl."
        )
    else:
        profile_note = (
            "Audit export: review rows are still in these parquet files. "
            "normalize_rejected.jsonl does not mean they were excluded "
            "from training."
        )
    output_manifest["comment"] = (
        "Formula answers normalized from pre-normalization text. "
        "aligned/array keep their environment and arguments; "
        "align becomes aligned; gather becomes gathered; "
        "equation drops only its shell; one outer inline pair is unwrapped. "
        + profile_note
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
                "excluded": 0, "written": 0,
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
                    changes_fp, rejected_fp, args.profile)
                _add_counts(total, stats)
                files.append({**file_info, "path": relative,
                              "rows": stats["written"],
                              "source_rows": stats["rows"]})
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
        "profile": args.profile,
        "rows_deleted_from_source": 0,
        "note": (
            "rewritten is not a pollution count. "
            "normalize_rejected.jsonl lists rows that were not auto-repaired. "
            "With --profile audit those rows remain in the parquet. "
            "With --profile trainable they are omitted from the parquet and "
            "the manifest row counts. Source files are not modified."
        ),
        "datasets": report,
    }
    with open(os.path.join(args.dst_root, "normalize_report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
