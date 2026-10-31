#!/usr/bin/env python3
"""Split manifest-driven VLM parquet datasets into train/val copies.

Reads a vlm_dataset manifest, streams each parquet shard, deterministically assigns
rows to train/val with crc32(relative_path#row_id), writes split-specific parquet
files, and emits matching manifests plus verl file-list JSON files.
"""

import argparse
import glob
import hashlib
import json
import os
import pyarrow.parquet as pq
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def norm_rel_path(path: str) -> str:
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


def assign_split(rel_path: str, row_id: int, val_ratio: int) -> str:
    digest = hashlib.md5(f"{rel_path}#{row_id}".encode("utf-8")).digest()
    return "val" if int.from_bytes(digest[:4], "big") % 10000 < val_ratio else "train"


class ShardSplitter:
    def __init__(self, chunk_rows: int):
        self.chunk_rows = chunk_rows

    def split(self, src: Path, outputs: dict[str, Path]):
        parquet_file = pq.ParquetFile(src)
        schema = parquet_file.schema_arrow
        total_rows = parquet_file.metadata.num_rows
        counts = {"train": 0, "val": 0}
        row_id = 0
        relative_path = outputs["__rel__"]
        val_ratio = outputs["__val_ratio__"]
        try:
            for split, output_path in outputs.items():
                if split.startswith("__"):
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
            tables_by_split = {split: [] for split in ("train", "val")}
            for batch in parquet_file.iter_batches(batch_size=self.chunk_rows):
                for offset in range(batch.num_rows):
                    row = batch.slice(offset, 1)
                    split = assign_split(relative_path, row_id, val_ratio)
                    tables_by_split[split].append(pa.Table.from_batches([row], schema=schema))
                    row_id += 1
                    counts[split] += 1
            for split, rows in tables_by_split.items():
                if rows:
                    writer = pq.ParquetWriter(outputs[split], schema)
                    writer.write_table(pa.concat_tables(rows))
                    writer.close()
        finally:
            pass
        if row_id != total_rows:
            raise RuntimeError(f"row count mismatch for {src}: walked={row_id}, footer={total_rows}")
        return total_rows, counts


def write_manifest(source_manifest: dict, data_root: Path, split_counts: dict, output_manifest_path: Path):
    manifest = {key: value for key, value in source_manifest.items() if key != "datasets"}
    manifest["data_root"] = str(data_root)
    manifest["split"] = output_manifest_path.stem.replace("dataset_manifest_", "")
    manifest["datasets"] = []
    for source_dataset in source_manifest["datasets"]:
        name = source_dataset["name"]
        files = split_counts.get(name, {}).get("files", {}).get(manifest["split"], [])
        if not files:
            continue
        dataset = {key: value for key, value in source_dataset.items() if key not in ("files", "total_rows", "num_files")}
        dataset["files"] = files
        dataset["num_files"] = len(files)
        dataset["total_rows"] = sum(item["rows"] for item in files)
        manifest["datasets"].append(dataset)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def split_output_relative(relative_path: str, split: str) -> str:
    """train/val 文件名必须不同。只改文件名, 不改目录。"""
    relative = Path(norm_rel_path(relative_path))
    name = relative.name
    if name.startswith("train-"):
        name = f"{split}-{name[len('train-'):]}"
    elif name.startswith(f"{split}-"):
        pass
    else:
        name = f"{split}-{name}"
    return (relative.parent / name).as_posix()


def load_source_manifest(manifest: Path, data_root: Path) -> dict:
    """读取真正的 dataset manifest。

    dataset_manifest.json 以及内容为 {datasets: [...]} 的 jsonl 会按清单切分。
    路径以 .jsonl 结尾且不是这份清单、或文件不存在时, 才扫描 data-root 下全部 parquet。
    """
    if manifest.exists():
        text = manifest.read_text(encoding="utf-8").strip()
        if text:
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and isinstance(obj.get("datasets"), list):
                print(f"[INFO] Using manifest {manifest} "
                      f"({len(obj['datasets'])} datasets)")
                return obj
        if manifest.suffix != ".jsonl":
            raise SystemExit(f"manifest is not a dataset manifest: {manifest}")
    print(f"[INFO] Building flat-parquet manifest from {data_root}")
    return build_flat_manifest(data_root)


def build_flat_manifest(data_root: Path) -> dict:
    """Auto-build a manifest from all *.parquet files under data_root."""
    parquet_files = sorted(glob.glob(str(data_root / "**" / "*.parquet"), recursive=True))
    if not parquet_files:
        raise SystemExit(f"no parquet files found under {data_root}")
    datasets = []
    for f in parquet_files:
        path = Path(f)
        name = path.stem
        rows = pq.ParquetFile(path).metadata.num_rows
        datasets.append({
            "name": name,
            "schema_adapter": "qwen_vl",
            "weight": 1.0,
            "total_rows": rows,
            "num_files": 1,
            "files": [{"path": path.relative_to(data_root).as_posix(), "rows": rows, "schema_adapter": "qwen_vl"}],
        })
    return {"version": 1, "data_root": str(data_root), "datasets": datasets}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--chunk-rows", type=int, default=200)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio must be in (0, 1)")
    if args.output_root.resolve() == args.data_root.resolve():
        raise SystemExit("--output-root must differ from --data-root")
    source_manifest = load_source_manifest(args.manifest, args.data_root)
    for dataset in source_manifest["datasets"]:
        print(f"[manifest] {dataset.get('name')}: "
              f"rows={dataset.get('total_rows')} "
              f"files={dataset.get('num_files', len(dataset.get('files') or []))} "
              f"weight={dataset.get('weight', 1)}")
    val_ratio = round(args.val_ratio * 10000)
    splitter = ShardSplitter(args.chunk_rows)

    split_counts = {}
    processed_files = 0
    processed_rows = 0
    for source_dataset in source_manifest["datasets"]:
        name = source_dataset["name"]
        if args.datasets and name not in args.datasets:
            continue
        files = source_dataset.get("files", [])
        if args.max_files:
            files = files[: args.max_files]
        split_counts[name] = {"train": 0, "val": 0, "files": {"train": [], "val": []}}
        print(f"=== {name}: {len(files)} shards ===", flush=True)
        for file_info in files:
            relative_path = norm_rel_path(file_info["path"])
            source_path = args.data_root / relative_path
            if not source_path.is_file():
                print(f"[skip] missing: {source_path}", file=sys.stderr)
                continue
            expected_rows = file_info.get("rows")
            split_paths = {}
            for split in ("train", "val"):
                output_relative = split_output_relative(relative_path, split)
                output_path = args.output_root / output_relative
                if output_path.exists() and not args.overwrite:
                    raise SystemExit(f"output exists (use --overwrite): {output_path}")
                split_paths[split] = output_path
            if split_paths["train"].resolve() == split_paths["val"].resolve():
                raise SystemExit(f"train/val output paths collide: {split_paths['train']}")
            split_paths["__rel__"] = relative_path
            split_paths["__val_ratio__"] = val_ratio
            total_rows, counts = splitter.split(source_path, split_paths)
            if expected_rows is not None and total_rows != expected_rows:
                raise SystemExit(f"manifest mismatch for {relative_path}: manifest={expected_rows}, actual={total_rows}")
            for split in ("train", "val"):
                output_path = split_paths[split]
                rows = counts[split]
                if rows == 0:
                    output_path.unlink(missing_ok=True)
                    continue
                split_counts[name][split] += rows
                split_counts[name]["files"][split].append(
                    {
                        "path": output_path.relative_to(args.output_root).as_posix(),
                        "rows": rows,
                        "schema_adapter": file_info.get("schema_adapter", "qwen_vl"),
                    }
                )
            processed_files += 1
            processed_rows += total_rows
            print(
                f"  {relative_path}: total={total_rows}, train={counts['train']}, val={counts['val']}",
                flush=True,
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    train_manifest = write_manifest(source_manifest, args.output_root, split_counts, args.output_root / "dataset_manifest_train.json")
    val_manifest = write_manifest(source_manifest, args.output_root, split_counts, args.output_root / "dataset_manifest_val.json")

    for split, manifest in (("train", train_manifest), ("val", val_manifest)):
        files = [item["path"] for dataset in manifest["datasets"] for item in dataset["files"]]
        with (args.output_root / f"{split}_files.json").open("w", encoding="utf-8") as handle:
            json.dump([str(args.output_root / path) for path in files], handle, indent=2)
            handle.write("\n")

    total_train = sum(dataset["total_rows"] for dataset in train_manifest["datasets"])
    total_val = sum(dataset["total_rows"] for dataset in val_manifest["datasets"])
    print(
        f"done: files={processed_files}, rows={processed_rows}, train={total_train}, val={total_val}, "
        f"actual_ratio={total_val / max(total_train + total_val, 1):.4%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
