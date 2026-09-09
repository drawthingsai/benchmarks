#!/usr/bin/env python3
"""Prepare reproducible, unmodified-image quick subsets from pinned local data."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import io
from PIL import Image
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq

SOURCES = {
    "real_world_qa": ("xai-org/RealworldQA", "17e7f75e092e47169732462ea3cdfebe911105dd", 765),
    "ocr_bench": ("echo840/OCRBench", "92a54bd1384387c178d5a07140a2d85e0a3d12e1", 1000),
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def select_indices(table: pa.Table, dataset: str, seed: int) -> list[int]:
    rng = random.Random(seed)
    if dataset == "real_world_qa":
        return sorted(rng.sample(range(len(table)), 100))
    labels = table["question_type"].to_pylist()
    groups = sorted(set(labels))
    if len(groups) != 10:
        raise ValueError("Expected exactly 10 OCRBench categories")
    return sorted(i for label in groups for i in rng.sample(
        [i for i, item in enumerate(labels) if item == label], 10))


def compatible_image(data: bytes) -> bytes:
    """Convert WebP to lossless PNG without changing decoded pixels or dimensions."""
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "WEBP":
            return data
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="benchmark data root containing datasets/vision/{repo}/{revision}/raw/data")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    profile = json.loads((Path(__file__).parent / "profiles/qwen3.8-vision-quick.json").read_text())
    records = {}
    for case in profile["cases"]:
        dataset = case["dataset"]
        repo, revision, expected = SOURCES[dataset]
        source = args.data_root / "datasets/vision" / repo / revision / "raw/data"
        files = sorted(source.glob("test-*.parquet"))
        if not files:
            raise FileNotFoundError(source)
        table = pa.concat_tables([pq.read_table(path) for path in files])
        if len(table) != expected:
            raise ValueError(f"{dataset}: expected {expected} rows, got {len(table)}")
        indices = select_indices(table, dataset, 42)
        selected = table.take(indices)
        original_images = [row["bytes"] for row in selected["image"].to_pylist()]
        images = [{**row, "bytes": compatible_image(row["bytes"])}
                  for row in selected["image"].to_pylist()]
        selected = selected.set_column(selected.schema.get_field_index("image"),
                                       selected.schema.field("image"), pa.array(images, type=selected.schema.field("image").type))
        # EvalScope 1.11 requires this metadata field; decoded pixels and QA stay intact.
        if dataset == "real_world_qa" and "image_path" not in selected.column_names:
            selected = selected.append_column("image_path", pa.array([f"test/{i}" for i in indices]))
        target = output / dataset
        (target / "data").mkdir(parents=True)
        artifact = target / "data/test-00000-of-00001.parquet"
        pq.write_table(selected, artifact)
        sample_records = []
        for index, original, row in zip(indices, original_images, selected.to_pylist()):
            sample_records.append({"source_index": index,
                "image_sha256": hashlib.sha256(original).hexdigest(),
                "transport_image_sha256": hashlib.sha256(row["image"]["bytes"]).hexdigest(),
                "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                "answer": row["answer"], "category": row.get("question_type")})
        records[dataset] = {"repo": repo, "revision": revision, "source_rows": len(table),
            "source_files": [{"path": str(p.resolve()), "sha256": digest(p)} for p in files],
            "selected_rows": len(indices), "seed": 42, "samples": sample_records,
            "parquet_sha256": digest(artifact),
            "categories": dict(Counter(row["category"] for row in sample_records if row["category"]))}
        case["dataset_args"]["dataset_id"] = str(target)
        print(f"{dataset}: {len(indices)} rows", flush=True)
    manifest = output / "samples.json"
    manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    profile["suite"]["samples_sha256"] = digest(manifest)
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(output / "profile.json")


if __name__ == "__main__":
    main()
