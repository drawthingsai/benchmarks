import argparse
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark
from prepare_vision_quick import select_indices


def test_vision_identity_changes_with_same_path_weights(tmp_path):
    weights = tmp_path / "vision.gguf"
    weights.write_bytes(b"old")
    args = argparse.Namespace(mmproj=weights, image_min_tokens=64, image_max_tokens=4096)
    old = benchmark.vision_identity(args)
    weights.write_bytes(b"new")
    assert benchmark.vision_identity(args)["sha256"] != old["sha256"]
    args.image_max_tokens = 2048
    assert benchmark.vision_identity(args)["image_max_tokens"] == 2048


def test_server_receives_vision_file_and_image_limits(tmp_path):
    args = benchmark.parser().parse_args(["run", "--gguf", str(tmp_path / "lm.gguf"),
                                          "--mmproj", str(tmp_path / "vision.gguf")])
    args.ctx_size, args.slots_per_server = 32768, 4
    server = benchmark.LlamaServer(args, "test", tmp_path / "server.log", "CUDA0")
    child = Mock()
    child.poll.return_value = None
    with patch.object(benchmark.shutil, "which", return_value="llama-server"), \
         patch.object(benchmark.subprocess, "Popen", return_value=child) as spawn, \
         patch.object(benchmark, "probe"), \
         patch.object(benchmark, "terminate_process_group"):
        with server:
            cmd = spawn.call_args.args[0]
            assert cmd[cmd.index("--mmproj") + 1] == str(args.mmproj.resolve())
            assert cmd[cmd.index("--image-max-tokens") + 1] == "4096"
            assert cmd[cmd.index("--parallel") + 1] == "4"


def test_balanced_ocr_sampling_is_reproducible():
    table = pa.table({"question_type": [str(i) for i in range(10) for _ in range(50)]})
    ids = select_indices(table, "ocr_bench", 42)
    assert ids == select_indices(table, "ocr_bench", 42)
    assert len(set(ids)) == 100
    assert all(sum(i // 50 == label for i in ids) == 10 for label in range(10))


def test_realworld_sampling_has_exactly_100_distinct_rows():
    table = pa.table({"question": list(range(765))})
    ids = select_indices(table, "real_world_qa", 42)
    assert len(set(ids)) == 100
    assert ids == select_indices(table, "real_world_qa", 42)


def test_vision_profile_rejects_missing_projector(tmp_path):
    profile = Path(__file__).resolve().parents[1] / "profiles/qwen3.8-vision-quick.json"
    args = benchmark.parser().parse_args(["run", "--profile", str(profile),
                                          "--gguf", str(tmp_path / "lm.gguf"), "--dry-run"])
    with pytest.raises(benchmark.report.BenchError, match="requires --mmproj"):
        benchmark.run(args)


def test_webp_transport_preserves_decoded_pixels():
    import io
    from PIL import Image
    from prepare_vision_quick import compatible_image
    source = io.BytesIO()
    Image.new("RGB", (19, 31), (18, 119, 201)).save(source, format="WEBP")
    raw = source.getvalue()
    converted = compatible_image(raw)
    before, after = Image.open(io.BytesIO(raw)), Image.open(io.BytesIO(converted))
    assert after.format == "PNG"
    assert before.size == after.size
    assert before.mode == after.mode
    assert before.tobytes() == after.tobytes()
    assert compatible_image(converted) == converted


def test_prepared_data_mutation_invalidates_profile(tmp_path):
    import hashlib
    import json
    artifact = tmp_path / "dataset/data/test-00000-of-00001.parquet"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"original")
    manifest = tmp_path / "samples.json"
    manifest.write_text(json.dumps({"real_world_qa": {
        "parquet_sha256": hashlib.sha256(b"original").hexdigest()}}))
    profile = {"suite": {"samples_sha256": benchmark.sha256_file(manifest)},
               "cases": [{"dataset": "real_world_qa", "dataset_args": {
                   "dataset_id": str(tmp_path / "dataset")}}]}
    benchmark.validate_vision_samples(profile, tmp_path / "profile.json")
    artifact.write_bytes(b"modified")
    with pytest.raises(benchmark.report.BenchError, match="input data fingerprint changed"):
        benchmark.validate_vision_samples(profile, tmp_path / "profile.json")
