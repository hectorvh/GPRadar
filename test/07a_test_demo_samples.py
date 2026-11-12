"""Checks for the demo-sample export on tiny fake inputs (no real data)."""

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

demo = importlib.import_module("07a_export_demo_samples")
prep = importlib.import_module("06a_prepare_real_yolo")
split_mod = importlib.import_module("04b_split_mendeley")

ROWS = [
    ("utility_0001", "Utilities/1.jpg", "utility", "train"),
    ("utility_0002", "Utilities/2.jpg", "utility", "test"),
    ("void_0001", "cavities/1.jpg", "void", "train"),
    ("intact_0001", "intact/1.jpg", "intact", "test"),
]


def _fake_split(tmp_path):
    """Split CSV, matching lock file and labels for ROWS."""

    split = tmp_path / "split.csv"
    with split.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "path", "class", "group_id", "split", "md5"])
        for i, (image_id, path, cls, part) in enumerate(ROWS):
            writer.writerow([image_id, path, cls, f"g{i}", part, "0" * 32])
    lock = tmp_path / "lock.txt"
    lock.write_text(split_mod.test_lock_hash([r[0] for r in ROWS if r[3] == "test"]) + "\n")
    labels = tmp_path / "labels"
    labels.mkdir()
    for image_id in ("utility_0001", "utility_0002", "void_0001"):
        (labels / f"{image_id}.txt").write_text("0 0.5 0.5 0.4 0.3\n")
    return split, lock, labels


def test_test_ids_are_refused(tmp_path):
    split, lock, labels = _fake_split(tmp_path)
    train, test_ids = demo.checked_train_rows(split, lock)
    assert {row["id"] for row in train} == {"utility_0001", "void_0001"}
    # Automatic choice never returns a test id.
    row, _boxes = demo.choose_mendeley(train, test_ids, labels, "utility")
    assert row["id"] == "utility_0001"
    # An override with a test id is refused.
    with pytest.raises(prep.TestSplitError):
        demo.choose_mendeley(train, test_ids, labels, "utility", override="utility_0002")
    # A wrong lock stops everything.
    lock.write_text("0" * 64 + "\n")
    with pytest.raises(RuntimeError, match="test lock mismatch"):
        demo.checked_train_rows(split, lock)


def test_manifest_fields_are_complete():
    entries = [
        demo.manifest_entry("img", "Image", "image", "img.jpg", "One real image."),
        demo.manifest_entry("sim", "Scene", "gprmax", "sim.npz", "One scene."),
    ]
    assert demo.check_manifest(entries)
    for entry in entries:
        assert set(entry) == set(demo.MANIFEST_FIELDS)
    assert entries[0]["default_mode"] == "detector"
    assert entries[1]["default_mode"] == "full"
    assert "CC BY 4.0" in entries[0]["citation"]
    assert "gprMax" in entries[1]["citation"]
    broken = dict(entries[0], description="")
    with pytest.raises(ValueError, match="misses"):
        demo.check_manifest([broken])
    with pytest.raises(ValueError, match="unique"):
        demo.check_manifest([entries[0], entries[0]])
    with pytest.raises(ValueError):
        demo.manifest_entry("x", "X", "video", "x.mp4", "No.")


def test_npz_round_trip_keeps_dt_and_frequency(tmp_path):
    data = np.random.default_rng(0).normal(size=(50, 7))
    scene = {"seed": 185, "frequency_hz": 4e8}
    path = tmp_path / "scene.npz"
    size = demo.write_scene_npz(
        path, data, 1.4723e-11, 0.05, 4e8, json.dumps(scene), np.zeros((224, 224), np.uint8)
    )
    assert 0 < size < demo.MAX_NPZ_BYTES
    back = demo.read_scene_npz(path)
    assert back["dt"] == 1.4723e-11
    assert back["frequency"] == 4e8
    assert back["trace_spacing"] == 0.05
    assert back["data"].dtype == np.float32
    assert np.allclose(back["data"], data, atol=1e-6)
    assert back["scene"] == scene
    assert back["processed_u8_224"].shape == (224, 224)
