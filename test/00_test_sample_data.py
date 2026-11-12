"""Checks on the committed sample data in data/sample/."""

import csv
import hashlib
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

prep = importlib.import_module("06a_prepare_real_yolo")

SAMPLE = ROOT / "data" / "sample"
pytestmark = pytest.mark.skipif(
    not (SAMPLE / "CITATION.txt").is_file(), reason="data/sample/ has not been built"
)


def _samples():
    with (SAMPLE / "mendeley" / "samples.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_mendeley_samples_are_train_only_and_unchanged():
    test_ids, rows = prep.verify_test_lock()
    split = {row["id"]: row for row in rows}
    samples = _samples()
    assert [s["class"] for s in samples].count("intact") == 2
    assert len(samples) - 2 == 5
    for sample in samples:
        assert sample["id"] not in test_ids
        assert split[sample["id"]]["split"] == "train"
        image = SAMPLE / "mendeley" / "images" / sample["file"]
        assert hashlib.md5(image.read_bytes()).hexdigest() == split[sample["id"]]["md5"]
        label = (SAMPLE / "mendeley" / "labels" / f"{sample['id']}.txt").read_text()
        n_boxes = len([line for line in label.splitlines() if line.strip()])
        if sample["class"] == "intact":
            assert n_boxes == 0
        else:
            assert 1 <= n_boxes <= 2
            assert all(line.split()[0] == "0" for line in label.splitlines())


def test_citation_and_size():
    text = (SAMPLE / "CITATION.txt").read_text(encoding="utf-8")
    for needle in ("CC BY 4.0", "10.17632/ww7fd9t325.1", "10.5066/F7JH3KBD", "public domain", "gprMax"):
        assert needle in text
    size = sum(p.stat().st_size for p in SAMPLE.rglob("*") if p.is_file())
    assert size < 15 * 1024 * 1024
