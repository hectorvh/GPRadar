"""Tests for the Mendeley original split, export and import. No real data."""

import importlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

split_mod = importlib.import_module("04b_split_mendeley")
inventory = importlib.import_module("04a_inventory_mendeley")


def _write_grey(path, seed, size=(48, 48)):
    """Lossless PNG so dHash near-duplicates stay stable in tests."""

    rng = np.random.default_rng(seed)
    arr = (rng.random(size) * 200 + 20).astype(np.uint8)
    Image.fromarray(arr, mode="L").convert("RGB").save(path, format="PNG")


def _build_tiny_dataset(root):
    """Several originals per class, including one near-duplicate pair."""

    (root / "cavities").mkdir(parents=True)
    (root / "Utilities").mkdir(parents=True)
    (root / "intact").mkdir(parents=True)
    # Enough groups per class for a ~30% test split.
    for index in range(1, 11):
        _write_grey(root / "cavities" / f"{index}.png", seed=100 + index)
    for index in range(1, 11):
        _write_grey(root / "Utilities" / f"{index:03d}.png", seed=200 + index)
    for index in range(1, 11):
        _write_grey(root / "intact" / f"{index}.png", seed=300 + index)
    # Near-duplicate of cavities/1 under horizontal flip (lossless).
    base = inventory.load_grey(root / "cavities" / "1.png")
    flipped = np.fliplr(base)
    Image.fromarray(np.clip(flipped * 255, 0, 255).astype(np.uint8), mode="L").convert(
        "RGB"
    ).save(root / "cavities" / "1 (2).png", format="PNG")
    (root / "augmented_cavities").mkdir()
    (root / "augmented_utilities").mkdir()


def test_near_duplicates_share_split(tmp_path):
    root = tmp_path / "mendeley"
    _build_tiny_dataset(root)
    split_csv = tmp_path / "split.csv"
    lock = tmp_path / "lock.txt"
    review = tmp_path / "review.png"
    split_mod.main(
        [
            "split",
            "--root",
            str(root),
            "--split-csv",
            str(split_csv),
            "--lock",
            str(lock),
            "--review-figure",
            str(review),
            "--seed",
            "0",
        ]
    )
    rows = split_mod.read_split_csv(split_csv)
    by_id = {row["id"]: row for row in rows}
    a = by_id["void_0001"]
    b = by_id["void_0001_2"]
    assert a["group_id"] == b["group_id"]
    assert a["split"] == b["split"]


def test_stratified_counts_and_seed(tmp_path):
    root = tmp_path / "mendeley"
    _build_tiny_dataset(root)
    split_csv = tmp_path / "split.csv"
    lock = tmp_path / "lock.txt"
    review = tmp_path / "review.png"
    args = [
        "split",
        "--root",
        str(root),
        "--split-csv",
        str(split_csv),
        "--lock",
        str(lock),
        "--review-figure",
        str(review),
        "--seed",
        "0",
    ]
    split_mod.main(args)
    rows = split_mod.read_split_csv(split_csv)
    for class_label in ("void", "utility", "intact"):
        class_rows = [r for r in rows if r["class"] == class_label]
        groups = {}
        for row in class_rows:
            groups.setdefault(row["group_id"], row["split"])
        n_test = sum(1 for split_name in groups.values() if split_name == "test")
        n_train = sum(1 for split_name in groups.values() if split_name == "train")
        assert n_test > 0 and n_train > 0
        # Split is defined on groups; share should stay near 30%.
        share = 100.0 * n_test / len(groups)
        assert abs(share - 30.0) <= 5.0 + 1e-9, (class_label, share, len(groups))

    first = [ (r["id"], r["split"]) for r in rows ]
    # Same seed reproduces the split; existing lock with same hash is fine.
    split_mod.main(args)
    second = [(r["id"], r["split"]) for r in split_mod.read_split_csv(split_csv)]
    assert first == second

    # A different split must be blocked by the lock.
    lock.write_text("0" * 64 + "\n", encoding="utf-8")
    try:
        split_mod.main(args)
        assert False, "expected SystemExit when lock disagrees"
    except SystemExit as exc:
        assert "refusing to overwrite" in str(exc)


def test_import_rejects_bad_labels(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels_yolo"
    images.mkdir()
    labels.mkdir()
    Image.fromarray(np.zeros((40, 50), dtype=np.uint8), mode="L").convert("RGB").save(
        images / "void_0001.jpg"
    )

    # Missing label file.
    failures = split_mod.validate_yolo_labels(images, labels)
    assert any("missing label" in item for item in failures)

    # Box outside the image.
    (labels / "void_0001.txt").write_text("0 0.9 0.9 0.4 0.4\n", encoding="utf-8")
    failures = split_mod.validate_yolo_labels(images, labels)
    assert any("outside image" in item for item in failures)

    # Class id other than 0.
    (labels / "void_0001.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    failures = split_mod.validate_yolo_labels(images, labels)
    assert any("is not 0" in item for item in failures)

    # Valid box passes.
    (labels / "void_0001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert split_mod.validate_yolo_labels(images, labels) == []


def test_export_copies_bytes_and_skips_dataset_root(tmp_path):
    root = tmp_path / "mendeley"
    _build_tiny_dataset(root)
    split_csv = tmp_path / "split.csv"
    lock = tmp_path / "lock.txt"
    review = tmp_path / "review.png"
    split_mod.main(
        [
            "split",
            "--root",
            str(root),
            "--split-csv",
            str(split_csv),
            "--lock",
            str(lock),
            "--review-figure",
            str(review),
        ]
    )
    export_dir = tmp_path / "export"
    guide = tmp_path / "guide.png"
    before = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    split_mod.main(
        [
            "export",
            "--root",
            str(root),
            "--split-csv",
            str(split_csv),
            "--export-dir",
            str(export_dir),
            "--guide-figure",
            str(guide),
        ]
    )
    after = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert after == before

    rows = [r for r in split_mod.read_split_csv(split_csv) if r["class"] in ("void", "utility")]
    for row in rows:
        dest = export_dir / "images" / f"{row['id']}.jpg"
        assert dest.is_file()
        assert inventory.md5_file(root / row["path"]) == inventory.md5_file(dest)
    assert (export_dir / "labels.txt").read_text(encoding="utf-8").strip() == "target"
