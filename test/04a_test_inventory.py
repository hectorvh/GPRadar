"""Unit tests for the Mendeley inventory helpers. No real dataset needed."""

import importlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

inventory = importlib.import_module("04a_inventory_mendeley")


def test_yolo_to_pixels_and_iou_match():
    width, height = 200, 100
    # Box covering the centre half of the image.
    line = "0 0.5 0.5 0.5 0.5"
    class_id, xmin, ymin, xmax, ymax = inventory.yolo_to_pixels(line, width, height)
    assert class_id == "0"
    assert abs(xmin - 50) < 1e-6
    assert abs(ymin - 25) < 1e-6
    assert abs(xmax - 150) < 1e-6
    assert abs(ymax - 75) < 1e-6

    yolo = [{"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}]
    voc = [{"xmin": 50.0, "ymin": 25.0, "xmax": 150.0, "ymax": 75.0}]
    ok, detail = inventory.boxes_match(yolo, voc, iou_threshold=0.95)
    assert ok, detail
    assert inventory.box_iou(yolo[0], voc[0]) > 0.95

    voc_bad = [{"xmin": 0.0, "ymin": 0.0, "xmax": 10.0, "ymax": 10.0}]
    ok_bad, _ = inventory.boxes_match(yolo, voc_bad, iou_threshold=0.95)
    assert not ok_bad


def test_dhash_copy_and_flip(tmp_path):
    rng = np.random.default_rng(0)
    grey = rng.random((64, 80))
    copy_hash = inventory.dhash64(grey)
    assert inventory.hamming64(copy_hash, inventory.dhash64(grey)) == 0

    flipped = np.fliplr(grey)
    # Identity distance to the flipped image should be larger than the
    # distance under the hflip transform of the original.
    identity_to_flip = inventory.hamming64(copy_hash, inventory.dhash64(flipped))
    under_hflip = inventory.hamming64(
        inventory.dhash64(flipped),
        inventory.transform_hash(grey, "hflip"),
    )
    assert under_hflip == 0
    assert identity_to_flip > under_hflip


def test_inventory_does_not_write_inside_root(tmp_path):
    root = tmp_path / "data"
    out = tmp_path / "out"
    root.mkdir()
    # Tiny synthetic tree: one original, one augmented with both labels.
    orig_dir = root / "cavities"
    aug_dir = root / "augmented_cavities"
    ann_yolo = aug_dir / "annotations" / "Yolo_format"
    ann_voc = aug_dir / "annotations" / "VOC_XML_format"
    orig_dir.mkdir()
    ann_yolo.mkdir(parents=True)
    ann_voc.mkdir(parents=True)

    def save_noise(path, seed):
        rng = np.random.default_rng(seed)
        arr = (rng.random((32, 32)) * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").convert("RGB").save(path)

    save_noise(orig_dir / "1.JPG", 1)
    save_noise(aug_dir / "1_aug_1.jpg", 2)
    (ann_yolo / "1_aug_1.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (ann_voc / "1_aug_1.xml").write_text(
        """
        <annotation>
          <size><width>32</width><height>32</height><depth>3</depth></size>
          <object>
            <name>cavities</name>
            <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>22</xmax><ymax>22</ymax></bndbox>
          </object>
        </annotation>
        """.strip(),
        encoding="utf-8",
    )

    before = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    inventory.run(root, out)
    after = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert after == before
    assert (out / "files.csv").is_file()
    assert (out / "boxes.csv").is_file()
    assert (out / "groups.csv").is_file()
    assert (out / "summary.md").is_file()
