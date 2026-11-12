"""ONNX export checks: pure letterbox/NMS unit tests and a parity test.

The parity test reads TRAIN-split originals only and is skipped when the
weights, the ONNX file, onnxruntime, ultralytics or the images are missing.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

onnx_mod = importlib.import_module("06f_export_onnx")

PAD = 114 / 255.0


@pytest.mark.parametrize(
    "shape, ratio, left, top",
    [
        ((100, 100), 2.24, 0, 0),  # square: scale up, no padding
        ((100, 200), 1.12, 0, 56),  # wide: pad top and bottom
        ((200, 50), 1.12, 84, 0),  # tall: pad left and right
        ((101, 224), 1.0, 0, 61),  # odd padding: 61 top, 62 bottom
    ],
)
def test_letterbox_shapes_and_padding(shape, ratio, left, top):
    grey = np.full(shape, 200, dtype=np.uint8)
    tensor, r, l, t = onnx_mod.letterbox(grey)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    assert r == pytest.approx(ratio)
    assert (l, t) == (left, top)
    # All three channels identical; content and padding values as expected.
    assert np.array_equal(tensor[0, 0], tensor[0, 1])
    assert np.array_equal(tensor[0, 0], tensor[0, 2])
    assert tensor[0, 0, 112, 112] == pytest.approx(200 / 255.0)
    if top:
        assert tensor[0, 0, 0, 112] == pytest.approx(PAD)
        assert tensor[0, 0, -1, 112] == pytest.approx(PAD)
    if left:
        assert tensor[0, 0, 112, 0] == pytest.approx(PAD)
        assert tensor[0, 0, 112, -1] == pytest.approx(PAD)
    if shape == (101, 224):
        content_rows = np.where(np.isclose(tensor[0, 0, :, 0], 200 / 255.0))[0]
        assert (content_rows.min(), content_rows.max()) == (61, 161)


def test_nms_three_toy_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], float)
    scores = np.array([0.9, 0.8, 0.7])
    # Boxes 0 and 1 overlap with IoU 81/119 = 0.68 > 0.5; box 2 is separate.
    assert onnx_mod.nms(boxes, scores, iou=0.5).tolist() == [0, 2]
    # With a looser threshold nothing is suppressed.
    assert onnx_mod.nms(boxes, scores, iou=0.7).tolist() == [0, 1, 2]


def test_detect_maps_back_to_original_pixels():
    # 100 x 200 image: ratio 1.12, top pad 56, left pad 0.
    out = np.zeros((1, 5, 3), dtype=np.float32)
    out[0, :, 0] = [56, 84, 112, 56, 0.9]  # input box x 0..112, y 56..112
    out[0, :, 1] = [57, 85, 112, 56, 0.8]  # duplicate, suppressed by NMS
    out[0, :, 2] = [200, 100, 80, 40, 0.01]  # below min_conf
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="images")],
        run=lambda names, feed: [out],
    )
    boxes, scores = onnx_mod.detect(session, np.zeros((100, 200), np.uint8))
    assert scores.tolist() == pytest.approx([0.9])
    assert boxes[0] == pytest.approx([0, 0, 100, 50], abs=1e-4)


def test_onnx_matches_ultralytics_on_train_images():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("ultralytics")
    weights = onnx_mod.DEFAULT_WEIGHTS
    onnx_path = weights.with_suffix(".onnx")
    if not weights.is_file() or not onnx_path.is_file():
        pytest.skip("best.pt or best.onnx missing; run src/06f_export_onnx.py")
    images = onnx_mod.train_images(20)
    if len(images) < 20:
        pytest.skip("train images missing under data/raw/mendeley_gpr")
    rows = onnx_mod.parity_check(weights, onnx_path, images)
    for row in rows:
        assert row["n_ultra"] == row["n_onnx"], row
        if row["n_ultra"]:
            assert row["min_iou"] >= 0.95, row
            assert row["max_dconf"] <= 0.01, row
