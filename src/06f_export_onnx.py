"""Export the shipped YOLO11n detector to ONNX and check parity with Ultralytics.

The ONNX graph has no NMS: its output is (1, 5, 1029), i.e. cx, cy, w, h and
score in pixels of the 224 x 224 input, for 28^2 + 14^2 + 7^2 anchor points.
The inference path below uses only NumPy, OpenCV and onnxruntime (no torch,
no ultralytics), so it can run in a CPU-only web service.

The parity check reads TRAIN-split originals only and asserts on test ids.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "A_f100_seed2" / "weights" / "best.pt"
DEFAULT_SPLIT_CSV = ROOT / "data" / "splits" / "mendeley_originals.csv"
DEFAULT_RAW = ROOT / "data" / "raw" / "mendeley_gpr"
IMGSZ = 224
OPSET = 12
PAD_VALUE = 114
CONF = 0.25
IOU = 0.5
PARITY_MIN_IOU = 0.95
PARITY_MAX_DCONF = 0.01


def export_onnx(weights=DEFAULT_WEIGHTS):
    """Export ``weights`` to ``best.onnx`` next to it; return the ONNX path."""

    from ultralytics import YOLO

    out = YOLO(str(weights)).export(
        format="onnx",
        imgsz=IMGSZ,
        opset=OPSET,
        simplify=True,
        dynamic=False,
        nms=False,
        verbose=False,
    )
    return Path(out)


def open_session(onnx_path):
    """Open a CPU onnxruntime session."""

    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def describe_session(session):
    """Return input and output (name, shape) pairs."""

    inputs = [(item.name, list(item.shape)) for item in session.get_inputs()]
    outputs = [(item.name, list(item.shape)) for item in session.get_outputs()]
    return inputs, outputs


def letterbox(grey, size=IMGSZ):
    """Resize a greyscale image into a padded square model input.

    Keeps the aspect ratio (cv2.INTER_LINEAR), pads with 114 and centres the
    image with left = round(dw - 0.1), top = round(dh - 0.1), as Ultralytics
    does. Returns (1x3xSxS float32 in [0, 1], ratio, left, top).
    """

    grey = np.asarray(grey)
    if grey.ndim != 2:
        raise ValueError(f"expected a 2D greyscale image, got shape {grey.shape}")
    h, w = grey.shape
    ratio = min(size / h, size / w)
    new_w, new_h = max(round(w * ratio), 1), max(round(h * ratio), 1)
    if (new_w, new_h) != (w, h):
        grey = cv2.resize(grey, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    padded = cv2.copyMakeBorder(
        grey, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_VALUE
    )
    tensor = np.repeat(padded[None, None].astype(np.float32), 3, axis=1) / 255.0
    return tensor, ratio, left, top


def _box_iou_one_to_many(box, boxes):
    """IoU of one xyxy box with an (N, 4) array of xyxy boxes."""

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-12)


def nms(boxes, scores, iou=IOU):
    """Greedy NMS on xyxy boxes; return kept indices, highest score first."""

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    keep = []
    while order.size:
        best = order[0]
        keep.append(int(best))
        rest = order[1:]
        order = rest[_box_iou_one_to_many(boxes[best], boxes[rest]) <= iou]
    return np.asarray(keep, dtype=np.int64)


def _decode(output, min_conf):
    """Turn the (1, 5, N) output into xyxy input-pixel boxes and scores."""

    pred = np.asarray(output)[0].T
    pred = pred[pred[:, 4] > min_conf]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, pred[:, 4]


def _axis_gains(shape, ratio):
    """Per-axis scale (x, y) of the rounded resized image, as Ultralytics uses."""

    h, w = shape
    return max(round(w * ratio), 1) / w, max(round(h * ratio), 1) / h


def detect(session, grey, min_conf=0.05, iou=IOU):
    """Run the ONNX model on a greyscale image.

    Returns (boxes, scores): xyxy boxes in original-image pixels, clipped to
    the image, and their scores, highest first.
    """

    grey = np.asarray(grey)
    tensor, ratio, left, top = letterbox(grey)
    name = session.get_inputs()[0].name
    output = session.run(None, {name: tensor})[0]
    boxes, scores = _decode(output, min_conf)
    keep = nms(boxes, scores, iou=iou)
    boxes, scores = boxes[keep], scores[keep]
    gain_x, gain_y = _axis_gains(grey.shape, ratio)
    boxes = (boxes - np.array([left, top, left, top], dtype=boxes.dtype)) / np.array(
        [gain_x, gain_y, gain_x, gain_y], dtype=boxes.dtype
    )
    h, w = grey.shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes, scores


def train_images(n, split_csv=DEFAULT_SPLIT_CSV, raw_root=DEFAULT_RAW):
    """Pick ``n`` train-split originals, evenly spaced over sorted ids.

    Verifies the test lock and refuses any test id. Returns (id, path) pairs
    whose files exist.
    """

    prep = importlib.import_module("06a_prepare_real_yolo")
    test_ids, rows = prep.verify_test_lock(split_csv)
    test_set = set(test_ids)
    train = sorted(
        (row for row in rows if row["split"] == "train"), key=lambda row: row["id"]
    )
    if not train or n <= 0:
        return []
    picks = np.linspace(0, len(train) - 1, min(n, len(train))).round().astype(int)
    out = []
    for index in picks:
        row = train[int(index)]
        prep.assert_not_test(row["id"], test_set)
        path = Path(raw_root) / row["path"]
        if path.is_file():
            out.append((row["id"], path))
    return out


def load_grey(path):
    """Read an image as 8-bit greyscale (same conversion as 06a)."""

    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def _ultralytics_boxes(model, grey, rect=False):
    """Ultralytics predict on the grey image repeated to 3 channels (CPU).

    ``Model.predict`` defaults to ``rect=True`` for .pt weights: it pads only
    to a multiple of 32 (e.g. 224 x 192). The static ONNX graph always pads to
    224 x 224, so ``rect=False`` is the like-for-like comparison.
    """

    bgr = np.repeat(grey[:, :, None], 3, axis=2)
    result = model.predict(
        bgr, imgsz=IMGSZ, conf=CONF, iou=IOU, device="cpu", rect=rect, verbose=False
    )[0]
    return result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()


def _match(boxes_a, scores_a, boxes_b, scores_b):
    """Greedy one-to-one match by IoU; return (ious, |dconf|) per match."""

    ious, dconf = [], []
    free = list(range(len(boxes_b)))
    for i in np.argsort(-scores_a):
        if not free:
            break
        cand = _box_iou_one_to_many(boxes_a[i], boxes_b[free])
        j = int(np.argmax(cand))
        ious.append(float(cand[j]))
        dconf.append(abs(float(scores_a[i]) - float(scores_b[free[j]])))
        free.pop(j)
    return ious, dconf


def parity_check(weights, onnx_path, images, rect=False):
    """Compare ONNX and Ultralytics detections per image at conf 0.25.

    ``rect`` is passed to Ultralytics predict (see ``_ultralytics_boxes``).
    Returns rows: id, n_ultra, n_onnx, min_iou, max_dconf, ok.
    """

    from ultralytics import YOLO

    model = YOLO(str(weights))
    session = open_session(onnx_path)
    rows = []
    for image_id, path in images:
        grey = load_grey(path)
        ub, us = _ultralytics_boxes(model, grey, rect=rect)
        ob, os_ = detect(session, grey, min_conf=CONF, iou=IOU)
        ious, dconf = _match(ub, us, ob, os_)
        min_iou = min(ious) if ious else float("nan")
        max_dconf = max(dconf) if dconf else float("nan")
        ok = len(ub) == len(ob) and all(v >= PARITY_MIN_IOU for v in ious) and all(
            v <= PARITY_MAX_DCONF for v in dconf
        )
        rows.append(
            {
                "id": image_id,
                "n_ultra": len(ub),
                "n_onnx": len(ob),
                "min_iou": min_iou,
                "max_dconf": max_dconf,
                "ok": ok,
            }
        )
    return rows


def _print_parity(rows):
    """Print the per-image parity table and a one-line summary."""

    print(f"{'id':<20} {'ultra':>5} {'onnx':>5} {'min_iou':>8} {'max_dconf':>10} ok")
    for row in rows:
        print(
            f"{row['id']:<20} {row['n_ultra']:>5} {row['n_onnx']:>5} "
            f"{row['min_iou']:>8.4f} {row['max_dconf']:>10.5f} {row['ok']}"
        )
    ious = [r["min_iou"] for r in rows if r["n_ultra"]]
    dconf = [r["max_dconf"] for r in rows if r["n_ultra"]]
    print(
        f"parity: {sum(r['ok'] for r in rows)}/{len(rows)} images ok; "
        f"boxes {sum(r['n_ultra'] for r in rows)} (ultralytics) vs "
        f"{sum(r['n_onnx'] for r in rows)} (onnx); "
        f"min IoU {min(ious) if ious else float('nan'):.4f}; "
        f"max conf diff {max(dconf) if dconf else float('nan'):.5f}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--check", type=int, default=20, help="parity check on N train images (0 = skip)"
    )
    args = parser.parse_args(argv)

    onnx_path = export_onnx(args.weights)
    session = open_session(onnx_path)
    inputs, outputs = describe_session(session)
    print(f"onnx: {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")
    for name, shape in inputs:
        print(f"input  {name}: {shape}")
    for name, shape in outputs:
        print(f"output {name}: {shape}")

    if args.check > 0:
        images = train_images(args.check)
        if not images:
            print("parity: no train images found; skipped")
            return 0
        print("\nparity vs Ultralytics predict, rect=False (square 224 input, the check):")
        rows = parity_check(args.weights, onnx_path, images, rect=False)
        _print_parity(rows)
        print("\nfor information: vs Ultralytics default predict, rect=True:")
        _print_parity(parity_check(args.weights, onnx_path, images, rect=True))
        return 0 if all(row["ok"] for row in rows) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
