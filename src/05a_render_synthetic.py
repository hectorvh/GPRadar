"""Render field_400mhz scenes into a 224 x 224 greyscale YOLO set.

Processing matches ``01c`` / ``02b`` (no reference subtraction on the image).
Boxes follow the hand-labelling rule on the kink-fixed predicted hyperbola.
Real train labels supply the box-size distribution. Test-split Mendeley ids are
never opened.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

loader = importlib.import_module("02a_gprmax_loader")
processing = importlib.import_module("02b_preprocessing")
scenes_mod = importlib.import_module("01c_scene_generator")

DEFAULT_SCENES = ROOT / "data" / "sim" / "scenes" / "field_400mhz"
DEFAULT_OUT = ROOT / "data" / "synth" / "field_400mhz"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "mendeley_originals.csv"
DEFAULT_LABELS = ROOT / "data" / "labels" / "mendeley_originals"
DEFAULT_RAW = ROOT / "data" / "raw" / "mendeley_gpr"
DEFAULT_STATS = ROOT / "data" / "synth" / "real_train_box_stats.json"
DEFAULT_BOX_CHECK = ROOT / "figures" / "synthetic_boxes_check.png"
DEFAULT_MATCH = ROOT / "figures" / "synthetic_vs_real.png"

FREQUENCY_HZ = 4.0e8
PERIOD_S = 1.0 / FREQUENCY_HZ
PERIOD_NS = 1e9 * PERIOD_S
IMAGE_SIZE = 224
ENVELOPE_FRAC = 0.20
CROPS_PER_SCENE = 6
NEGATIVE_FRAC = 0.20
MAX_CROP_DRAWS = 50
TRAIN_SEED_MAX = 179  # seeds 0..179 train, 180..199 val
VAL_SEED_MIN = 180
JPEG_Q_LO, JPEG_Q_HI = 75, 95
# Aspect rule: crop width and height are set from independent real width/height
# fractions, then the rectangle is bilinear-resized to 224 x 224 (anisotropic).


class TestSplitError(RuntimeError):
    """Raised when a Mendeley test-split id would be loaded."""


def load_split_rows(split_csv=DEFAULT_SPLIT):
    """Return all split rows and the set of test ids."""

    rows = list(csv.DictReader(Path(split_csv).open(encoding="utf-8")))
    test_ids = {row["id"] for row in rows if row["split"] == "test"}
    return rows, test_ids


def assert_not_test(image_id, test_ids):
    """Fail if ``image_id`` belongs to the locked test split."""

    if image_id in test_ids:
        raise TestSplitError(f"refusing to load test-split id: {image_id}")


def train_label_rows(split_csv=DEFAULT_SPLIT):
    """Train void/utility rows only. Intact has no boxes."""

    rows, test_ids = load_split_rows(split_csv)
    out = []
    for row in rows:
        if row["split"] != "train":
            continue
        if row["class"] not in ("void", "utility"):
            continue
        assert_not_test(row["id"], test_ids)
        out.append(row)
    return out, test_ids


def real_train_box_stats(
    split_csv=DEFAULT_SPLIT,
    labels_dir=DEFAULT_LABELS,
    raw_root=DEFAULT_RAW,
):
    """Box width/height fractions from train YOLO labels only."""

    rows, test_ids = train_label_rows(split_csv)
    labels_dir = Path(labels_dir)
    by_class = defaultdict(lambda: {"width": [], "height": []})
    pooled_w, pooled_h = [], []
    for row in rows:
        assert_not_test(row["id"], test_ids)
        path = labels_dir / f"{row['id']}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"bad YOLO line in {path}: {line!r}")
            class_id, _xc, _yc, bw, bh = parts
            if class_id != "0":
                raise ValueError(f"expected class 0 in {path}, got {class_id}")
            width = float(bw)
            height = float(bh)
            by_class[row["class"]]["width"].append(width)
            by_class[row["class"]]["height"].append(height)
            pooled_w.append(width)
            pooled_h.append(height)

    def summarise(widths, heights):
        return {
            "n_boxes": len(widths),
            "width_median": float(np.median(widths)) if widths else None,
            "height_median": float(np.median(heights)) if heights else None,
            "widths": [float(v) for v in widths],
            "heights": [float(v) for v in heights],
        }

    stats = {
        "pooled": summarise(pooled_w, pooled_h),
        "utility": summarise(by_class["utility"]["width"], by_class["utility"]["height"]),
        "void": summarise(by_class["void"]["width"], by_class["void"]["height"]),
        "source": "train split of data/splits/mendeley_originals.csv only",
    }
    # Sanity against the known hand-label summary.
    assert stats["utility"]["n_boxes"] == 113, stats["utility"]["n_boxes"]
    assert stats["void"]["n_boxes"] == 109, stats["void"]["n_boxes"]
    assert abs(stats["utility"]["width_median"] - 0.49) < 0.02
    assert abs(stats["utility"]["height_median"] - 0.55) < 0.02
    assert abs(stats["void"]["width_median"] - 0.46) < 0.02
    assert abs(stats["void"]["height_median"] - 0.48) < 0.02
    return stats


def processing_windows(frequency_hz=FREQUENCY_HZ):
    """Dewow and AGC windows in ns, and the band-pass edges in Hz."""

    dewow_ns, agc_ns = scenes_mod.processing_windows_ns(frequency_hz)
    # Band edges need dt; caller fills high from Nyquist. Half to twice centre.
    return {
        "dewow_ns": float(dewow_ns),
        "agc_ns": float(agc_ns),
        "period_ns": float(1e9 / frequency_hz),
        "band_low_hz": 0.5 * float(frequency_hz),
        "band_high_factor": 2.0,
        "frequency_hz": float(frequency_hz),
    }


def process_scene_array(data, dt, frequency_hz=FREQUENCY_HZ):
    """Time zero, dewow, mean-trace removal, band-pass, AGC. No reference sub."""

    windows = processing_windows(frequency_hz)
    shifted, picks = processing.time_zero(data, dt)
    dewowed = processing.dewow(shifted, dt, windows["dewow_ns"])
    cleaned = processing.remove_background(dewowed)
    nyquist = 0.5 / float(dt)
    low, high = processing.default_band(frequency_hz, nyquist)
    filtered = processing.bandpass(cleaned, dt, low, high)
    gained = processing.gain_agc(filtered, dt, windows["agc_ns"])
    windows = dict(windows)
    windows["band_low_hz"] = float(low)
    windows["band_high_hz"] = float(high)
    return gained, picks, windows


def reference_envelope(scene_data, reference_trace, dt, picks, frequency_hz=FREQUENCY_HZ):
    """Envelope of reference-subtracted, time-zeroed, band-passed data."""

    ref = np.asarray(reference_trace, dtype=np.float64).reshape(-1, 1)
    if ref.shape[0] != scene_data.shape[0]:
        raise ValueError("reference and scene sample counts differ")
    diff = np.asarray(scene_data, dtype=np.float64) - ref
    shifted = np.zeros_like(diff)
    n_samples = diff.shape[0]
    for index, pick in enumerate(np.asarray(picks, dtype=int)):
        pick = int(pick)
        shifted[: n_samples - pick, index] = diff[pick:, index]
    dewow_ns, _agc_ns = scenes_mod.processing_windows_ns(frequency_hz)
    dewowed = processing.dewow(shifted, dt, dewow_ns)
    nyquist = 0.5 / float(dt)
    low, high = processing.default_band(frequency_hz, nyquist)
    filtered = processing.bandpass(dewowed, dt, low, high)
    analytic = hilbert(np.asarray(filtered, dtype=np.float64), axis=0)
    return np.abs(analytic)


def fix_hyperbola_branch(x_m, t_s, apex_x, time_window_s):
    """Keep the continuous hyperbola branch through the apex.

    Walk left and right from the apex. Drop points once the first difference
    of travel time shrinks (the kink onto the straight-ray branch) or the
    time leaves the window.
    """

    x_m = np.asarray(x_m, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    if x_m.size == 0:
        return x_m, t_s, np.array([], dtype=int)
    apex_i = int(np.argmin(np.abs(x_m - float(apex_x))))
    keep = np.zeros(x_m.size, dtype=bool)
    keep[apex_i] = True

    def walk(indices):
        prev_diff = None
        prev_t = t_s[apex_i]
        for index in indices:
            t_val = t_s[index]
            if not np.isfinite(t_val) or t_val > time_window_s + 1e-15:
                break
            diff = abs(t_val - prev_t)
            if prev_diff is not None and diff + 1e-15 < 0.5 * prev_diff:
                break
            if t_val + 1e-15 < prev_t:
                # Travel time must not fall while leaving the apex.
                break
            keep[index] = True
            prev_diff = diff
            prev_t = t_val

    walk(range(apex_i + 1, x_m.size))
    walk(range(apex_i - 1, -1, -1))
    indices = np.flatnonzero(keep)
    return x_m[indices], t_s[indices], indices


def _sample_envelope(envelope, trace_i, sample_i, half=2):
    """Median envelope in a small window around (sample, trace)."""

    n_samples, n_traces = envelope.shape
    t0 = max(0, int(sample_i) - half)
    t1 = min(n_samples, int(sample_i) + half + 1)
    x0 = max(0, int(trace_i) - 1)
    x1 = min(n_traces, int(trace_i) + 2)
    patch = envelope[t0:t1, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.median(patch))


def box_from_hyperbola(
    x_m,
    t_s,
    apex_x,
    apex_t_s,
    positions_x,
    dt,
    picks,
    envelope,
    time_window_s,
    period_s=PERIOD_S,
    envelope_frac=ENVELOPE_FRAC,
):
    """Physical box for one visible target, in metres and seconds."""

    x_fixed, t_fixed, indices = fix_hyperbola_branch(x_m, t_s, apex_x, time_window_s)
    if x_fixed.size == 0:
        return None

    # Time-zero shift per trace (same picks as the processed image).
    picks = np.asarray(picks, dtype=int)
    shift_s = picks * float(dt)
    # Map hyperbola samples onto the nearest traces.
    positions_x = np.asarray(positions_x, dtype=float)
    trace_index = np.array(
        [int(np.argmin(np.abs(positions_x - float(x)))) for x in x_fixed],
        dtype=int,
    )
    t_shifted = t_fixed - shift_s[trace_index]
    apex_i = int(np.argmin(np.abs(x_fixed - float(apex_x))))
    apex_trace = int(trace_index[apex_i])
    apex_t_shifted = float(apex_t_s) - float(shift_s[apex_trace])
    apex_sample = int(round(apex_t_shifted / float(dt)))
    apex_sample = int(np.clip(apex_sample, 0, envelope.shape[0] - 1))
    apex_env = _sample_envelope(envelope, apex_trace, apex_sample)
    threshold = envelope_frac * apex_env if apex_env > 0 else 0.0

    # Walk out from the apex along the fixed branch until the envelope fades.
    def flank(order):
        last = apex_i
        for index in order:
            sample = int(round(t_shifted[index] / float(dt)))
            if sample < 0 or sample >= envelope.shape[0]:
                break
            value = _sample_envelope(envelope, int(trace_index[index]), sample)
            if threshold > 0 and value < threshold:
                break
            last = index
        return last

    left_i = flank(range(apex_i - 1, -1, -1))
    right_i = flank(range(apex_i + 1, x_fixed.size))
    used = slice(left_i, right_i + 1)
    x_used = x_fixed[used]
    t_used = t_shifted[used]
    if x_used.size == 0:
        return None

    half_period = 0.5 * float(period_s)
    # Two-trace margin in metres from the profile step.
    if positions_x.size >= 2:
        step = float(np.median(np.diff(positions_x)))
    else:
        step = 0.05
    margin_x = 2.0 * step

    xmin = float(x_used.min()) - margin_x
    xmax = float(x_used.max()) + margin_x
    tmin = float(apex_t_shifted) - half_period
    tmax = float(t_used.max()) + half_period

    # Clip to the processed record.
    x_lo = float(positions_x[0])
    x_hi = float(positions_x[-1])
    t_lo = 0.0
    t_hi = (envelope.shape[0] - 1) * float(dt)
    xmin = max(xmin, x_lo)
    xmax = min(xmax, x_hi)
    tmin = max(tmin, t_lo)
    tmax = min(tmax, t_hi)
    if xmax <= xmin or tmax <= tmin:
        return None
    return {
        "xmin_m": xmin,
        "xmax_m": xmax,
        "tmin_s": tmin,
        "tmax_s": tmax,
        "apex_x_m": float(apex_x),
        "apex_t_s": float(apex_t_shifted),
        "curve_x_m": [float(v) for v in x_used],
        "curve_t_s": [float(v) for v in t_used],
        "envelope_frac": float(envelope_frac),
        "apex_envelope": float(apex_env),
    }


def load_scene_bundle(scene_dir):
    """Load scene.json, merged scan, reference, and run.json."""

    scene_dir = Path(scene_dir)
    scene = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    run = json.loads((scene_dir / "run.json").read_text(encoding="utf-8"))
    scan = loader.load_gprmax(scene_dir / "scene_merged.h5")
    ref_scan = loader.load_gprmax(scene_dir / "reference.h5")
    return scene, run, scan, ref_scan


def physical_boxes_for_scene(scene_dir, envelope_frac=ENVELOPE_FRAC):
    """Compute and return physical boxes for visible targets."""

    scene, run, scan, ref_scan = load_scene_bundle(scene_dir)
    processed, picks, windows = process_scene_array(
        scan.data, scan.dt, scene["frequency_hz"]
    )
    envelope = reference_envelope(
        scan.data, ref_scan.data[:, 0], scan.dt, picks, scene["frequency_hz"]
    )
    positions = np.asarray(scan.positions[:, 0], dtype=float)
    # Midpoints for hyperbola x match the stored ground-truth table.
    truth = scene["ground_truth"]["targets"]
    boxes = []
    for target in truth:
        if not target.get("visible", False):
            continue
        box = box_from_hyperbola(
            target["hyperbola_x_m"],
            target["hyperbola_t_s"],
            target["x"],
            target["apex_time_s"],
            positions,
            scan.dt,
            picks,
            envelope,
            scene["time_window_s"],
            period_s=1.0 / scene["frequency_hz"],
            envelope_frac=envelope_frac,
        )
        if box is None:
            continue
        box["name"] = target["name"]
        box["type"] = target["type"]
        boxes.append(box)
    payload = {
        "seed": scene["seed"],
        "preset": scene["preset"],
        "windows": windows,
        "envelope_frac": envelope_frac,
        "boxes": boxes,
    }
    return payload, processed, picks, scan, scene, run


def write_boxes_physical(scene_dir, payload):
    """Write boxes_physical.json beside the scene outputs."""

    path = Path(scene_dir) / "boxes_physical.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def flagged_apex_rows(scenes_root=DEFAULT_SCENES, n_scenes=200):
    """Targets with |error| > quarter period or on_window_edge."""

    rows = []
    for seed in range(int(n_scenes)):
        run_path = Path(scenes_root) / f"seed{seed}" / "run.json"
        if not run_path.is_file():
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        for row in run.get("apex", []):
            err = abs(float(row.get("error_ns") or 0.0))
            qp = float(row.get("quarter_period_ns") or 0.625)
            edge = bool(row.get("on_window_edge"))
            if err > qp or edge:
                rows.append({"seed": seed, **row})
    return rows


def clip_box_to_crop(box, crop, min_area_frac=0.5):
    """Clip a pixel box to a crop. Return 'keep', 'clip', or 'reject'.

    ``box`` and ``crop`` are (xmin, ymin, xmax, ymax) in the same coordinates.
    """

    bx0, by0, bx1, by1 = box
    cx0, cy0, cx1, cy1 = crop
    area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if area <= 0:
        return "reject", None
    ix0 = max(bx0, cx0)
    iy0 = max(by0, cy0)
    ix1 = min(bx1, cx1)
    iy1 = min(by1, cy1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    frac = inter / area
    if frac <= 0:
        return "reject", None
    if frac < min_area_frac:
        return "reject", None
    clipped = (ix0, iy0, ix1, iy1)
    if clipped == (bx0, by0, bx1, by1):
        return "keep", clipped
    return "clip", clipped


def box_to_yolo(xmin, ymin, xmax, ymax, width, height):
    """Normalised YOLO line values for class 0."""

    bw = (xmax - xmin) / float(width)
    bh = (ymax - ymin) / float(height)
    xc = ((xmin + xmax) / 2.0) / float(width)
    yc = ((ymin + ymax) / 2.0) / float(height)
    # Clip numerically to [0, 1].
    xc = float(np.clip(xc, 0.0, 1.0))
    yc = float(np.clip(yc, 0.0, 1.0))
    bw = float(np.clip(bw, 0.0, 1.0))
    bh = float(np.clip(bh, 0.0, 1.0))
    return xc, yc, bw, bh


def robust_to_uint8(data):
    """Symmetric robust scale to 8-bit greyscale (0 at mid grey)."""

    values = np.asarray(data, dtype=np.float64)
    limit = float(np.percentile(np.abs(values), 99))
    if limit <= 0:
        limit = 1.0
    scaled = np.clip(values / limit, -1.0, 1.0)
    return np.round((scaled + 1.0) * 0.5 * 255.0).astype(np.uint8)


def apply_realism(grey_u8, rng):
    """Noise, contrast, gain, jitter, marker lines, JPEG. Returns uint8 array."""

    image = Image.fromarray(grey_u8, mode="L")
    contrast = float(rng.uniform(0.85, 1.25))
    image = ImageEnhance.Contrast(image).enhance(contrast)
    brightness = float(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Brightness(image).enhance(brightness)
    arr = np.asarray(image, dtype=np.float64)
    # Trace-to-trace jitter of 1 sample.
    jittered = np.zeros_like(arr)
    for col in range(arr.shape[1]):
        shift = int(rng.integers(-1, 2))
        jittered[:, col] = np.roll(arr[:, col], shift)
    noise_level = float(rng.uniform(2.0, 12.0))
    noisy = jittered + rng.normal(0.0, noise_level, size=jittered.shape)
    noisy = np.clip(noisy, 0, 255)
    # Thin vertical white markers in the top 15%.
    n_markers = int(rng.integers(0, 4))
    top = max(1, int(0.15 * noisy.shape[0]))
    for _ in range(n_markers):
        x = int(rng.integers(0, noisy.shape[1]))
        noisy[:top, x] = 255
    out = Image.fromarray(noisy.astype(np.uint8), mode="L")
    quality = int(rng.integers(JPEG_Q_LO, JPEG_Q_HI + 1))
    from io import BytesIO

    buffer = BytesIO()
    out.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("L")), {
        "contrast": contrast,
        "brightness": brightness,
        "noise_level": noise_level,
        "n_markers": n_markers,
        "jpeg_quality": quality,
    }


def physical_to_indices(box, positions_x, dt):
    """Map a physical box to inclusive-exclusive float indices (trace, sample)."""

    positions_x = np.asarray(positions_x, dtype=float)
    x0 = float(np.interp(box["xmin_m"], positions_x, np.arange(positions_x.size)))
    x1 = float(np.interp(box["xmax_m"], positions_x, np.arange(positions_x.size)))
    y0 = float(box["tmin_s"] / dt)
    y1 = float(box["tmax_s"] / dt)
    return x0, y0, x1, y1


def draw_crop(
    processed,
    positions_x,
    dt,
    boxes,
    stats,
    rng,
    want_negative,
):
    """Draw one crop window. Returns dict or None after MAX_CROP_DRAWS."""

    n_samples, n_traces = processed.shape
    widths = np.asarray(stats["pooled"]["widths"], dtype=float)
    heights = np.asarray(stats["pooled"]["heights"], dtype=float)
    for _attempt in range(MAX_CROP_DRAWS):
        if want_negative or not boxes:
            # Negative: random window with no box retaining >=50% area.
            w_frac = float(rng.choice(widths))
            h_frac = float(rng.choice(heights))
            crop_w = max(8, int(round(n_traces * w_frac)))
            crop_h = max(8, int(round(n_samples * h_frac)))
            # Prefer empty regions: try random placement.
            if n_traces - crop_w < 1 or n_samples - crop_h < 1:
                continue
            x0 = int(rng.integers(0, n_traces - crop_w + 1))
            y0 = int(rng.integers(0, n_samples - crop_h + 1))
            crop = (x0, y0, x0 + crop_w, y0 + crop_h)
            ok = True
            for box in boxes:
                px = physical_to_indices(box, positions_x, dt)
                decision, _ = clip_box_to_crop(px, crop, min_area_frac=0.5)
                if decision != "reject":
                    ok = False
                    break
            if not ok:
                continue
            return {
                "crop": crop,
                "boxes_px": [],
                "negative": True,
                "w_frac_draw": w_frac,
                "h_frac_draw": h_frac,
            }

        target = boxes[int(rng.integers(0, len(boxes)))]
        w_frac = float(rng.choice(widths))
        h_frac = float(rng.choice(heights))
        w_frac = float(np.clip(w_frac, 0.15, 0.95))
        h_frac = float(np.clip(h_frac, 0.15, 0.95))
        tx0, ty0, tx1, ty1 = physical_to_indices(target, positions_x, dt)
        box_w = max(tx1 - tx0, 1.0)
        box_h = max(ty1 - ty0, 1.0)
        crop_w = int(round(box_w / w_frac))
        crop_h = int(round(box_h / h_frac))
        crop_w = int(np.clip(crop_w, 8, n_traces))
        crop_h = int(np.clip(crop_h, 8, n_samples))
        # Place so the target stays inside; random offset within slack.
        slack_x = crop_w - box_w
        slack_y = crop_h - box_h
        if slack_x < 0 or slack_y < 0:
            continue
        x0 = int(round(tx0 - float(rng.uniform(0, max(slack_x, 1e-6)))))
        y0 = int(round(ty0 - float(rng.uniform(0, max(slack_y, 1e-6)))))
        x0 = int(np.clip(x0, 0, n_traces - crop_w))
        y0 = int(np.clip(y0, 0, n_samples - crop_h))
        crop = (x0, y0, x0 + crop_w, y0 + crop_h)

        kept = []
        rejected = False
        for box in boxes:
            px = physical_to_indices(box, positions_x, dt)
            decision, clipped = clip_box_to_crop(px, crop, min_area_frac=0.5)
            if decision == "reject":
                # Partial overlap under 50%: reject the crop.
                bx0, by0, bx1, by1 = px
                cx0, cy0, cx1, cy1 = crop
                inter = max(0.0, min(bx1, cx1) - max(bx0, cx0)) * max(
                    0.0, min(by1, cy1) - max(by0, cy0)
                )
                area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
                if area > 0 and 0 < inter / area < 0.5:
                    rejected = True
                    break
                continue
            # Shift to crop coordinates.
            ix0, iy0, ix1, iy1 = clipped
            kept.append((ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0))
        if rejected:
            continue
        if not kept:
            continue
        return {
            "crop": crop,
            "boxes_px": kept,
            "negative": False,
            "w_frac_draw": w_frac,
            "h_frac_draw": h_frac,
        }
    return None


def render_crop_image(processed, crop, boxes_px, rng):
    """Extract, resize to 224, apply realism, map boxes to 224 space."""

    x0, y0, x1, y1 = crop
    patch = processed[y0:y1, x0:x1]
    grey = robust_to_uint8(patch)
    pil = Image.fromarray(grey, mode="L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR
    )
    arr = np.asarray(pil, dtype=np.uint8)
    arr, meta = apply_realism(arr, rng)
    sx = IMAGE_SIZE / float(max(x1 - x0, 1))
    sy = IMAGE_SIZE / float(max(y1 - y0, 1))
    yolo_boxes = []
    for bx0, by0, bx1, by1 in boxes_px:
        X0, Y0, X1, Y1 = bx0 * sx, by0 * sy, bx1 * sx, by1 * sy
        X0 = float(np.clip(X0, 0, IMAGE_SIZE))
        Y0 = float(np.clip(Y0, 0, IMAGE_SIZE))
        X1 = float(np.clip(X1, 0, IMAGE_SIZE))
        Y1 = float(np.clip(Y1, 0, IMAGE_SIZE))
        if X1 - X0 < 1 or Y1 - Y0 < 1:
            continue
        yolo_boxes.append(box_to_yolo(X0, Y0, X1, Y1, IMAGE_SIZE, IMAGE_SIZE))
    return arr, yolo_boxes, meta


def scene_split(seed):
    """train or val from the scene seed."""

    return "train" if int(seed) <= TRAIN_SEED_MAX else "val"


def ensure_synth_dirs(out_root):
    out_root = Path(out_root)
    for split_name in ("train", "val"):
        (out_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    return out_root


def render_dataset(
    scenes_root=DEFAULT_SCENES,
    out_root=DEFAULT_OUT,
    stats_path=DEFAULT_STATS,
    n_crops=CROPS_PER_SCENE,
    n_scenes=200,
    seed0=0,
):
    """Render all scenes into YOLO images/labels and a manifest."""

    stats = real_train_box_stats()
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    # Store medians without the full sample lists in the tracked file.
    slim = {
        "pooled": {
            "n_boxes": stats["pooled"]["n_boxes"],
            "width_median": stats["pooled"]["width_median"],
            "height_median": stats["pooled"]["height_median"],
            "widths": stats["pooled"]["widths"],
            "heights": stats["pooled"]["heights"],
        },
        "utility": {
            "n_boxes": stats["utility"]["n_boxes"],
            "width_median": stats["utility"]["width_median"],
            "height_median": stats["utility"]["height_median"],
        },
        "void": {
            "n_boxes": stats["void"]["n_boxes"],
            "width_median": stats["void"]["width_median"],
            "height_median": stats["void"]["height_median"],
        },
        "source": stats["source"],
    }
    Path(stats_path).write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")

    out_root = ensure_synth_dirs(out_root)
    windows_note = processing_windows()
    print(
        "Processing windows: "
        f"dewow={windows_note['dewow_ns']:.2f} ns, "
        f"agc={windows_note['agc_ns']:.2f} ns, "
        f"period={windows_note['period_ns']:.2f} ns, "
        f"band≈{windows_note['band_low_hz']/1e6:.0f}-"
        f"{2*FREQUENCY_HZ/1e6:.0f} MHz (clipped to Nyquist)"
    )

    manifest = []
    failed_crops = []
    for seed in range(int(seed0), int(seed0) + int(n_scenes)):
        scene_dir = Path(scenes_root) / f"seed{seed}"
        if not (scene_dir / "scene_merged.h5").is_file():
            print(f"skip missing scene {scene_dir}")
            continue
        payload, processed, picks, scan, scene, run = physical_boxes_for_scene(scene_dir)
        write_boxes_physical(scene_dir, payload)
        boxes = payload["boxes"]
        positions = np.asarray(scan.positions[:, 0], dtype=float)
        split_name = scene_split(seed)
        n_neg = int(round(n_crops * NEGATIVE_FRAC))
        for crop_index in range(n_crops):
            rng = np.random.default_rng(10007 * int(seed) + 17 * crop_index + 3)
            want_negative = crop_index < n_neg
            drawn = draw_crop(
                processed, positions, scan.dt, boxes, stats, rng, want_negative
            )
            crop_id = f"seed{seed}_crop{crop_index:02d}"
            if drawn is None:
                failed_crops.append(crop_id)
                print(f"gave up on crop {crop_id}")
                continue
            arr, yolo_boxes, meta = render_crop_image(
                processed, drawn["crop"], drawn["boxes_px"], rng
            )
            img_path = out_root / "images" / split_name / f"{crop_id}.jpg"
            lab_path = out_root / "labels" / split_name / f"{crop_id}.txt"
            Image.fromarray(arr, mode="L").save(img_path, quality=95)
            if yolo_boxes:
                lab_path.write_text(
                    "".join(
                        f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
                        for xc, yc, bw, bh in yolo_boxes
                    ),
                    encoding="utf-8",
                )
            else:
                lab_path.write_text("", encoding="utf-8")
            x0, y0, x1, y1 = drawn["crop"]
            manifest.append(
                {
                    "crop_id": crop_id,
                    "seed": seed,
                    "split": split_name,
                    "trace0": x0,
                    "trace1": x1,
                    "sample0": y0,
                    "sample1": y1,
                    "scale_w": IMAGE_SIZE / max(x1 - x0, 1),
                    "scale_h": IMAGE_SIZE / max(y1 - y0, 1),
                    "n_boxes": len(yolo_boxes),
                    "negative": int(drawn["negative"] or len(yolo_boxes) == 0),
                    **{f"realism_{k}": v for k, v in meta.items()},
                }
            )
        print(f"seed {seed}: {len(boxes)} boxes, split={split_name}", flush=True)

    manifest_path = out_root / "manifest.csv"
    if manifest:
        fieldnames = list(manifest[0].keys())
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest)

    yaml_path = out_root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: target",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "stats": stats,
        "manifest": manifest,
        "failed_crops": failed_crops,
        "windows": windows_note,
        "out_root": out_root,
    }


def plot_boxes_check(
    scenes_root=DEFAULT_SCENES,
    figure_path=DEFAULT_BOX_CHECK,
    n_scenes=200,
):
    """Processed panels for every scene that has a flagged apex target."""

    flagged = flagged_apex_rows(scenes_root, n_scenes)
    seeds = sorted({row["seed"] for row in flagged})
    if len(seeds) < 20:
        # Pad with early seeds so the figure still has 20 panels when possible.
        for seed in range(n_scenes):
            if seed not in seeds:
                seeds.append(seed)
            if len(seeds) >= 20:
                break
    seeds = seeds[:20]
    flagged_keys = {(row["seed"], row["name"]) for row in flagged}

    cols = 5
    rows = int(np.ceil(len(seeds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows))
    axes = np.atleast_2d(axes)
    coverage = []
    for index, seed in enumerate(seeds):
        ax = axes[index // cols, index % cols]
        scene_dir = Path(scenes_root) / f"seed{seed}"
        payload, processed, picks, scan, scene, run = physical_boxes_for_scene(scene_dir)
        write_boxes_physical(scene_dir, payload)
        positions = np.asarray(scan.positions[:, 0], dtype=float)
        time_ns = np.arange(processed.shape[0]) * scan.dt * 1e9
        limit = np.percentile(np.abs(processed), 99) or 1.0
        ax.imshow(
            processed,
            extent=[positions[0], positions[-1], time_ns[-1], time_ns[0]],
            aspect="auto",
            cmap="gray",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        for box in payload["boxes"]:
            flagged_here = (seed, box["name"]) in flagged_keys
            colour = "#ff3300" if flagged_here else "#33ff66"
            ax.plot(box["curve_x_m"], np.asarray(box["curve_t_s"]) * 1e9, color=colour, lw=1.0)
            ax.add_patch(
                plt.Rectangle(
                    (box["xmin_m"], box["tmin_s"] * 1e9),
                    box["xmax_m"] - box["xmin_m"],
                    (box["tmax_s"] - box["tmin_s"]) * 1e9,
                    fill=False,
                    edgecolor=colour,
                    linewidth=1.2,
                )
            )
            # Does the box cover the visible hyperbola near the apex?
            covers = (
                box["xmin_m"] <= box["apex_x_m"] <= box["xmax_m"]
                and box["tmin_s"] <= box["apex_t_s"] <= box["tmax_s"]
                and len(box["curve_x_m"]) >= 3
            )
            if flagged_here:
                coverage.append(
                    {
                        "seed": seed,
                        "name": box["name"],
                        "type": box["type"],
                        "covers_hyperbola": bool(covers),
                    }
                )
        ax.set_title(f"seed {seed}", fontsize=9)
        ax.set_xticks([])
        ax.tick_params(labelsize=7)
    for index in range(len(seeds), rows * cols):
        axes[index // cols, index % cols].axis("off")
    fig.suptitle(
        "Synthetic boxes on processed B-scans (red = flagged apex check)",
        fontsize=12,
    )
    fig.tight_layout()
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=130)
    plt.close(fig)

    # Ensure every flagged target appears in coverage (even if box missing).
    covered_names = {(c["seed"], c["name"]) for c in coverage}
    for row in flagged:
        key = (row["seed"], row["name"])
        if key not in covered_names:
            coverage.append(
                {
                    "seed": row["seed"],
                    "name": row["name"],
                    "type": row.get("type"),
                    "covers_hyperbola": False,
                    "note": "no physical box written for this target",
                }
            )
    return coverage, flagged, seeds


def load_real_train_thumbs(n=12, split_csv=DEFAULT_SPLIT, raw_root=DEFAULT_RAW, labels_dir=DEFAULT_LABELS):
    """Load n train void/utility images with boxes. Never touches test ids."""

    rows, test_ids = train_label_rows(split_csv)
    thumbs = []
    for row in rows:
        assert_not_test(row["id"], test_ids)
        img_path = Path(raw_root) / row["path"]
        lab_path = Path(labels_dir) / f"{row['id']}.txt"
        if not img_path.is_file() or not lab_path.is_file():
            continue
        with Image.open(img_path) as image:
            rgb = image.convert("RGB")
            draw = ImageDraw.Draw(rgb)
            w, h = rgb.size
            for line in lab_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                _c, xc, yc, bw, bh = map(float, line.split())
                xmin = (xc - bw / 2) * w
                ymin = (yc - bh / 2) * h
                xmax = (xc + bw / 2) * w
                ymax = (yc + bh / 2) * h
                draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 40, 40), width=2)
            thumbs.append((row["id"], rgb.copy()))
        if len(thumbs) >= n:
            break
    return thumbs


def plot_synthetic_vs_real(
    out_root=DEFAULT_OUT,
    stats_path=DEFAULT_STATS,
    figure_path=DEFAULT_MATCH,
):
    """Histograms and side-by-side thumbnails."""

    stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    real_w = np.asarray(stats["pooled"]["widths"], dtype=float)
    real_h = np.asarray(stats["pooled"]["heights"], dtype=float)

    synth_w, synth_h = [], []
    labels_root = Path(out_root) / "labels"
    for split_name in ("train", "val"):
        folder = labels_root / split_name
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.txt")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                _c, _xc, _yc, bw, bh = map(float, line.split())
                synth_w.append(bw)
                synth_h.append(bh)
    synth_w = np.asarray(synth_w, dtype=float)
    synth_h = np.asarray(synth_h, dtype=float)

    synth_imgs = sorted((Path(out_root) / "images" / "train").glob("*.jpg"))[:12]
    real_thumbs = load_real_train_thumbs(12)

    fig = plt.figure(figsize=(14, 11))
    ax_w = fig.add_subplot(3, 2, 1)
    ax_h = fig.add_subplot(3, 2, 2)
    ax_w.hist(real_w, bins=20, alpha=0.6, label="real train", density=True, color="#1f77b4")
    if synth_w.size:
        ax_w.hist(synth_w, bins=20, alpha=0.6, label="synthetic", density=True, color="#ff7f0e")
    ax_w.set_xlabel("width / image width")
    ax_w.set_title(
        f"Width med real={np.median(real_w):.3f} "
        f"synth={np.median(synth_w) if synth_w.size else float('nan'):.3f}"
    )
    ax_w.legend(fontsize=8)
    ax_h.hist(real_h, bins=20, alpha=0.6, label="real train", density=True, color="#1f77b4")
    if synth_h.size:
        ax_h.hist(synth_h, bins=20, alpha=0.6, label="synthetic", density=True, color="#ff7f0e")
    ax_h.set_xlabel("height / image height")
    ax_h.set_title(
        f"Height med real={np.median(real_h):.3f} "
        f"synth={np.median(synth_h) if synth_h.size else float('nan'):.3f}"
    )
    ax_h.legend(fontsize=8)

    for index in range(12):
        ax = fig.add_subplot(3, 12, 13 + index)
        if index < len(synth_imgs):
            with Image.open(synth_imgs[index]) as image:
                ax.imshow(np.asarray(image.convert("L")), cmap="gray", vmin=0, vmax=255)
            ax.set_title("S", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for index in range(12):
        ax = fig.add_subplot(3, 12, 25 + index)
        if index < len(real_thumbs):
            ax.imshow(np.asarray(real_thumbs[index][1]))
            ax.set_title("R", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Synthetic vs real train (S = synthetic, R = real)", fontsize=12)
    fig.tight_layout()
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=130)
    plt.close(fig)
    return {
        "real_width_median": float(np.median(real_w)),
        "real_height_median": float(np.median(real_h)),
        "synth_width_median": float(np.median(synth_w)) if synth_w.size else None,
        "synth_height_median": float(np.median(synth_h)) if synth_h.size else None,
        "n_synth_boxes": int(synth_w.size),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--n-crops", type=int, default=CROPS_PER_SCENE)
    parser.add_argument("--n-scenes", type=int, default=200)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)

    coverage = None
    match = None
    result = None
    if not args.skip_render:
        result = render_dataset(
            scenes_root=args.scenes,
            out_root=args.out,
            stats_path=args.stats,
            n_crops=args.n_crops,
            n_scenes=args.n_scenes,
            seed0=args.seed0,
        )
        print(f"failed crops: {len(result['failed_crops'])}")
    if not args.skip_figures:
        coverage, flagged, seeds = plot_boxes_check(args.scenes, DEFAULT_BOX_CHECK, args.n_scenes)
        print(f"box-check scenes: {seeds}")
        for row in coverage:
            print(
                f"flagged seed {row['seed']} {row['name']} ({row.get('type')}): "
                f"covers={row['covers_hyperbola']} {row.get('note', '')}"
            )
        if (Path(args.out) / "images" / "train").is_dir():
            match = plot_synthetic_vs_real(args.out, args.stats, DEFAULT_MATCH)
            print("medians:", match)
    return result, coverage, match


if __name__ == "__main__":
    main()
