"""Render field_400mhz scenes into a 224 x 224 greyscale YOLO set.

Processing matches ``01c`` / ``02b`` (no reference subtraction on the image).
Boxes follow the hand-labelling rule on the kink-fixed predicted hyperbola,
including rocks that pass a visibility and amplitude check, and a downward
ringing extension under each apex. Real train labels supply the box-size
distribution. Test-split Mendeley ids are never opened.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from collections import defaultdict
from io import BytesIO
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
DEFAULT_ROCKS = ROOT / "figures" / "synthetic_rocks_check.png"

FREQUENCY_HZ = 4.0e8
PERIOD_S = 1.0 / FREQUENCY_HZ
PERIOD_NS = 1e9 * PERIOD_S
IMAGE_SIZE = 224
ENVELOPE_FRAC = 0.20
ROCK_REL_FRAC = 0.25
ROCK_NOISE_FACTOR = 5.0
CROPS_PER_SCENE = 6
NEGATIVE_FRAC = 0.20
MAX_CROP_DRAWS = 50
TRAIN_SEED_MAX = 179
JPEG_Q_LO, JPEG_Q_HI = 75, 95
# Aspect rule: crop width and height from independent real width/height
# fractions, then anisotropic bilinear resize to 224 x 224.


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
    assert stats["utility"]["n_boxes"] == 113, stats["utility"]["n_boxes"]
    assert stats["void"]["n_boxes"] == 109, stats["void"]["n_boxes"]
    assert abs(stats["utility"]["width_median"] - 0.49) < 0.02
    assert abs(stats["utility"]["height_median"] - 0.55) < 0.02
    assert abs(stats["void"]["width_median"] - 0.46) < 0.02
    assert abs(stats["void"]["height_median"] - 0.48) < 0.02
    return stats


def midpoints_for_scene(scene):
    """Antenna midpoints for each trace, in metres."""

    return np.asarray(
        scene["midpoint_x0"]
        + scene["trace_step_m"] * np.arange(int(scene["n_traces"])),
        dtype=float,
    )


def processing_windows(frequency_hz=FREQUENCY_HZ):
    """Dewow and AGC windows in ns, and the band-pass edges in Hz."""

    dewow_ns, agc_ns = scenes_mod.processing_windows_ns(frequency_hz)
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


def usable_trace_mask(processed, relative=1e-3):
    """True for traces with enough energy to enter a crop.

    The white strip on the left of some check panels was a plot-extent bug
    (boxes in midpoint metres drawn on a receiver-x axis), not blank data.
    This mask still keeps near-empty padded columns out of crops if any appear.
    """

    values = np.asarray(processed, dtype=np.float64)
    col_std = values.std(axis=0)
    positive = col_std[col_std > 0]
    if positive.size == 0:
        return np.zeros(col_std.shape, dtype=bool)
    floor = float(np.median(positive)) * float(relative)
    return col_std > floor


def fix_hyperbola_branch(x_m, t_s, apex_x, time_window_s):
    """Keep the continuous hyperbola branch through the apex."""

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


def ringing_bottom_sample(envelope, apex_trace, apex_sample, period_samples, threshold):
    """Lowest sample of stacked ringing under the apex.

    Walk down the apex trace. Stop when the envelope stays below ``threshold``
    for ``period_samples`` samples, or at the window end. Never returns a
    sample above ``apex_sample``.
    """

    n_samples = envelope.shape[0]
    apex_trace = int(np.clip(apex_trace, 0, envelope.shape[1] - 1))
    apex_sample = int(np.clip(apex_sample, 0, n_samples - 1))
    period_samples = max(int(period_samples), 1)
    below = 0
    last_loud = apex_sample
    for sample in range(apex_sample, n_samples):
        value = _sample_envelope(envelope, apex_trace, sample)
        if threshold > 0 and value < threshold:
            below += 1
            if below >= period_samples:
                return max(last_loud, apex_sample)
        else:
            below = 0
            last_loud = sample
    return max(last_loud, apex_sample)


def box_from_hyperbola(
    x_m,
    t_s,
    apex_x,
    apex_t_s,
    midpoints,
    dt,
    picks,
    envelope,
    time_window_s,
    period_s=PERIOD_S,
    envelope_frac=ENVELOPE_FRAC,
    extend_ringing=True,
):
    """Physical box for one visible object, in metres and seconds."""

    x_fixed, t_fixed, _indices = fix_hyperbola_branch(x_m, t_s, apex_x, time_window_s)
    if x_fixed.size == 0:
        return None

    picks = np.asarray(picks, dtype=int)
    shift_s = picks * float(dt)
    midpoints = np.asarray(midpoints, dtype=float)
    trace_index = np.array(
        [int(np.argmin(np.abs(midpoints - float(x)))) for x in x_fixed],
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
    if midpoints.size >= 2:
        step = float(np.median(np.diff(midpoints)))
    else:
        step = 0.05
    margin_x = 2.0 * step

    xmin = float(x_used.min()) - margin_x
    xmax = float(x_used.max()) + margin_x
    tmin = float(apex_t_shifted) - half_period
    flank_bottom = float(t_used.max())
    period_samples = max(int(round(float(period_s) / float(dt))), 1)
    if extend_ringing:
        ring_sample = ringing_bottom_sample(
            envelope, apex_trace, apex_sample, period_samples, threshold
        )
        ringing_bottom = float(ring_sample) * float(dt)
        # Never move the bottom up relative to the flank bottom.
        body_bottom = max(flank_bottom, ringing_bottom)
    else:
        ringing_bottom = flank_bottom
        body_bottom = flank_bottom
    tmax = body_bottom + half_period

    x_lo = float(midpoints[0])
    x_hi = float(midpoints[-1])
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
        "flank_bottom_s": flank_bottom,
        "ringing_bottom_s": float(ringing_bottom),
        "apex_x_m": float(apex_x),
        "apex_t_s": float(apex_t_shifted),
        "curve_x_m": [float(v) for v in x_used],
        "curve_t_s": [float(v) for v in t_used],
        "envelope_frac": float(envelope_frac),
        "apex_envelope": float(apex_env),
        "apex_trace": int(apex_trace),
        "apex_sample": int(apex_sample),
    }


def noise_level(envelope, midpoints, objects, clearance_traces=6):
    """Median absolute envelope in a target-free part of the record."""

    midpoints = np.asarray(midpoints, dtype=float)
    n_traces = midpoints.size
    blocked = np.zeros(n_traces, dtype=bool)
    step = float(np.median(np.diff(midpoints))) if n_traces > 1 else 1.0
    for obj in objects:
        centre = int(np.argmin(np.abs(midpoints - float(obj["x"]))))
        radius_traces = int(round(float(obj.get("radius_m", 0.05)) / step)) + clearance_traces
        lo = max(0, centre - radius_traces)
        hi = min(n_traces, centre + radius_traces + 1)
        blocked[lo:hi] = True
    free = np.flatnonzero(~blocked)
    if free.size == 0:
        free = np.arange(n_traces)
    return float(np.median(np.abs(envelope[:, free])))


def predict_rock_row(scene, rock):
    """Ground-truth style row for one rock, using ``01c.hyperbola_time``."""

    midpoints = midpoints_for_scene(scene)
    times = scenes_mod.hyperbola_time(scene, rock, midpoints)
    apex = float(scenes_mod.hyperbola_time(scene, rock, rock["x"]))
    direct = float(scenes_mod.direct_wave_time(scene))
    period = 1.0 / float(scene["frequency_hz"])
    visible, reason = scenes_mod.is_visible(
        apex, direct, period, scene["time_window_s"]
    )
    return {
        "name": rock["name"],
        "x": rock["x"],
        "top_depth_m": rock["top_depth_m"],
        "centre_depth_m": rock["centre_depth_m"],
        "radius_m": rock["radius_m"],
        "type": "rock",
        "material": rock.get("material"),
        "eps_r": rock.get("eps_r"),
        "apex_time_s": apex,
        "direct_wave_time_s": direct,
        "visible": bool(visible),
        "visibility_reason": reason,
        "hyperbola_x_m": [float(v) for v in midpoints],
        "hyperbola_t_s": [float(v) for v in np.asarray(times).reshape(-1)],
    }


def rock_passes_amplitude(apex_env, target_envs, noise):
    """Amplitude gate for labelling a rock as a clear hyperbola."""

    if not target_envs:
        median_target = 0.0
    else:
        median_target = float(np.median(target_envs))
    return (
        float(apex_env) >= ROCK_REL_FRAC * median_target
        and float(apex_env) >= ROCK_NOISE_FACTOR * float(noise)
    )


def load_scene_bundle(scene_dir):
    """Load scene.json, merged scan, reference, and run.json."""

    scene_dir = Path(scene_dir)
    scene = json.loads((scene_dir / "scene.json").read_text(encoding="utf-8"))
    run = json.loads((scene_dir / "run.json").read_text(encoding="utf-8"))
    scan = loader.load_gprmax(scene_dir / "scene_merged.h5")
    ref_scan = loader.load_gprmax(scene_dir / "reference.h5")
    return scene, run, scan, ref_scan


def physical_boxes_for_scene(scene_dir, envelope_frac=ENVELOPE_FRAC, extend_ringing=True):
    """Compute physical boxes for visible targets and clear rocks."""

    scene, run, scan, ref_scan = load_scene_bundle(scene_dir)
    processed, picks, windows = process_scene_array(
        scan.data, scan.dt, scene["frequency_hz"]
    )
    envelope = reference_envelope(
        scan.data, ref_scan.data[:, 0], scan.dt, picks, scene["frequency_hz"]
    )
    midpoints = midpoints_for_scene(scene)
    period_s = 1.0 / float(scene["frequency_hz"])
    objects = list(scene.get("targets", [])) + list(scene.get("rocks", []))
    noise = noise_level(envelope, midpoints, objects)

    target_boxes = []
    target_envs = []
    for target in scene["ground_truth"]["targets"]:
        if not target.get("visible", False):
            continue
        box = box_from_hyperbola(
            target["hyperbola_x_m"],
            target["hyperbola_t_s"],
            target["x"],
            target["apex_time_s"],
            midpoints,
            scan.dt,
            picks,
            envelope,
            scene["time_window_s"],
            period_s=period_s,
            envelope_frac=envelope_frac,
            extend_ringing=extend_ringing,
        )
        if box is None:
            continue
        box["name"] = target["name"]
        box["type"] = target["type"]
        box["kind"] = "target"
        target_boxes.append(box)
        target_envs.append(box["apex_envelope"])

    rock_boxes = []
    rock_rejected = []
    for rock in scene.get("rocks", []):
        row = predict_rock_row(scene, rock)
        if not row["visible"]:
            rock_rejected.append(
                {
                    "seed": scene["seed"],
                    "name": row["name"],
                    "reason": row["visibility_reason"] or "not visible",
                    "apex_envelope": None,
                }
            )
            continue
        box = box_from_hyperbola(
            row["hyperbola_x_m"],
            row["hyperbola_t_s"],
            row["x"],
            row["apex_time_s"],
            midpoints,
            scan.dt,
            picks,
            envelope,
            scene["time_window_s"],
            period_s=period_s,
            envelope_frac=envelope_frac,
            extend_ringing=extend_ringing,
        )
        if box is None:
            rock_rejected.append(
                {
                    "seed": scene["seed"],
                    "name": row["name"],
                    "reason": "no box from hyperbola",
                    "apex_envelope": None,
                }
            )
            continue
        if not rock_passes_amplitude(box["apex_envelope"], target_envs, noise):
            rock_rejected.append(
                {
                    "seed": scene["seed"],
                    "name": row["name"],
                    "reason": "amplitude gate",
                    "apex_envelope": box["apex_envelope"],
                    "median_target_env": float(np.median(target_envs)) if target_envs else 0.0,
                    "noise": noise,
                }
            )
            continue
        box["name"] = row["name"]
        box["type"] = "rock"
        box["kind"] = "rock"
        box["eps_r"] = row.get("eps_r")
        rock_boxes.append(box)

    boxes = target_boxes + rock_boxes
    payload = {
        "seed": scene["seed"],
        "preset": scene["preset"],
        "windows": windows,
        "envelope_frac": envelope_frac,
        "extend_ringing": bool(extend_ringing),
        "noise_level": noise,
        "boxes": boxes,
        "n_target_boxes": len(target_boxes),
        "n_rock_boxes": len(rock_boxes),
        "rocks_rejected": rock_rejected,
    }
    return payload, processed, picks, scan, scene, run, midpoints


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
    """Clip a pixel box to a crop. Return 'keep', 'clip', or 'reject'."""

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


def match_histogram(source_u8, reference_u8):
    """Match the grey histogram of ``source`` to ``reference`` (8-bit)."""

    source = np.asarray(source_u8, dtype=np.uint8)
    reference = np.asarray(reference_u8, dtype=np.uint8)
    src_counts = np.bincount(source.ravel(), minlength=256).astype(np.float64)
    ref_counts = np.bincount(reference.ravel(), minlength=256).astype(np.float64)
    src_cdf = np.cumsum(src_counts)
    ref_cdf = np.cumsum(ref_counts)
    if src_cdf[-1] <= 0 or ref_cdf[-1] <= 0:
        return source.copy()
    src_cdf /= src_cdf[-1]
    ref_cdf /= ref_cdf[-1]
    lut = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for level in range(256):
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[level]:
            ref_idx += 1
        lut[level] = ref_idx
    matched = lut[source]
    assert matched.dtype == np.uint8
    assert int(matched.min()) >= 0 and int(matched.max()) <= 255
    return matched


def appearance_stats(images):
    """Median std-dev and share of pixels within 20 grey of the image median."""

    stds = []
    near = []
    for image in images:
        arr = np.asarray(image, dtype=np.float64)
        stds.append(float(arr.std()))
        med = float(np.median(arr))
        near.append(float(np.mean(np.abs(arr - med) <= 20.0)))
    if not stds:
        return {"median_std": None, "frac_near_median": None, "n": 0}
    return {
        "median_std": float(np.median(stds)),
        "frac_near_median": float(np.mean(near)),
        "n": len(stds),
    }


def apply_realism_pre_jpeg(grey_u8, rng):
    """Noise, contrast, gain, jitter, marker lines. No JPEG yet."""

    image = Image.fromarray(np.asarray(grey_u8, dtype=np.uint8), mode="L")
    contrast = float(rng.uniform(0.85, 1.25))
    image = ImageEnhance.Contrast(image).enhance(contrast)
    brightness = float(rng.uniform(0.85, 1.15))
    image = ImageEnhance.Brightness(image).enhance(brightness)
    arr = np.asarray(image, dtype=np.float64)
    jittered = np.zeros_like(arr)
    for col in range(arr.shape[1]):
        shift = int(rng.integers(-1, 2))
        jittered[:, col] = np.roll(arr[:, col], shift)
    noise_level = float(rng.uniform(2.0, 12.0))
    noisy = jittered + rng.normal(0.0, noise_level, size=jittered.shape)
    noisy = np.clip(noisy, 0, 255)
    n_markers = int(rng.integers(0, 4))
    top = max(1, int(0.15 * noisy.shape[0]))
    for _ in range(n_markers):
        x = int(rng.integers(0, noisy.shape[1]))
        noisy[:top, x] = 255
    return noisy.astype(np.uint8), {
        "contrast": contrast,
        "brightness": brightness,
        "noise_level": noise_level,
        "n_markers": n_markers,
    }


def jpeg_compress(grey_u8, rng):
    """JPEG round-trip at a random quality in [75, 95]."""

    quality = int(rng.integers(JPEG_Q_LO, JPEG_Q_HI + 1))
    out = Image.fromarray(np.asarray(grey_u8, dtype=np.uint8), mode="L")
    buffer = BytesIO()
    out.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("L")), quality


def load_real_train_grey_pool(split_csv=DEFAULT_SPLIT, raw_root=DEFAULT_RAW, limit=80):
    """Train originals resized to 224, for histogram matching. No test ids."""

    rows, test_ids = train_label_rows(split_csv)
    # Also allow intact train images as appearance references.
    all_rows, _ = load_split_rows(split_csv)
    pool_rows = [
        row
        for row in all_rows
        if row["split"] == "train"
    ]
    images = []
    for row in pool_rows:
        assert_not_test(row["id"], test_ids)
        path = Path(raw_root) / row["path"]
        if not path.is_file():
            continue
        with Image.open(path) as image:
            grey = image.convert("L").resize(
                (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR
            )
            images.append(np.asarray(grey, dtype=np.uint8))
        if len(images) >= int(limit):
            break
    if not images:
        raise RuntimeError("no real train images found for histogram matching")
    return images


def physical_to_indices(box, midpoints, dt):
    """Map a physical box to float indices on the midpoint axis."""

    midpoints = np.asarray(midpoints, dtype=float)
    x0 = float(np.interp(box["xmin_m"], midpoints, np.arange(midpoints.size)))
    x1 = float(np.interp(box["xmax_m"], midpoints, np.arange(midpoints.size)))
    y0 = float(box["tmin_s"] / dt)
    y1 = float(box["tmax_s"] / dt)
    return x0, y0, x1, y1


def draw_crop(
    processed,
    midpoints,
    dt,
    boxes,
    stats,
    rng,
    want_negative,
    usable_mask=None,
):
    """Draw one crop window. Returns dict or None after MAX_CROP_DRAWS."""

    n_samples, n_traces = processed.shape
    if usable_mask is None:
        usable_mask = np.ones(n_traces, dtype=bool)
    usable = np.flatnonzero(usable_mask)
    if usable.size < 8:
        return None
    trace_lo = int(usable.min())
    trace_hi = int(usable.max()) + 1
    usable_width = trace_hi - trace_lo

    widths = np.asarray(stats["pooled"]["widths"], dtype=float)
    heights = np.asarray(stats["pooled"]["heights"], dtype=float)
    for _attempt in range(MAX_CROP_DRAWS):
        if want_negative or not boxes:
            w_frac = float(rng.choice(widths))
            h_frac = float(rng.choice(heights))
            crop_w = max(8, int(round(usable_width * w_frac)))
            crop_h = max(8, int(round(n_samples * h_frac)))
            crop_w = min(crop_w, usable_width)
            if usable_width - crop_w < 1 or n_samples - crop_h < 1:
                continue
            x0 = int(rng.integers(trace_lo, trace_hi - crop_w + 1))
            y0 = int(rng.integers(0, n_samples - crop_h + 1))
            crop = (x0, y0, x0 + crop_w, y0 + crop_h)
            if not np.all(usable_mask[x0 : x0 + crop_w]):
                continue
            ok = True
            for box in boxes:
                px = physical_to_indices(box, midpoints, dt)
                decision, _ = clip_box_to_crop(px, crop, min_area_frac=0.5)
                if decision != "reject":
                    ok = False
                    break
            if not ok:
                continue
            return {
                "crop": crop,
                "boxes_px": [],
                "box_kinds": [],
                "negative": True,
                "w_frac_draw": w_frac,
                "h_frac_draw": h_frac,
            }

        target = boxes[int(rng.integers(0, len(boxes)))]
        w_frac = float(np.clip(rng.choice(widths), 0.15, 0.95))
        h_frac = float(np.clip(rng.choice(heights), 0.15, 0.95))
        tx0, ty0, tx1, ty1 = physical_to_indices(target, midpoints, dt)
        box_w = max(tx1 - tx0, 1.0)
        box_h = max(ty1 - ty0, 1.0)
        crop_w = int(np.clip(round(box_w / w_frac), 8, usable_width))
        crop_h = int(np.clip(round(box_h / h_frac), 8, n_samples))
        slack_x = crop_w - box_w
        slack_y = crop_h - box_h
        if slack_x < 0 or slack_y < 0:
            continue
        x0 = int(round(tx0 - float(rng.uniform(0, max(slack_x, 1e-6)))))
        y0 = int(round(ty0 - float(rng.uniform(0, max(slack_y, 1e-6)))))
        x0 = int(np.clip(x0, trace_lo, trace_hi - crop_w))
        y0 = int(np.clip(y0, 0, n_samples - crop_h))
        crop = (x0, y0, x0 + crop_w, y0 + crop_h)
        if not np.all(usable_mask[x0 : x0 + crop_w]):
            continue

        kept = []
        kinds = []
        rejected = False
        for box in boxes:
            px = physical_to_indices(box, midpoints, dt)
            decision, clipped = clip_box_to_crop(px, crop, min_area_frac=0.5)
            if decision == "reject":
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
            ix0, iy0, ix1, iy1 = clipped
            kept.append((ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0))
            kinds.append(box.get("kind", "target"))
        if rejected or not kept:
            continue
        return {
            "crop": crop,
            "boxes_px": kept,
            "box_kinds": kinds,
            "negative": False,
            "w_frac_draw": w_frac,
            "h_frac_draw": h_frac,
        }
    return None


def render_crop_image(processed, crop, boxes_px, rng, real_pool):
    """Extract, resize, realism, histogram match, JPEG; map boxes to 224."""

    x0, y0, x1, y1 = crop
    patch = processed[y0:y1, x0:x1]
    grey = robust_to_uint8(patch)
    pil = Image.fromarray(grey, mode="L").resize(
        (IMAGE_SIZE, IMAGE_SIZE), resample=Image.Resampling.BILINEAR
    )
    arr = np.asarray(pil, dtype=np.uint8)
    arr, meta = apply_realism_pre_jpeg(arr, rng)
    before_match = arr.copy()
    reference = real_pool[int(rng.integers(0, len(real_pool)))]
    arr = match_histogram(arr, reference)
    after_match = arr.copy()
    arr, quality = jpeg_compress(arr, rng)
    meta["jpeg_quality"] = quality

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
    return arr, yolo_boxes, meta, before_match, after_match


def scene_split(seed):
    """train or val from the scene seed."""

    return "train" if int(seed) <= TRAIN_SEED_MAX else "val"


def ensure_synth_dirs(out_root):
    out_root = Path(out_root)
    for split_name in ("train", "val"):
        (out_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    return out_root


def _box_height_frac_on_record(box, n_samples, dt):
    """Height of a physical box as a fraction of the full record length."""

    return (box["tmax_s"] - box["tmin_s"]) / (n_samples * dt)


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
    # Clear previous crops so regeneration does not mix old files.
    for split_name in ("train", "val"):
        for path in (out_root / "images" / split_name).glob("*"):
            path.unlink()
        for path in (out_root / "labels" / split_name).glob("*"):
            path.unlink()

    windows_note = processing_windows()
    print(
        "Processing windows: "
        f"dewow={windows_note['dewow_ns']:.2f} ns, "
        f"agc={windows_note['agc_ns']:.2f} ns, "
        f"period={windows_note['period_ns']:.2f} ns, "
        f"band≈{windows_note['band_low_hz']/1e6:.0f}-"
        f"{2 * FREQUENCY_HZ / 1e6:.0f} MHz"
    )

    real_pool = load_real_train_grey_pool()
    real_app = appearance_stats(real_pool)
    before_match_images = []
    after_match_images = []

    manifest = []
    failed_crops = []
    rocks_labelled = []
    rocks_rejected = []
    heights_before = []
    heights_after = []
    n_target_boxes_total = 0
    n_rock_boxes_total = 0

    for seed in range(int(seed0), int(seed0) + int(n_scenes)):
        scene_dir = Path(scenes_root) / f"seed{seed}"
        if not (scene_dir / "scene_merged.h5").is_file():
            print(f"skip missing scene {scene_dir}")
            continue

        # Height median before ringing (targets+rocks with same amplitude rule).
        payload_before, processed, picks, scan, scene, run, midpoints = physical_boxes_for_scene(
            scene_dir, extend_ringing=False
        )
        for box in payload_before["boxes"]:
            heights_before.append(
                _box_height_frac_on_record(box, processed.shape[0], scan.dt)
            )

        payload, processed, picks, scan, scene, run, midpoints = physical_boxes_for_scene(
            scene_dir, extend_ringing=True
        )
        write_boxes_physical(scene_dir, payload)
        for box in payload["boxes"]:
            heights_after.append(
                _box_height_frac_on_record(box, processed.shape[0], scan.dt)
            )
            if box.get("kind") == "rock":
                rocks_labelled.append({"seed": seed, "name": box["name"], "box": box})
                n_rock_boxes_total += 1
            else:
                n_target_boxes_total += 1
        for row in payload.get("rocks_rejected", []):
            rocks_rejected.append(row)

        boxes = payload["boxes"]
        usable = usable_trace_mask(processed)
        split_name = scene_split(seed)
        n_neg = int(round(n_crops * NEGATIVE_FRAC))
        for crop_index in range(n_crops):
            rng = np.random.default_rng(10007 * int(seed) + 17 * crop_index + 3)
            want_negative = crop_index < n_neg
            drawn = draw_crop(
                processed,
                midpoints,
                scan.dt,
                boxes,
                stats,
                rng,
                want_negative,
                usable_mask=usable,
            )
            crop_id = f"seed{seed}_crop{crop_index:02d}"
            if drawn is None:
                failed_crops.append(crop_id)
                print(f"gave up on crop {crop_id}")
                continue
            # Guard: crop must not contain unusable columns.
            x0, y0, x1, y1 = drawn["crop"]
            assert np.all(usable[x0:x1]), crop_id
            arr, yolo_boxes, meta, before_m, after_m = render_crop_image(
                processed, drawn["crop"], drawn["boxes_px"], rng, real_pool
            )
            before_match_images.append(before_m)
            after_match_images.append(after_m)
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
            n_rock = sum(1 for kind in drawn["box_kinds"] if kind == "rock")
            n_target = len(drawn["box_kinds"]) - n_rock
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
                    "n_target_boxes": n_target,
                    "n_rock_boxes": n_rock,
                    "negative": int(drawn["negative"] or len(yolo_boxes) == 0),
                    **{f"realism_{k}": v for k, v in meta.items()},
                }
            )
        print(
            f"seed {seed}: {payload['n_target_boxes']} targets, "
            f"{payload['n_rock_boxes']} rocks, split={split_name}",
            flush=True,
        )

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

    before_app = appearance_stats(before_match_images[:200])
    after_app = appearance_stats(after_match_images[:200])
    summary = {
        "stats": stats,
        "manifest": manifest,
        "failed_crops": failed_crops,
        "windows": windows_note,
        "out_root": out_root,
        "rocks_labelled": rocks_labelled,
        "rocks_rejected": rocks_rejected,
        "n_target_boxes_physical": n_target_boxes_total,
        "n_rock_boxes_physical": n_rock_boxes_total,
        "height_median_before_ringing": float(np.median(heights_before)) if heights_before else None,
        "height_median_after_ringing": float(np.median(heights_after)) if heights_after else None,
        "appearance_real": real_app,
        "appearance_synth_before_match": before_app,
        "appearance_synth_after_match": after_app,
    }
    print(
        f"rocks labelled={len(rocks_labelled)} rejected={len(rocks_rejected)}; "
        f"height med before={summary['height_median_before_ringing']:.4f} "
        f"after={summary['height_median_after_ringing']:.4f}"
    )
    print("appearance real", real_app)
    print("appearance synth before hist-match", before_app)
    print("appearance synth after hist-match", after_app)
    return summary


def plot_boxes_check(
    scenes_root=DEFAULT_SCENES,
    figure_path=DEFAULT_BOX_CHECK,
    n_scenes=200,
):
    """Processed panels for every scene that has a flagged apex target."""

    flagged = flagged_apex_rows(scenes_root, n_scenes)
    seeds = sorted({row["seed"] for row in flagged})
    if len(seeds) < 20:
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
        payload, processed, picks, scan, scene, run, midpoints = physical_boxes_for_scene(
            scene_dir
        )
        write_boxes_physical(scene_dir, payload)
        time_ns = np.arange(processed.shape[0]) * scan.dt * 1e9
        limit = np.percentile(np.abs(processed), 99) or 1.0
        # Midpoint extent: matches box x and removes the false left white strip.
        ax.imshow(
            processed,
            extent=[midpoints[0], midpoints[-1], time_ns[-1], time_ns[0]],
            aspect="auto",
            cmap="gray",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ax.set_xlim(midpoints[0], midpoints[-1])
        for box in payload["boxes"]:
            flagged_here = (seed, box["name"]) in flagged_keys
            if flagged_here:
                colour = "#ff3300"
            elif box.get("kind") == "rock":
                colour = "#33aaff"
            else:
                colour = "#33ff66"
            ax.plot(
                box["curve_x_m"],
                np.asarray(box["curve_t_s"]) * 1e9,
                color=colour,
                lw=1.0,
            )
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
        "Synthetic boxes (red=flagged apex, green=target, blue=rock)",
        fontsize=12,
    )
    fig.tight_layout()
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=130)
    plt.close(fig)

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


def plot_rocks_check(
    scenes_root=DEFAULT_SCENES,
    figure_path=DEFAULT_ROCKS,
    n_scenes=200,
):
    """Six labelled rocks and six rejected rocks."""

    labelled = []
    rejected = []
    for seed in range(int(n_scenes)):
        scene_dir = Path(scenes_root) / f"seed{seed}"
        if not (scene_dir / "scene_merged.h5").is_file():
            continue
        payload, processed, picks, scan, scene, run, midpoints = physical_boxes_for_scene(
            scene_dir
        )
        for box in payload["boxes"]:
            if box.get("kind") == "rock":
                labelled.append((seed, box, processed, midpoints, scan.dt))
        for row in payload.get("rocks_rejected", []):
            rejected.append((seed, row, processed, midpoints, scan.dt, scene))
        if len(labelled) >= 6 and len(rejected) >= 6:
            break

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for index in range(6):
        ax = axes[0, index]
        if index < len(labelled):
            seed, box, processed, midpoints, dt = labelled[index]
            time_ns = np.arange(processed.shape[0]) * dt * 1e9
            limit = np.percentile(np.abs(processed), 99) or 1.0
            ax.imshow(
                processed,
                extent=[midpoints[0], midpoints[-1], time_ns[-1], time_ns[0]],
                aspect="auto",
                cmap="gray",
                vmin=-limit,
                vmax=limit,
            )
            ax.add_patch(
                plt.Rectangle(
                    (box["xmin_m"], box["tmin_s"] * 1e9),
                    box["xmax_m"] - box["xmin_m"],
                    (box["tmax_s"] - box["tmin_s"]) * 1e9,
                    fill=False,
                    edgecolor="#33aaff",
                    linewidth=1.2,
                )
            )
            ax.set_title(f"pass s{seed} {box['name']}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    for index in range(6):
        ax = axes[1, index]
        if index < len(rejected):
            seed, row, processed, midpoints, dt, scene = rejected[index]
            time_ns = np.arange(processed.shape[0]) * dt * 1e9
            limit = np.percentile(np.abs(processed), 99) or 1.0
            ax.imshow(
                processed,
                extent=[midpoints[0], midpoints[-1], time_ns[-1], time_ns[0]],
                aspect="auto",
                cmap="gray",
                vmin=-limit,
                vmax=limit,
            )
            ax.set_title(f"rej s{seed} {row['name']}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Rocks: labelled (top) vs rejected (bottom)", fontsize=12)
    fig.tight_layout()
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=130)
    plt.close(fig)
    return labelled, rejected


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
    """Histograms and a readable 4 x 6 thumbnail grid."""

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

    # Histograms on top; then a 4 x 6 grid (12 synth + 12 real).
    # 6 cols at ~2.5 cm each needs about 15 cm (~6 in) of width; use 16 in.
    fig = plt.figure(figsize=(16, 14))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 3.6], hspace=0.28)
    hist_gs = outer[0].subgridspec(1, 2, wspace=0.25)
    ax_w = fig.add_subplot(hist_gs[0, 0])
    ax_h = fig.add_subplot(hist_gs[0, 1])
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

    thumb_gs = outer[1].subgridspec(4, 6, wspace=0.12, hspace=0.35)
    for index in range(12):
        ax = fig.add_subplot(thumb_gs[index // 6, index % 6])
        if index < len(synth_imgs):
            path = synth_imgs[index]
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            draw = ImageDraw.Draw(rgb)
            lab = Path(out_root) / "labels" / "train" / (path.stem + ".txt")
            if lab.is_file():
                for line in lab.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    _c, xc, yc, bw, bh = map(float, line.split())
                    w, h = rgb.size
                    draw.rectangle(
                        [
                            (xc - bw / 2) * w,
                            (yc - bh / 2) * h,
                            (xc + bw / 2) * w,
                            (yc + bh / 2) * h,
                        ],
                        outline=(255, 40, 40),
                        width=2,
                    )
            ax.imshow(np.asarray(rgb))
            ax.set_title(f"S {path.stem}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for index in range(12):
        ax = fig.add_subplot(thumb_gs[2 + index // 6, index % 6])
        if index < len(real_thumbs):
            ax.imshow(np.asarray(real_thumbs[index][1]))
            ax.set_title(f"R {real_thumbs[index][0]}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Synthetic vs real train (S = synthetic, R = real)", fontsize=12)
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=120)
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
        labelled, rejected = plot_rocks_check(args.scenes, DEFAULT_ROCKS, args.n_scenes)
        print(f"rocks figure: labelled sample {len(labelled)}, rejected sample {len(rejected)}")
        if (Path(args.out) / "images" / "train").is_dir():
            match = plot_synthetic_vs_real(args.out, args.stats, DEFAULT_MATCH)
            print("medians:", match)
    return result, coverage, match


if __name__ == "__main__":
    main()
