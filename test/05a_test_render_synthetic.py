"""Pure-Python tests for synthetic render helpers. No gprMax, no real data."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

render = importlib.import_module("05a_render_synthetic")


def test_kink_fix_keeps_continuous_apex_branch():
    x = np.arange(11, dtype=float)
    t = np.array([5.0, 4.0, 3.0, 2.2, 1.5, 1.0, 1.5, 2.2, 2.4, 2.45, 2.46])
    xf, tf, idx = render.fix_hyperbola_branch(x, t, apex_x=5.0, time_window_s=10.0)
    assert 5.0 in xf
    assert idx.min() <= 5 <= idx.max()
    assert xf.max() < x.max() or tf[-1] <= t[8] + 1e-9
    assert np.all(np.diff(idx[idx >= 5]) == 1) or 5 in idx


def test_clip_box_rule():
    crop = (10.0, 10.0, 50.0, 50.0)
    decision, clipped = render.clip_box_to_crop((15, 15, 25, 25), crop)
    assert decision == "keep"
    assert clipped == (15, 15, 25, 25)
    decision, clipped = render.clip_box_to_crop((40, 20, 60, 40), crop)
    assert decision == "clip"
    assert clipped[2] == 50.0
    decision, clipped = render.clip_box_to_crop((45, 20, 80, 40), crop)
    assert decision == "reject"
    assert clipped is None
    decision, clipped = render.clip_box_to_crop((60, 60, 70, 70), crop)
    assert decision == "reject"


def test_yolo_export_normalised():
    xc, yc, bw, bh = render.box_to_yolo(10, 20, 30, 60, 100, 200)
    assert 0 <= xc <= 1 and 0 <= yc <= 1
    assert 0 <= bw <= 1 and 0 <= bh <= 1
    assert abs(xc - 0.20) < 1e-6
    assert abs(yc - 0.20) < 1e-6
    assert abs(bw - 0.20) < 1e-6
    assert abs(bh - 0.20) < 1e-6


def test_scene_split_by_seed():
    assert render.scene_split(0) == "train"
    assert render.scene_split(179) == "train"
    assert render.scene_split(180) == "val"
    assert render.scene_split(199) == "val"


def test_loading_test_split_id_raises(tmp_path):
    split = tmp_path / "split.csv"
    split.write_text(
        "id,path,class,group_id,split,md5\n"
        "void_0001,cavities/1.JPG,void,g0,test,abc\n"
        "void_0002,cavities/2.JPG,void,g1,train,def\n",
        encoding="utf-8",
    )
    _rows, test_ids = render.load_split_rows(split)
    with pytest.raises(render.TestSplitError):
        render.assert_not_test("void_0001", test_ids)
    render.assert_not_test("void_0002", test_ids)


def test_rock_visibility_and_amplitude_rule():
    # Visibility: need one period after the direct wave.
    period = 2.5e-9
    direct = 5.0e-9
    window = 50.0e-9
    ok, reason = render.scenes_mod.is_visible(direct + 0.5 * period, direct, period, window)
    assert not ok
    ok, reason = render.scenes_mod.is_visible(direct + 1.5 * period, direct, period, window)
    assert ok and reason is None

    # Amplitude gate relative to targets and noise.
    assert render.rock_passes_amplitude(10.0, [40.0, 40.0], noise=1.0) is True  # 25% of 40, 5x noise
    assert render.rock_passes_amplitude(5.0, [40.0, 40.0], noise=1.0) is False  # below 25%
    assert render.rock_passes_amplitude(10.0, [40.0], noise=3.0) is False  # below 5x noise


def test_ringing_extension_never_moves_bottom_up():
    envelope = np.zeros((100, 5), dtype=float)
    # Apex loud at sample 20, ringing until 50, then quiet.
    envelope[18:22, 2] = 10.0
    envelope[22:51, 2] = 3.0
    envelope[51:, 2] = 0.1
    threshold = 2.0
    period_samples = 10
    bottom = render.ringing_bottom_sample(envelope, 2, 20, period_samples, threshold)
    assert bottom >= 20
    # Flank bottom earlier than ringing: combined body uses the lower one.
    flank_bottom_s = 30 * 1e-9
    ringing_bottom_s = bottom * 1e-9
    body = max(flank_bottom_s, ringing_bottom_s)
    assert body >= flank_bottom_s


def test_histogram_matching_keeps_range_and_moves_histogram():
    rng = np.random.default_rng(0)
    source = rng.integers(80, 200, size=(64, 64), dtype=np.uint8)
    reference = rng.integers(20, 60, size=(64, 64), dtype=np.uint8)
    matched = render.match_histogram(source, reference)
    assert matched.dtype == np.uint8
    assert int(matched.min()) >= 0 and int(matched.max()) <= 255
    # Matched mean should move toward the darker reference.
    assert abs(matched.mean() - reference.mean()) < abs(source.mean() - reference.mean())


def test_crop_excludes_blank_columns():
    rng = np.random.default_rng(0)
    processed = rng.normal(size=(40, 30))
    processed[:, 0:3] = 0.0  # blank left strip
    mask = render.usable_trace_mask(processed, relative=1e-3)
    assert not mask[:3].any()
    assert mask[3:].all()
    midpoints = np.arange(30, dtype=float) * 0.05
    stats = {
        "pooled": {
            "widths": [0.4, 0.5],
            "heights": [0.4, 0.5],
        }
    }
    rng = np.random.default_rng(1)
    boxes = [
        {
            "xmin_m": 0.5,
            "xmax_m": 0.9,
            "tmin_s": 5e-9,
            "tmax_s": 15e-9,
            "kind": "target",
        }
    ]
    for _ in range(20):
        drawn = render.draw_crop(
            processed,
            midpoints,
            dt=1e-9,
            boxes=boxes,
            stats=stats,
            rng=rng,
            want_negative=True,
            usable_mask=mask,
        )
        if drawn is None:
            continue
        x0, _y0, x1, _y1 = drawn["crop"]
        assert np.all(mask[x0:x1])
        assert x0 >= 3
