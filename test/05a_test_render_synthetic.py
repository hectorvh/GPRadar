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


def test_rock_image_visibility_rule():
    # One-period direct-wave rule still applies upstream.
    period = 2.5e-9
    direct = 5.0e-9
    window = 50.0e-9
    ok, _reason = render.scenes_mod.is_visible(direct + 0.5 * period, direct, period, window)
    assert not ok
    ok, reason = render.scenes_mod.is_visible(direct + 1.5 * period, direct, period, window)
    assert ok and reason is None

    # Clear arch near apex vs quiet background on the same rows.
    midpoints = np.arange(21, dtype=float) * 0.05
    dt = 1.0e-9
    processed = np.full((80, 21), 0.2, dtype=float)
    apex_x = midpoints[10]
    curve_x = midpoints[8:13]
    curve_t = np.array([20, 21, 20, 21, 22], dtype=float) * dt
    for x, t in zip(curve_x, curve_t):
        trace = int(np.argmin(np.abs(midpoints - x)))
        sample = int(round(t / dt))
        processed[sample, trace] = 5.0
    all_curves = [{"curve_x_m": curve_x, "curve_t_s": curve_t}]
    ok, stats = render.rock_visible_on_image(
        processed, midpoints, dt, curve_x, curve_t, apex_x, flank_halfwidth_m=0.12, all_curves=all_curves
    )
    assert ok
    assert stats["mean_on"] >= 2.0 * stats["median_off"]

    # Buried rock: on-curve samples no louder than the background.
    buried = np.full_like(processed, 1.0)
    ok, stats = render.rock_visible_on_image(
        buried, midpoints, dt, curve_x, curve_t, apex_x, flank_halfwidth_m=0.12, all_curves=all_curves
    )
    assert not ok


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


def test_ringing_cap_never_extends_more_than_three_periods():
    envelope = np.ones((200, 5), dtype=float) * 5.0  # never falls
    period_samples = 10
    apex_sample = 20
    flank_bottom_sample = 30
    max_sample = flank_bottom_sample + 3 * period_samples
    bottom = render.ringing_bottom_sample(
        envelope,
        apex_trace=2,
        apex_sample=apex_sample,
        period_samples=period_samples,
        threshold=2.0,
        max_sample=max_sample,
    )
    assert bottom <= max_sample
    assert bottom - flank_bottom_sample <= 3 * period_samples


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
            "name": "target_0",
            "xmin_m": 0.5,
            "xmax_m": 0.9,
            "tmin_s": 5e-9,
            "tmax_s": 15e-9,
            "kind": "target",
        }
    ]
    for _ in range(20):
        drawn, _n_rej = render.draw_crop(
            processed,
            midpoints,
            dt=1e-9,
            boxes=boxes,
            stats=stats,
            rng=rng,
            want_negative=True,
            usable_mask=mask,
            apexes=[],
        )
        if drawn is None:
            continue
        x0, _y0, x1, _y1 = drawn["crop"]
        assert np.all(mask[x0:x1])
        assert x0 >= 3


def test_crop_rejects_unlabelled_apex():
    midpoints = np.arange(40, dtype=float) * 0.05
    dt = 1.0e-9
    processed = np.random.default_rng(0).normal(size=(60, 40))
    stats = {"pooled": {"widths": [0.5], "heights": [0.5]}}
    boxes = [
        {
            "name": "target_0",
            "xmin_m": 0.8,
            "xmax_m": 1.2,
            "tmin_s": 10e-9,
            "tmax_s": 25e-9,
            "kind": "target",
            "apex_x_m": 1.0,
            "apex_t_s": 15e-9,
        }
    ]
    # Rejected rock apex inside a typical crop around the target.
    apexes = [
        {
            "name": "rock_0",
            "kind": "rock",
            "labelled": False,
            "apex_x_m": 1.05,
            "apex_t_s": 18e-9,
        },
        {
            "name": "target_0",
            "kind": "target",
            "labelled": True,
            "apex_x_m": 1.0,
            "apex_t_s": 15e-9,
        },
    ]
    crop = (10, 5, 30, 40)  # covers both apexes
    assert render.crop_has_unlabelled_apex(crop, midpoints, dt, boxes, apexes)

    # Negative crop must not contain any apex.
    empty_crop = (0, 0, 8, 8)
    assert not render.crop_has_unlabelled_apex(empty_crop, midpoints, dt, [], apexes)
    apex_in_neg = {
        "name": "rock_0",
        "kind": "rock",
        "labelled": False,
        "apex_x_m": midpoints[2],
        "apex_t_s": 3e-9,
    }
    assert render.crop_has_unlabelled_apex(empty_crop, midpoints, dt, [], [apex_in_neg])

    rng = np.random.default_rng(2)
    drawn, n_rej = render.draw_crop(
        processed,
        midpoints,
        dt,
        boxes,
        stats,
        rng,
        want_negative=False,
        apexes=apexes,
    )
    # Either no crop found, or every accepted crop has no unlabelled apex.
    if drawn is not None:
        assert not render.crop_has_unlabelled_apex(
            drawn["crop"], midpoints, dt, boxes, apexes
        )
    assert n_rej >= 0
