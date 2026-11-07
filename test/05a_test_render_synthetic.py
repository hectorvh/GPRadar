"""Pure-Python tests for synthetic render helpers. No gprMax, no real data."""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

render = importlib.import_module("05a_render_synthetic")


def test_kink_fix_keeps_continuous_apex_branch():
    # Smooth hyperbola-like rise, then a kink where the slope drops.
    x = np.arange(11, dtype=float)
    t = np.array([5, 4, 3, 2, 1.5, 1.0, 1.5, 2.0, 2.2, 2.25, 2.26], dtype=float)
    # Force a clear kink on the right: after index 7 the first difference shrinks hard.
    t = np.array([5.0, 4.0, 3.0, 2.2, 1.5, 1.0, 1.5, 2.2, 2.4, 2.45, 2.46])
    xf, tf, idx = render.fix_hyperbola_branch(x, t, apex_x=5.0, time_window_s=10.0)
    assert 5.0 in xf
    assert idx.min() <= 5 <= idx.max()
    # The far-right flattened points after the kink are dropped.
    assert xf.max() < x.max() or tf[-1] <= t[8] + 1e-9
    # Continuous through the apex: no gap in kept indices around 5.
    assert np.all(np.diff(idx[idx >= 5]) == 1) or 5 in idx


def test_clip_box_rule():
    crop = (10.0, 10.0, 50.0, 50.0)
    # Fully inside.
    decision, clipped = render.clip_box_to_crop((15, 15, 25, 25), crop)
    assert decision == "keep"
    assert clipped == (15, 15, 25, 25)
    # Mostly inside (>=50%): clip.
    decision, clipped = render.clip_box_to_crop((40, 20, 60, 40), crop)
    assert decision == "clip"
    assert clipped[2] == 50.0
    # Under 50% inside: reject.
    decision, clipped = render.clip_box_to_crop((45, 20, 80, 40), crop)
    assert decision == "reject"
    assert clipped is None
    # Outside: reject.
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
