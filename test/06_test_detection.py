"""Pure-Python tests for detection prepare/train guards and matching."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

prep = importlib.import_module("06a_prepare_real_yolo")
train_mod = importlib.import_module("06b_train_detectors")
eval_mod = importlib.import_module("06c_evaluate_detectors")


def _fake_train_groups():
    # 8 groups per class, 1 image each.
    groups = {}
    for class_name in prep.CLASSES:
        groups[class_name] = [
            (f"{class_name}_g{i}", [{"id": f"{class_name}_{i}", "class": class_name, "group_id": f"{class_name}_g{i}"}])
            for i in range(8)
        ]
    return groups


def test_validation_and_subsets_by_group_stratified():
    groups = _fake_train_groups()
    val_ids, pool_ids = prep.carve_validation(groups, seed=0, fraction=0.25)
    # 25% of 8 = 2 groups per class.
    for class_name in prep.CLASSES:
        n_val = sum(1 for gid, _ in groups[class_name] if gid in val_ids)
        n_pool = sum(1 for gid, _ in groups[class_name] if gid in pool_ids)
        assert n_val == 2
        assert n_pool == 6
    assert val_ids.isdisjoint(pool_ids)

    subsets = prep.nested_fraction_subsets(groups, pool_ids, seeds=(0, 1))
    for seed, fracs in subsets.items():
        assert fracs[0.25].issubset(fracs[1.0])
        assert fracs[1.0] == pool_ids
        for class_name in prep.CLASSES:
            n25 = sum(1 for gid, _ in groups[class_name] if gid in fracs[0.25])
            assert n25 == 2  # 25% of 6 pool groups rounded


def test_test_id_raises_in_06a_and_06b_paths():
    test_ids = {"void_9999"}
    with pytest.raises(prep.TestSplitError):
        prep.assert_not_test("void_9999", test_ids)
    prep.assert_not_test("void_0001", test_ids)
    # 06b guard uses the same helper via verify + assert.
    with pytest.raises(prep.TestSplitError):
        train_mod.prep.assert_not_test("void_9999", test_ids)


def test_confidence_iou_matching_toy():
    gt = [(0, 0, 10, 10), (20, 20, 30, 30)]
    pred = [(1, 1, 9, 9), (40, 40, 50, 50)]  # one hit, one FP
    tp, fp, fn = eval_mod.match_detections(gt, pred, iou_thresh=0.5)
    assert tp == 1
    assert fp == 1
    assert fn == 1
    # Perfect overlap.
    tp, fp, fn = eval_mod.match_detections([(0, 0, 10, 10)], [(0, 0, 10, 10)], iou_thresh=0.5)
    assert (tp, fp, fn) == (1, 0, 0)
    # No overlap.
    tp, fp, fn = eval_mod.match_detections([(0, 0, 5, 5)], [(10, 10, 15, 15)], iou_thresh=0.5)
    assert (tp, fp, fn) == (0, 1, 1)
