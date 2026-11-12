"""Ground truth is the cylinder centre and the soil permittivity."""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ground_truth_from_model = importlib.import_module("01a_pipe_bscan_study").ground_truth_from_model

ROOT = Path(__file__).resolve().parents[1]


def test_pipe_template_records_centre_and_permittivity():
    truth = ground_truth_from_model((ROOT / "src" / "pipe.in").read_text())

    assert truth["eps_r"] == 6
    assert truth["target_x_m"] == 0.12
    assert truth["target_depth_m"] == 0.09
    assert truth["target_y_m"] == 0.08
    assert truth["target_radius_m"] == 0.01


def test_each_saved_model_has_a_ground_truth_file():
    # The full pipe study is gitignored; data/sample/ keeps one case as committed copies.
    models = list((ROOT / "data" / "sim" / "pipe_study").glob("*/pipe.in"))
    models = models or list((ROOT / "data" / "sample" / "sim" / "pipe_study").glob("*/pipe.in"))
    assert models
    for model in models:
        truth = ground_truth_from_model(model.read_text())
        saved = (model.parent / "ground_truth.json").read_text()
        assert f'"eps_r": {truth["eps_r"]}' in saved
        assert f'"target_x_m": {truth["target_x_m"]}' in saved
        assert f'"target_depth_m": {truth["target_depth_m"]}' in saved
