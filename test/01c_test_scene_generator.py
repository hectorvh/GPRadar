"""Scene draws, the grid limit, and the gprMax input file. No gprMax run."""

import importlib
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

scenes = importlib.import_module("01c_scene_generator")
derive_grid = scenes.derive_grid
ground_truth = scenes.ground_truth
hyperbola_time = scenes.hyperbola_time
sample_scene = scenes.sample_scene
write_model = scenes.write_model

C = 299792458.0


def test_same_seed_gives_the_same_scene():
    first = sample_scene("field_80mhz", 0)
    second = sample_scene("field_80mhz", 0)

    assert first == second
    assert first["seed"] == 0
    assert sample_scene("field_80mhz", 1) != first


def test_targets_stay_inside_and_do_not_overlap():
    scene = sample_scene("field_400mhz", 2)
    circles = scene["targets"] + scene["rocks"]
    assert scene["targets"]

    for index, item in enumerate(circles):
        assert item["x"] - item["radius_m"] > scene["pml_m"]
        assert item["x"] + item["radius_m"] < scene["domain_x"] - scene["pml_m"]
        assert item["centre_y"] - item["radius_m"] > scene["pml_m"]
        assert item["centre_y"] + item["radius_m"] < scene["surface_y"]
        for other in circles[index + 1 :]:
            distance = math.hypot(item["x"] - other["x"], item["centre_y"] - other["centre_y"])
            needed = item["radius_m"] + other["radius_m"] + scene["clearance_m"]
            assert distance >= needed - 1e-9

    for target in scene["targets"]:
        assert target["x"] - target["radius_m"] >= scene["midpoint_x0"] + scene["clearance_m"] - 1e-9
        assert target["x"] + target["radius_m"] <= scene["midpoint_x1"] - scene["clearance_m"] + 1e-9


def test_cell_size_is_at_most_lambda_over_10():
    for preset in ("field_80mhz", "field_400mhz"):
        scenes.check_preset(preset)
        scene = sample_scene(preset, 3)
        # The fixed cell may be 0.1 mm coarser than lambda_min/10 so the step
        # and the offset land on whole cells. derive_grid itself stays at or under the limit.
        assert scene["dx"] <= scene["lambda_min_m"] / 10.0 + 1e-4
        grid = derive_grid(
            scene["frequency_hz"],
            scene["bound_eps_max"],
            scene["bound_bottom_m"],
            scene["tx_rx_offset_m"],
            f_max_factor=scene["f_max_factor"],
            time_margin=scene["time_margin"],
            pml_cells=scene["pml_cells"],
        )
        assert grid["dx"] <= grid["lambda_min_m"] / 10.0


def test_one_grid_per_preset():
    for preset in ("field_80mhz", "field_400mhz"):
        drawn = [sample_scene(preset, seed) for seed in range(6)]
        first = drawn[0]
        for scene in drawn[1:]:
            assert scene["dx"] == first["dx"]
            assert scene["time_window_s"] == first["time_window_s"]
            assert scene["domain_x"] == first["domain_x"]
            assert scene["domain_y"] == first["domain_y"]
            assert scene["n_traces"] == first["n_traces"]
        step_cells = first["trace_step_m"] / first["dx"]
        offset_cells = first["tx_rx_offset_m"] / first["dx"]
        assert abs(step_cells - round(step_cells)) < 1e-9
        assert abs(offset_cells - round(offset_cells)) < 1e-9


def test_reference_model_has_no_targets_or_rocks(tmp_path):
    scene = sample_scene("field_80mhz", 5)
    path = write_model(scene, tmp_path / "reference.in", reference=True)
    text = path.read_text()

    assert "#cylinder: " not in text
    assert "rock_" not in text
    assert "pvc" not in text
    assert "pec" not in text
    assert "free_space" not in text
    assert "#box: " in text
    assert f"#domain: {scene['domain_x']:.6f} {scene['domain_y']:.6f} inf" in text


def test_visibility_threshold():
    direct = 20e-9
    period = 12.5e-9
    window = 100e-9
    below, reason_below = scenes.is_visible(direct + period - 1e-12, direct, period, window)
    above, reason_above = scenes.is_visible(direct + period, direct, period, window)
    late, reason_late = scenes.is_visible(window, direct, period, window)

    assert below is False
    assert reason_below == "less than one period after the direct wave"
    assert above is True
    assert reason_above is None
    assert late is False
    assert reason_late == "outside the time window"

    scene = sample_scene("field_80mhz", 7)
    truth = ground_truth(scene)
    period = 1.0 / scene["frequency_hz"]
    for target in truth["targets"]:
        visible, reason = scenes.is_visible(
            target["apex_time_s"], truth["direct_wave_time_s"], period, scene["time_window_s"]
        )
        assert target["visible"] is visible
        assert target["visibility_reason"] == reason
        assert target["direct_wave_time_s"] == truth["direct_wave_time_s"]


def test_model_file_contains_the_commands(tmp_path):
    scene = sample_scene("field_80mhz", 4)
    path = write_model(scene, tmp_path / "scene.in")
    text = path.read_text()

    assert "#domain_mode: TM" in text
    assert f"#domain: {scene['domain_x']:.6f} {scene['domain_y']:.6f} inf" in text
    assert f"#dx_dy_dz: {scene['dx']:.6f} {scene['dx']:.6f} {scene['dx']:.6f}" in text
    assert f"#time_window: {scene['time_window_s']:.8e}" in text
    assert "#pml_cells: 10 10 0 10 10 0" in text
    assert f"#waveform: ricker 1 {scene['frequency_hz']:.8e} my_ricker" in text
    assert "#hertzian_dipole: z " in text
    assert "#rx: " in text
    assert f"#src_steps: {scene['trace_step_m']:.6f} 0 0" in text
    assert f"#rx_steps: {scene['trace_step_m']:.6f} 0 0" in text
    assert "#box: " in text
    assert "#cylinder: " in text
    assert "half_space" in text
    for target in scene["targets"]:
        assert target["material"] in text


def test_homogeneous_apex_matches_the_bistatic_path():
    delay = math.sqrt(2.0) / 80e6
    velocity = C / 2.0
    scene = {
        "tx_rx_offset_m": 0.48,
        "antenna_y": 1.0,
        "surface_y": 1.0,
        "ricker_delay_s": delay,
        "column": [
            {"name": "half_space", "y0": 0.0, "y1": 1.0, "velocity_m_s": velocity, "eps_r": 4.0}
        ],
    }
    target = {"x": 5.0, "centre_y": 0.5, "radius_m": 0.1}
    path = 2.0 * math.hypot(0.24, 0.5) - 2.0 * 0.1
    expected = path / velocity + delay

    assert abs(hyperbola_time(scene, target, 5.0) - expected) < 1e-9
    off_axis = hyperbola_time(scene, target, np.array([5.0, 6.0]))
    assert off_axis[1] > off_axis[0]
