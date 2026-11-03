"""Field-scale 2D gprMax scenes, with hyperbola travel times.

A scene is one parameter dict. ``sample_scene`` draws it, ``write_model``
writes the ``.in`` file in the same form as ``pipe.in``, and ``ground_truth``
predicts the apex and the hyperbola. Every scene in a preset uses the same
cell size, time window, domain and trace count. ``derive_grid`` checks that
those fixed values still cover the preset.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "scenes.yaml"
C = 299792458.0
# Three periods. The 2 ns dewow and AGC windows on the 1.5 GHz pipe are three cycles.
PROCESSING_PERIODS = 3.0


def load_config(path=None):
    """Read the preset file."""

    config_path = CONFIG_PATH if path is None else Path(path)
    with config_path.open() as handle:
        return yaml.safe_load(handle)


def ricker_delay(frequency_hz):
    """Time of the gprMax Ricker peak, in seconds.

    gprMax sets chi = sqrt(2) / f and evaluates the pulse at t - chi, so the
    peak is not at the start of the record.
    """

    frequency_hz = float(frequency_hz)
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    return math.sqrt(2.0) / frequency_hz


def processing_windows_ns(frequency_hz):
    """Dewow and AGC windows, in nanoseconds: three periods of the centre frequency."""

    period_ns = 1e9 / float(frequency_hz)
    window_ns = PROCESSING_PERIODS * period_ns
    return window_ns, window_ns


def derive_grid(
    frequency_hz,
    eps_max,
    deepest_bottom_m,
    tx_rx_offset_m,
    *,
    f_max_factor,
    time_margin,
    pml_cells,
):
    """Cell size and time window from the slowest soil and the deepest target.

    The cell is lambda_min / 10. lambda_min = v_min / f_max, v_min = c /
    sqrt(eps_max), and f_max is ``f_max_factor`` times the centre frequency.
    The time window is the bistatic two-way time to ``deepest_bottom_m`` at
    v_min, plus the air path from an antenna two cells above the surface,
    plus the Ricker delay, with ``time_margin`` on the propagation time.
    """

    if eps_max <= 1:
        raise ValueError("eps_max must be greater than 1")
    if deepest_bottom_m <= 0:
        raise ValueError("deepest_bottom_m must be positive")
    f_max = float(f_max_factor) * float(frequency_hz)
    v_min = C / math.sqrt(float(eps_max))
    lambda_min = v_min / f_max
    # 1 micrometre, and never coarser than lambda_min/10. The written dx then
    # round-trips through the .in file.
    dx = math.floor((lambda_min / 10.0) * 1e6) / 1e6
    if dx <= 0:
        raise ValueError("cell size rounded to zero")
    antenna_height = 2.0 * dx
    soil_path = 2.0 * math.hypot(float(tx_rx_offset_m) / 2.0, float(deepest_bottom_m))
    air_path = 2.0 * antenna_height
    propagation = soil_path / v_min + air_path / C
    delay = ricker_delay(frequency_hz)
    time_window = propagation * (1.0 + float(time_margin)) + delay
    return {
        "f_max_hz": f_max,
        "f_max_factor": float(f_max_factor),
        "v_min_m_s": v_min,
        "lambda_min_m": lambda_min,
        "dx": dx,
        "antenna_height_m": antenna_height,
        "time_window_s": time_window,
        "ricker_delay_s": delay,
        "pml_cells": int(pml_cells),
        "pml_m": int(pml_cells) * dx,
        "deepest_bottom_m": float(deepest_bottom_m),
        "eps_max": float(eps_max),
    }


def check_preset(preset, config=None):
    """Fail if the fixed grid is coarser than the preset bound.

    ``dx`` may be up to 0.1 mm coarser than lambda_min/10 so that a round
    cell size can keep the step and the offset on whole cells. The 80 MHz
    cell is 0.0375 m against a derived 0.037474 m.
    """

    config = load_config() if config is None else config
    if preset not in config["presets"]:
        raise KeyError(f"unknown preset {preset}")
    spec = config["presets"][preset]
    dx = float(spec["dx"])
    step = float(spec["trace_step_m"])
    offset = float(spec["tx_rx_offset_m"])
    window = float(spec["time_window_s"])
    domain_y = float(spec["domain_y_m"])
    profile = float(spec["profile_length_m"])
    eps_max = float(spec["eps_max"])
    bottom = float(spec["top_depth_max_m"]) + 2.0 * float(config["targets"]["radius_max_m"])
    derived = derive_grid(
        spec["frequency_hz"],
        eps_max,
        bottom,
        offset,
        f_max_factor=config["f_max_factor"],
        time_margin=config["time_margin"],
        pml_cells=config["pml_cells"],
    )
    limit = derived["lambda_min_m"] / 10.0
    if dx > limit + 1e-4:
        raise ValueError(
            f"{preset}: dx {dx:.6f} m is coarser than lambda_min/10 "
            f"({limit:.6f} m) by more than 0.1 mm"
        )
    if window + 1e-12 < derived["time_window_s"]:
        raise ValueError(
            f"{preset}: time window {window * 1e9:.3f} ns does not cover the bound "
            f"{derived['time_window_s'] * 1e9:.3f} ns"
        )
    _whole_cells(step, dx, f"{preset} trace step")
    _whole_cells(offset, dx, f"{preset} Tx-Rx offset")
    _whole_cells(domain_y, dx, f"{preset} domain height")
    _whole_cells(profile, step, f"{preset} profile length")
    pml = int(config["pml_cells"]) * dx
    air = max(0.10, 4.0 * dx)
    soil = domain_y - 2.0 * pml - 2.0 * dx - air
    if soil + 1e-9 < bottom:
        raise ValueError(
            f"{preset}: domain height {domain_y:.4f} m leaves {soil:.3f} m of soil, "
            f"short of the {bottom:.3f} m target bottom"
        )
    return derived


def fixed_layout(preset, config=None):
    """Domain and survey shared by every scene in the preset."""

    config = load_config() if config is None else config
    derived = check_preset(preset, config)
    spec = config["presets"][preset]
    dx = float(spec["dx"])
    step = float(spec["trace_step_m"])
    offset = float(spec["tx_rx_offset_m"])
    profile = float(spec["profile_length_m"])
    domain_y = float(spec["domain_y_m"])
    pml_cells = int(config["pml_cells"])
    pml = pml_cells * dx
    n_steps = int(round(profile / step))
    src_x = pml + dx
    midpoint_x0 = src_x + offset / 2.0
    midpoint_x1 = midpoint_x0 + n_steps * step
    antenna_height = 2.0 * dx
    air = max(0.10, 4.0 * dx)
    soil_depth = domain_y - 2.0 * pml - antenna_height - air
    surface_y = pml + soil_depth
    last_rx = src_x + n_steps * step + offset
    domain_x = last_rx + pml + dx
    nx = int(round(domain_x / dx))
    ny = int(round(domain_y / dx))
    layout = {
        "dx": dx,
        "dy": dx,
        "dz": dx,
        "nx": nx,
        "ny": ny,
        "nz": 1,
        "domain_x": nx * dx,
        "domain_y": ny * dx,
        "pml_cells": pml_cells,
        "pml_m": pml,
        "trace_step_m": step,
        "tx_rx_offset_m": offset,
        "profile_length_m": n_steps * step,
        "n_traces": n_steps + 1,
        "src_x": src_x,
        "midpoint_x0": midpoint_x0,
        "midpoint_x1": midpoint_x1,
        "antenna_height_m": antenna_height,
        "soil_depth_m": soil_depth,
        "surface_y": surface_y,
        "antenna_y": surface_y + antenna_height,
        "src_y": surface_y + antenna_height,
        "time_window_s": float(spec["time_window_s"]),
        "ricker_delay_s": derived["ricker_delay_s"],
        "f_max_hz": derived["f_max_hz"],
        "f_max_factor": derived["f_max_factor"],
        "v_min_m_s": derived["v_min_m_s"],
        "lambda_min_m": derived["lambda_min_m"],
        "bound_bottom_m": derived["deepest_bottom_m"],
        "bound_eps_max": derived["eps_max"],
    }
    return layout


def direct_wave_time(scene):
    """Arrival of the direct air wave, in seconds.

    The gprMax Ricker peaks at sqrt(2)/f, and the air path between the
    antennas is the offset divided by c.
    """

    return float(scene["ricker_delay_s"]) + float(scene["tx_rx_offset_m"]) / C


def is_visible(apex_s, direct_s, period_s, window_s):
    """Whether a reflection can be separated from the direct wave.

    Visible means the apex is at least one centre period after the direct
    wave and still inside the time window. The reason is None when it is.
    """

    if float(apex_s) < float(direct_s) + float(period_s):
        return False, "less than one period after the direct wave"
    if float(apex_s) >= float(window_s):
        return False, "outside the time window"
    return True, None


def format_grid(name, grid):
    """One block of derived grid numbers."""

    period_ns = 1e9 / (grid["f_max_hz"] / grid["f_max_factor"])
    return "\n".join(
        [
            f"{name}",
            f"  eps_max {grid['eps_max']:.4g}, deepest bottom {grid['deepest_bottom_m']:.3f} m",
            f"  f_max {grid['f_max_factor']:g} x centre = {grid['f_max_hz'] / 1e6:.1f} MHz",
            f"  v_min {grid['v_min_m_s']:.6g} m/s, lambda_min {grid['lambda_min_m']:.4f} m",
            f"  dx {grid['dx']:.4f} m (lambda_min/10), PML {grid['pml_cells']} cells ({grid['pml_m']:.3f} m)",
            f"  antenna height {grid['antenna_height_m']:.4f} m, Ricker delay {grid['ricker_delay_s'] * 1e9:.2f} ns",
            f"  time window {grid['time_window_s'] * 1e9:.2f} ns, centre period {period_ns:.2f} ns",
        ]
    )


def sample_scene(preset, seed, config=None):
    """Draw one scene. The same preset and seed return the same dict."""

    config = load_config() if config is None else config
    if preset not in config["presets"]:
        raise KeyError(f"unknown preset {preset}")
    spec = config["presets"][preset]
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    soil = config["soil"]
    target_spec = config["targets"]
    rock_spec = config["rocks"]

    layout = fixed_layout(preset, config)
    eps_min = float(spec["eps_min"])
    eps_max_limit = float(spec["eps_max"])
    n_layers = int(rng.integers(soil["extra_layers_min"], soil["extra_layers_max"] + 1))
    half = _material(rng, "half_space", eps_min, eps_max_limit, soil["sigma_min"], soil["sigma_max"])
    layers = []
    for index in range(n_layers):
        layer = _material(
            rng,
            f"layer_{index + 1}",
            eps_min,
            eps_max_limit,
            soil["sigma_min"],
            soil["sigma_max"],
        )
        layer["thickness_m"] = float(
            rng.uniform(soil["layer_thickness_min_m"], soil["layer_thickness_max_m"])
        )
        layers.append(layer)

    n_targets = int(rng.integers(target_spec["count_min"], target_spec["count_max"] + 1))
    n_rocks = int(rng.integers(rock_spec["count_min"], rock_spec["count_max"] + 1))
    kinds = [str(rng.choice(target_spec["types"])) for _ in range(n_targets)]
    rock_eps = [float(rng.uniform(rock_spec["eps_min"], rock_spec["eps_max"])) for _ in range(n_rocks)]

    # A cylinder has to cover a cell centre. Half a cell diagonal is dx/sqrt(2);
    # 0.75 dx is just past that, wherever the centre sits.
    dx = layout["dx"]
    cell_radius = 0.75 * dx
    clearance = float(config["clearance_m"])
    occupied = []
    targets = []
    for index, kind in enumerate(kinds):
        radius_min = max(float(target_spec["radius_min_m"]), cell_radius)
        wall = None
        if kind == "pvc":
            wall = max(2.0 * dx, 0.01)
            radius_min = max(radius_min, wall + cell_radius)
            if radius_min >= target_spec["radius_max_m"]:
                wall = dx
                radius_min = max(float(target_spec["radius_min_m"]), cell_radius, wall + cell_radius)
        # The x limits depend on the radius, so they are applied inside _place
        # by passing the full midpoint span and then rejecting the ends below.
        placed = _place(
            rng,
            radius_min,
            float(target_spec["radius_max_m"]),
            float(spec["top_depth_min_m"]),
            float(spec["top_depth_max_m"]),
            layout["midpoint_x0"] + clearance,
            layout["midpoint_x1"] - clearance,
            occupied,
            clearance,
        )
        if placed is None:
            raise RuntimeError(f"could not place target {index} for seed {seed}")
        x, top_depth, centre_depth, radius = placed
        material, eps_r = _target_material(kind, target_spec["pvc_eps_r"])
        target = {
            "name": f"target_{index}",
            "type": kind,
            "material": material,
            "eps_r": eps_r,
            "x": x,
            "top_depth_m": top_depth,
            "centre_depth_m": centre_depth,
            "radius_m": radius,
        }
        if kind == "pvc":
            target["wall_m"] = wall
            target["inner_radius_m"] = radius - wall
        targets.append(target)
        occupied.append(target)

    soil_depth = layout["soil_depth_m"]
    deepest_bottom = max(item["centre_depth_m"] + item["radius_m"] for item in targets)
    _fit_layers(layers, soil_depth)
    surface_y = layout["surface_y"]
    column = _column(half, layers, surface_y)
    for layer in layers:
        band = next(item for item in column if item["name"] == layer["name"])
        layer["y0"] = band["y0"]
        layer["y1"] = band["y1"]
    for target in targets:
        target["centre_y"] = surface_y - target["centre_depth_m"]

    rocks = _place_rocks(
        rng,
        n_rocks,
        rock_eps,
        rock_spec,
        targets,
        occupied,
        clearance,
        soil_depth,
        deepest_bottom,
        cell_radius,
        layout["midpoint_x0"],
        layout["midpoint_x1"],
        surface_y,
    )

    scene = {
        "preset": preset,
        "seed": int(seed),
        "frequency_hz": float(spec["frequency_hz"]),
        "time_margin": float(config["time_margin"]),
        "clearance_m": clearance,
        "eps_max": max([half["eps_r"], *(layer["eps_r"] for layer in layers), *rock_eps]),
        "deepest_bottom_m": deepest_bottom,
        "half_space": half,
        "layers": layers,
        "column": column,
        "targets": targets,
        "rocks": rocks,
        **layout,
    }
    _check_scene(scene)
    return scene


def write_model(scene, path, reference=False):
    """Write a 2D gprMax input file. Returns the path.

    ``reference=True`` keeps the soil and drops the targets and the rocks,
    so one trace of that file is the direct wave and the horizontal reflections.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = [] if reference else scene["targets"]
    rocks = [] if reference else scene["rocks"]
    title = f"{scene['preset']} seed {scene['seed']}"
    if reference:
        title += " reference"
    lines = [
        f"#title: {title}",
        "#domain_mode: TM",
        f"#domain: {_m(scene['domain_x'])} {_m(scene['domain_y'])} inf",
        f"#dx_dy_dz: {_m(scene['dx'])} {_m(scene['dy'])} {_m(scene['dz'])}",
        f"#time_window: {scene['time_window_s']:.8e}",
        f"#pml_cells: {scene['pml_cells']} {scene['pml_cells']} 0 {scene['pml_cells']} {scene['pml_cells']} 0",
        "",
    ]
    for material in [scene["half_space"], *scene["layers"], *_rock_materials(rocks), *_pvc_materials(targets)]:
        lines.append(
            "#material: "
            f"{_m(material['eps_r'])} {_m(material['sigma_s_m'])} 1 0 {material['name']}"
        )
    lines.append(f"#waveform: ricker 1 {scene['frequency_hz']:.8e} my_ricker")
    lines.append(
        f"#hertzian_dipole: z {_m(scene['src_x'])} {_m(scene['src_y'])} inf my_ricker"
    )
    rx_x = scene["src_x"] + scene["tx_rx_offset_m"]
    lines.append(f"#rx: {_m(rx_x)} {_m(scene['src_y'])} inf")
    lines.append(f"#src_steps: {_m(scene['trace_step_m'])} 0 0")
    lines.append(f"#rx_steps: {_m(scene['trace_step_m'])} 0 0")
    lines.append("")
    # The half-space fills the model up to the surface, including the bottom
    # PML, which is how pipe.in fills its half-space. Later objects replace it.
    lines.append(
        f"#box: 0 0 0 {_m(scene['domain_x'])} {_m(scene['surface_y'])} inf {scene['half_space']['name']}"
    )
    for layer in reversed(scene["layers"]):
        lines.append(
            "#box: "
            f"0 {_m(layer['y0'])} 0 {_m(scene['domain_x'])} {_m(layer['y1'])} inf {layer['name']}"
        )
    for rock in rocks:
        lines.append(_cylinder(rock["x"], rock["centre_y"], rock["radius_m"], rock["name"]))
    for target in targets:
        lines.append(_cylinder(target["x"], target["centre_y"], target["radius_m"], target["material"]))
        if target["type"] == "pvc":
            lines.append(
                _cylinder(target["x"], target["centre_y"], target["inner_radius_m"], "free_space")
            )
    path.write_text("\n".join(lines) + "\n")
    return path


def hyperbola_time(scene, target, x):
    """Predicted two-way time, in seconds, at antenna-midpoint positions ``x``.

    Each leg is the refracted path from that antenna to the near side of the
    cylinder. The times include the air gap, the soil layers, the Tx-Rx
    offset, and the Ricker delay.
    """

    xs = np.atleast_1d(np.asarray(x, dtype=float))
    times = np.empty(xs.shape, dtype=float)
    offset = float(scene["tx_rx_offset_m"])
    delay = float(scene["ricker_delay_s"])
    bands = _bands(scene)
    for index, mid in enumerate(xs.reshape(-1)):
        src = float(mid) - offset / 2.0
        rx = float(mid) + offset / 2.0
        times.reshape(-1)[index] = (
            _leg(scene, target, src, bands) + _leg(scene, target, rx, bands) + delay
        )
    if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
        return float(times.reshape(-1)[0])
    return times


def ground_truth(scene):
    """Per-target position and the predicted apex and hyperbola.

    ``hyperbola_x_m`` are the antenna midpoints. ``hyperbola_t_s`` is the
    predicted two-way time at those midpoints.
    """

    midpoints = scene["midpoint_x0"] + scene["trace_step_m"] * np.arange(scene["n_traces"])
    rows = []
    for target in scene["targets"]:
        times = hyperbola_time(scene, target, midpoints)
        apex = float(hyperbola_time(scene, target, target["x"]))
        direct = direct_wave_time(scene)
        visible, reason = is_visible(
            apex, direct, 1.0 / scene["frequency_hz"], scene["time_window_s"]
        )
        rows.append(
            {
                "name": target["name"],
                "x": target["x"],
                "top_depth_m": target["top_depth_m"],
                "centre_depth_m": target["centre_depth_m"],
                "radius_m": target["radius_m"],
                "type": target["type"],
                "material": target["material"],
                "apex_time_s": apex,
                "direct_wave_time_s": direct,
                "visible": visible,
                "visibility_reason": reason,
                "hyperbola_x_m": [float(value) for value in midpoints],
                "hyperbola_t_s": [float(value) for value in np.asarray(times).reshape(-1)],
            }
        )
    return {"direct_wave_time_s": direct_wave_time(scene), "targets": rows}


def save_scene(scene, path):
    """Write the ``.in`` file and the scene JSON beside it."""

    model = write_model(scene, path)
    payload = dict(scene)
    payload["ground_truth"] = ground_truth(scene)
    model.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    return model


def run_bscan(model_path, n_traces):
    """Run gprMax and merge the traces. A finished merge is left as it is.

    Returns the merged path, the wall time in seconds (None if skipped), and
    the parsed cell count and iteration count.
    """

    model_path = Path(model_path)
    merged = model_path.with_name(model_path.stem + "_merged.h5")
    if (
        merged.exists()
        and merged.stat().st_mtime >= model_path.stat().st_mtime
        and _trace_count(merged) == int(n_traces)
    ):
        print(f"skip {model_path.name}: {merged.name} already exists", flush=True)
        return merged, None, _geometry_from_log(model_path.with_suffix(".log"))

    base = model_path.with_suffix("")
    command = [
        sys.executable,
        "-m",
        "gprMax",
        str(model_path),
        "-n",
        str(n_traces),
        "--geometry-fixed",
        "-o",
        str(base),
    ]
    log_path = model_path.with_suffix(".log")
    print(" ".join(command), flush=True)
    started = time.perf_counter()
    with log_path.open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    _merge_traces(base, n_traces, merged)
    wall = time.perf_counter() - started
    return merged, wall, _geometry_from_log(log_path)


def plot_scene(merged, scene, truth, figure_path):
    """Raw and processed B-scans with the predicted hyperbolas."""

    import importlib

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(ROOT / "src"))
    loader = importlib.import_module("02a_gprmax_loader")
    processing = importlib.import_module("02b_preprocessing")
    scan = loader.load_gprmax(merged)
    if scan.data.shape[1] != scene["n_traces"]:
        raise ValueError(
            f"{merged} has {scan.data.shape[1]} traces, the scene has {scene['n_traces']}"
        )
    dewow_ns, agc_ns = processing_windows_ns(scene["frequency_hz"])
    shifted, picks = processing.time_zero(scan.data, scan.dt)
    dewowed = processing.dewow(shifted, scan.dt, dewow_ns)
    cleaned = processing.remove_background(dewowed)
    low, high = processing.default_band(scene["frequency_hz"], 0.5 / scan.dt)
    filtered = processing.bandpass(cleaned, scan.dt, low, high)
    gained = processing.gain_agc(filtered, scan.dt, agc_ns)

    position = np.asarray(truth["targets"][0]["hyperbola_x_m"])
    time_ns = np.arange(scan.data.shape[0]) * scan.dt * 1e9
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False, constrained_layout=True)
    _panel(axes[0], scan.data, position, time_ns, "Raw")
    _panel(axes[1], gained, position, time_ns, "Processed")
    colours = ("#d62728", "#ff7f0e", "#1f77b4")
    for target, colour in zip(truth["targets"], colours):
        curve_ns = np.asarray(target["hyperbola_t_s"]) * 1e9
        axes[0].plot(position, curve_ns, color=colour, lw=1.2, label=target["type"])
        shift_ns = picks[: curve_ns.size] * scan.dt * 1e9
        axes[1].plot(position, curve_ns - shift_ns, color=colour, lw=1.2, label=target["type"])
    for axis in axes:
        axis.legend(loc="lower right", fontsize=8)
        axis.set_xlabel("Antenna midpoint [m]")
    axes[0].set_ylabel("Time [ns]")
    axes[1].set_ylabel("Time after time zero [ns]")
    figure.suptitle(
        f"{scene['preset']} seed {scene['seed']}: dewow {dewow_ns:.1f} ns, AGC {agc_ns:.1f} ns",
        fontsize=12,
    )
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)
    return scan, picks, dewow_ns, agc_ns


def _panel(axis, data, position, time_ns, title):
    limit = np.percentile(np.abs(np.asarray(data)), 99)
    if limit == 0:
        limit = 1.0
    axis.imshow(
        data,
        extent=[position[0], position[-1], time_ns[-1], time_ns[0]],
        aspect="auto",
        cmap="seismic",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axis.set_title(title)


def run_reference(scene, model_dir):
    """One trace of the same soil with no targets and no rocks.

    The earth is laterally uniform, so that trace is the direct wave and the
    horizontal reflections on every trace of the B-scan.
    """

    model_dir = Path(model_dir)
    infile = model_dir / "reference.in"
    write_model(scene, infile, reference=True)
    output = model_dir / "reference.h5"
    if output.exists() and output.stat().st_mtime >= infile.stat().st_mtime:
        print("skip reference.h5: already exists", flush=True)
        return output, None
    command = [
        sys.executable,
        "-m",
        "gprMax",
        str(infile),
        "-n",
        "1",
        "-o",
        str(output),
    ]
    log_path = model_dir / "reference.log"
    print(" ".join(command), flush=True)
    started = time.perf_counter()
    with log_path.open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    if not output.exists():
        raise FileNotFoundError(output)
    return output, wall


def subtract_reference(scan, reference):
    """Remove the direct wave by subtracting the one-trace reference."""

    background = np.asarray(reference.data[:, 0], dtype=float)
    data = np.asarray(scan.data, dtype=float)
    if background.shape[0] != data.shape[0]:
        raise ValueError(
            f"reference has {background.shape[0]} samples, the B-scan has {data.shape[0]}"
        )
    return data - background[:, np.newaxis]


def apex_check(scan, scene, truth, reference):
    """Envelope peak on the reference-subtracted apex trace.

    The search is half a period either side of the prediction. The reference
    trace has already removed the direct wave.
    """

    from scipy.signal import hilbert

    period = 1.0 / scene["frequency_hz"]
    data = subtract_reference(scan, reference)
    rows = []
    for target in truth["targets"]:
        midpoints = np.asarray(target["hyperbola_x_m"])
        index = int(np.argmin(np.abs(midpoints - target["x"])))
        envelope = np.abs(hilbert(data[:, index]))
        times = np.arange(data.shape[0]) * scan.dt
        predicted = float(target["apex_time_s"])
        gate = (times >= predicted - 0.5 * period) & (times <= predicted + 0.5 * period)
        if not np.any(gate):
            raise ValueError(f"{target['name']} prediction falls outside the record")
        local = np.where(gate)[0]
        peak = local[int(np.argmax(envelope[local]))]
        picked = float(times[peak])
        on_edge = peak == local[0] or peak == local[-1]
        rows.append(
            {
                "name": target["name"],
                "type": target["type"],
                "x": target["x"],
                "visible": bool(target["visible"]),
                "visibility_reason": target["visibility_reason"],
                "predicted_ns": predicted * 1e9,
                "picked_ns": picked * 1e9,
                "error_ns": (picked - predicted) * 1e9,
                "quarter_period_ns": 0.25 * period * 1e9,
                "on_window_edge": bool(on_edge),
            }
        )
    return rows


def visibility_report(preset, n=1000, seed=0, config=None):
    """Share of targets whose apex is at least one period after the direct wave."""

    config = load_config() if config is None else config
    totals = {}
    shown = {}
    for index in range(int(n)):
        scene = sample_scene(preset, int(seed) + index, config)
        direct = direct_wave_time(scene)
        period = 1.0 / scene["frequency_hz"]
        for target in scene["targets"]:
            apex = hyperbola_time(scene, target, target["x"])
            visible, _reason = is_visible(apex, direct, period, scene["time_window_s"])
            kind = target["type"]
            totals[kind] = totals.get(kind, 0) + 1
            shown[kind] = shown.get(kind, 0) + int(visible)
    return {
        kind: {"count": totals[kind], "visible": shown[kind], "share": shown[kind] / totals[kind]}
        for kind in sorted(totals)
    }


def preset_bound(preset, config=None):
    """Grid at the slowest soil and the deepest target bottom the preset allows."""

    config = load_config() if config is None else config
    spec = config["presets"][preset]
    bottom = float(spec["top_depth_max_m"]) + 2.0 * float(config["targets"]["radius_max_m"])
    return derive_grid(
        spec["frequency_hz"],
        spec["eps_max"],
        bottom,
        spec["tx_rx_offset_m"],
        f_max_factor=config["f_max_factor"],
        time_margin=config["time_margin"],
        pml_cells=config["pml_cells"],
    )


def _material(rng, name, eps_min, eps_max, sigma_min, sigma_max):
    return {
        "name": name,
        "eps_r": float(rng.uniform(eps_min, eps_max)),
        "sigma_s_m": float(rng.uniform(sigma_min, sigma_max)),
    }


def _target_material(kind, pvc_eps):
    if kind == "metal":
        return "pec", None
    if kind == "pvc":
        return "pvc", float(pvc_eps)
    if kind == "air":
        return "free_space", 1.0
    raise ValueError(f"unknown target type {kind}")


def _place(rng, radius_min, radius_max, depth_min, depth_max, x_lo, x_hi, occupied, clearance):
    for _ in range(400):
        radius = float(rng.uniform(radius_min, radius_max))
        if depth_max <= depth_min or x_hi - x_lo <= 2.0 * radius:
            return None
        top_depth = float(rng.uniform(depth_min, depth_max))
        centre_depth = top_depth + radius
        x = float(rng.uniform(x_lo + radius, x_hi - radius))
        if _fits(x, centre_depth, radius, occupied, clearance):
            return x, top_depth, centre_depth, radius
    return None


def _fits(x, centre_depth, radius, occupied, clearance):
    for other in occupied:
        distance = math.hypot(x - other["x"], centre_depth - other["centre_depth_m"])
        if distance < radius + other["radius_m"] + clearance:
            return False
    return True


def _fit_layers(layers, soil_depth):
    """Keep at least 0.3 m of half-space under the layered stack."""

    limit = max(soil_depth - 0.3, 0.0)
    total = sum(layer["thickness_m"] for layer in layers)
    if total > limit > 0:
        scale = limit / total
        for layer in layers:
            layer["thickness_m"] *= scale


def _column(half, layers, surface_y):
    """Soil units from the bottom of the model up to the surface.

    ``layers`` is top-first. The half-space is whatever is left underneath.
    """

    cursor = surface_y
    built = []
    for layer in layers:
        y1 = cursor
        y0 = cursor - layer["thickness_m"]
        built.append(_band(layer, y0, y1))
        cursor = y0
    built.append(_band(half, 0.0, cursor))
    built.reverse()
    return built


def _band(material, y0, y1):
    eps_r = float(material["eps_r"])
    return {
        "name": material["name"],
        "eps_r": eps_r,
        "sigma_s_m": float(material["sigma_s_m"]),
        "y0": float(y0),
        "y1": float(y1),
        "velocity_m_s": C / math.sqrt(eps_r),
    }


def _half_width(scene, target):
    """Lateral offset at which the hyperbola reaches the time window."""

    window = scene["time_window_s"]
    origin = target["x"]

    def arrival(offset):
        return hyperbola_time(scene, target, origin + offset)

    if arrival(0.0) >= window:
        raise RuntimeError(f"{target['name']} apex is outside the time window")
    lo = 0.0
    hi = scene["trace_step_m"]
    while arrival(hi) < window:
        hi *= 2.0
        if hi > 1000.0:
            break
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if arrival(mid) < window:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _place_rocks(
    rng,
    n_rocks,
    rock_eps,
    rock_spec,
    _targets,
    occupied,
    clearance,
    soil_depth,
    deepest_bottom,
    cell_radius,
    x_lo,
    x_hi,
    surface_y,
):
    if n_rocks == 0:
        return []
    rocks = []
    depth_hi = max(0.06, min(deepest_bottom, soil_depth) - float(rock_spec["radius_max_m"]))
    for index, eps_r in enumerate(rock_eps):
        placed = _place(
            rng,
            max(float(rock_spec["radius_min_m"]), cell_radius),
            float(rock_spec["radius_max_m"]),
            0.05,
            depth_hi,
            x_lo + clearance,
            x_hi - clearance,
            occupied,
            clearance,
        )
        if placed is None:
            break
        x, top_depth, centre_depth, radius = placed
        if centre_depth + radius > soil_depth:
            continue
        rock = {
            "name": f"rock_{index}",
            "type": "rock",
            "material": f"rock_{index}",
            "eps_r": float(eps_r),
            "sigma_s_m": 0.0,
            "x": x,
            "top_depth_m": top_depth,
            "centre_depth_m": centre_depth,
            "centre_y": surface_y - centre_depth,
            "radius_m": radius,
        }
        rocks.append(rock)
        occupied.append(rock)
    return rocks


def _check_scene(scene):
    pml = scene["pml_m"]
    circles = scene["targets"] + scene["rocks"]
    for index, item in enumerate(circles):
        if item["centre_y"] + item["radius_m"] >= scene["surface_y"]:
            raise RuntimeError(f"{item['name']} intersects the surface")
        if item["centre_y"] - item["radius_m"] <= pml:
            raise RuntimeError(f"{item['name']} intersects the bottom PML")
        if item["x"] - item["radius_m"] <= pml or item["x"] + item["radius_m"] >= scene["domain_x"] - pml:
            raise RuntimeError(f"{item['name']} intersects the side PML")
        for other in circles[index + 1 :]:
            distance = math.hypot(item["x"] - other["x"], item["centre_y"] - other["centre_y"])
            needed = item["radius_m"] + other["radius_m"] + scene["clearance_m"]
            if distance < needed - 1e-9:
                raise RuntimeError(f"{item['name']} overlaps {other['name']}")
    last_rx = scene["src_x"] + (scene["n_traces"] - 1) * scene["trace_step_m"] + scene["tx_rx_offset_m"]
    if last_rx >= scene["domain_x"] - pml:
        raise RuntimeError("receiver steps into the PML")
    if scene["antenna_y"] >= scene["domain_y"] - pml:
        raise RuntimeError("antenna is inside the top PML")


def _bands(scene):
    air_top = max(scene["antenna_y"], scene["surface_y"]) + 1.0
    bands = [(scene["surface_y"], air_top, C)]
    for layer in scene["column"]:
        bands.append((layer["y0"], layer["y1"], layer["velocity_m_s"]))
    return bands


def _leg(scene, target, x_ant, bands):
    hit_x, hit_y = _scatter(x_ant, scene["antenna_y"], target["x"], target["centre_y"], target["radius_m"])
    return _refracted_time(x_ant, scene["antenna_y"], hit_x, hit_y, bands)


def _scatter(x_ant, y_ant, x_centre, y_centre, radius):
    dx = x_centre - x_ant
    dy = y_centre - y_ant
    distance = math.hypot(dx, dy)
    if distance <= radius:
        raise ValueError("antenna is inside a target")
    scale = radius / distance
    return x_centre - dx * scale, y_centre - dy * scale


def _refracted_time(x0, y0, x1, y1, bands):
    """Travel time of the Snell ray between two points through horizontal bands."""

    slabs = _slabs(y0, y1, bands)
    if not slabs:
        return math.hypot(x1 - x0, y1 - y0) / C
    horizontal = abs(x1 - x0)
    if horizontal == 0.0:
        return sum(thickness / velocity for thickness, velocity in slabs)
    fastest = max(velocity for _thickness, velocity in slabs)
    p_hi = 0.999 / fastest

    def reach(ray):
        total = 0.0
        for thickness, velocity in slabs:
            sine = ray * velocity
            if sine >= 1.0:
                return math.inf
            total += thickness * sine / math.sqrt(1.0 - sine * sine)
        return total

    if reach(p_hi) < horizontal:
        return _straight_time(x0, y0, x1, y1, bands)
    lo = 0.0
    hi = p_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if reach(mid) < horizontal:
            lo = mid
        else:
            hi = mid
    ray = 0.5 * (lo + hi)
    total = 0.0
    for thickness, velocity in slabs:
        sine = ray * velocity
        total += thickness / (velocity * math.sqrt(1.0 - sine * sine))
    return total


def _straight_time(x0, y0, x1, y1, bands):
    length = math.hypot(x1 - x0, y1 - y0)
    if length == 0.0:
        return 0.0
    edges = {y0, y1}
    low, high = (y0, y1) if y0 < y1 else (y1, y0)
    for y_start, y_end, _velocity in bands:
        if low < y_start < high:
            edges.add(y_start)
        if low < y_end < high:
            edges.add(y_end)
    ordered = sorted(edges)
    total = 0.0
    for start, end in zip(ordered, ordered[1:]):
        y_mid = 0.5 * (start + end)
        fraction = abs(end - start) / abs(y1 - y0)
        total += fraction * length / _velocity_at(y_mid, bands)
    return total


def _slabs(y_from, y_to, bands):
    low, high = (y_from, y_to) if y_from < y_to else (y_to, y_from)
    slabs = []
    for y0, y1, velocity in bands:
        bottom = max(y0, low)
        top = min(y1, high)
        if top - bottom > 1e-12:
            slabs.append((top - bottom, velocity))
    return slabs


def _velocity_at(y_value, bands):
    for y0, y1, velocity in bands:
        if y0 - 1e-9 <= y_value <= y1 + 1e-9:
            return velocity
    raise ValueError(f"no layer contains y={y_value}")


def _cylinder(x_value, y_value, radius, material):
    return (
        "#cylinder: "
        f"{_m(x_value)} {_m(y_value)} 0 {_m(x_value)} {_m(y_value)} inf {_m(radius)} {material}"
    )


def _rock_materials(rocks):
    return [
        {"name": rock["name"], "eps_r": rock["eps_r"], "sigma_s_m": rock["sigma_s_m"]}
        for rock in rocks
    ]


def _pvc_materials(targets):
    if any(target["type"] == "pvc" for target in targets):
        eps_r = next(target["eps_r"] for target in targets if target["type"] == "pvc")
        return [{"name": "pvc", "eps_r": eps_r, "sigma_s_m": 0.0}]
    return []


def _m(value):
    """Metre or material value that round-trips to 1 micrometre."""

    return f"{float(value):.6f}"


def _whole_cells(length, dx, name):
    """Raise if ``length`` is not a positive whole number of ``dx``."""

    cells = float(length) / float(dx)
    if cells < 1.0 - 1e-6 or abs(cells - round(cells)) > 1e-6:
        raise ValueError(f"{name} {length} m is {cells:.6f} cells of {dx} m, not a whole number")


def _trace_count(path):
    import h5py

    with h5py.File(path, "r") as handle:
        return int(handle.attrs["ntraces"])


def _merge_traces(base, n_traces, merged):
    """Merge ``scene1.h5`` through ``sceneN.h5`` by name, then check the count."""

    files = [Path(f"{base}{index}.h5") for index in range(1, int(n_traces) + 1)]
    missing = [path.name for path in files if not path.is_file()]
    if missing:
        present = sorted(path.name for path in Path(base).parent.glob(Path(base).name + "*.h5"))
        raise FileNotFoundError(
            f"expected {n_traces} traces, missing {len(missing)} ({', '.join(missing[:8])}). "
            f"found {', '.join(present[:12])}"
        )
    from gprMax.toolboxes.Utilities.outputfiles_merge import merge_files

    merge_files([str(path) for path in files], merged_outputfile=merged, removefiles=True)
    count = _trace_count(merged)
    if count != int(n_traces):
        raise RuntimeError(f"{merged} has {count} traces, expected {n_traces}")


def _geometry_from_log(path):
    if not Path(path).is_file():
        return {}
    text = Path(path).read_text(errors="replace")
    found = {}
    cells = re.search(r"\((\d+) x (\d+) x (\d+) = (\d+) cells\)", text)
    if cells:
        found["nx"] = int(cells.group(1))
        found["ny"] = int(cells.group(2))
        found["nz"] = int(cells.group(3))
        found["cells"] = int(cells.group(4))
    window = re.search(r"Time window: .* \((\d+) iterations\)", text)
    if window:
        found["iterations"] = int(window.group(1))
    return found


def _print_apex(rows):
    for row in rows:
        state = "visible" if row["visible"] else f"not visible ({row['visibility_reason']})"
        edge = " edge" if row["on_window_edge"] else ""
        print(
            f"  {row['name']} {row['type']}: predicted {row['predicted_ns']:.2f} ns, "
            f"picked {row['picked_ns']:.2f} ns, error {row['error_ns']:.2f} ns, "
            f"quarter period {row['quarter_period_ns']:.2f} ns, {state}{edge}",
            flush=True,
        )


def _print_bounds(config):
    print("Fixed grid, checked against lambda_min/10 and the deepest target bottom.", flush=True)
    print("f_max factor is 2.5: little Ricker energy remains above that multiple.", flush=True)
    for name in config["presets"]:
        layout = fixed_layout(name, config)
        bound = preset_bound(name, config)
        print(format_grid(f"{name} bound", bound), flush=True)
        print(
            f"  fixed dx {layout['dx']:.4f} m, window {layout['time_window_s'] * 1e9:.2f} ns, "
            f"domain {layout['domain_x']:.3f} x {layout['domain_y']:.3f} m, "
            f"profile {layout['profile_length_m']:.2f} m, traces {layout['n_traces']}",
            flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write a field-scale 2D gprMax scene.")
    parser.add_argument("--preset", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "sim" / "scenes")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args(argv)
    if args.n < 1:
        raise SystemExit("--n must be at least 1")
    if args.figure is not None and (args.n != 1 or not args.run):
        raise SystemExit("--figure is for one scene and needs --run")

    config = load_config()
    _print_bounds(config)
    for offset in range(args.n):
        seed = args.seed + offset
        scene = sample_scene(args.preset, seed, config)
        out_dir = args.out / args.preset / f"seed{seed}"
        model = save_scene(scene, out_dir / "scene.in")
        print(format_grid(f"{args.preset} seed {seed}", scene), flush=True)
        print(
            f"  domain {scene['nx']} x {scene['ny']} x 1, traces {scene['n_traces']}, "
            f"targets {len(scene['targets'])}, rocks {len(scene['rocks'])}",
            flush=True,
        )
        if not args.run:
            print(f"wrote {model}", flush=True)
            continue
        merged, wall, geometry = run_bscan(model, scene["n_traces"])
        reference, reference_wall = run_reference(scene, out_dir)
        total = (wall or 0.0) + (reference_wall or 0.0)
        record = {
            "wall_s": None if wall is None and reference_wall is None else total,
            "scene_wall_s": wall,
            "reference_wall_s": reference_wall,
            "traces": scene["n_traces"],
            **geometry,
        }
        if record["wall_s"] is not None:
            print(
                f"wall {record['wall_s']:.1f} s "
                f"(scene {(wall or 0):.1f} s, reference {(reference_wall or 0):.1f} s), "
                f"cells {geometry.get('cells')}, iterations {geometry.get('iterations')}, "
                f"traces {scene['n_traces']}",
                flush=True,
            )
        truth = ground_truth(scene)
        if args.figure is not None and seed == args.seed:
            import importlib

            sys.path.insert(0, str(ROOT / "src"))
            loader = importlib.import_module("02a_gprmax_loader")
            scan = loader.load_gprmax(merged)
            reference_scan = loader.load_gprmax(reference)
            _scan, _picks, dewow_ns, agc_ns = plot_scene(merged, scene, truth, args.figure)
            rows = apex_check(scan, scene, truth, reference_scan)
            record["dewow_ns"] = dewow_ns
            record["agc_ns"] = agc_ns
            record["apex"] = rows
            print(f"wrote {args.figure}", flush=True)
            print(f"dewow {dewow_ns:.2f} ns, AGC {agc_ns:.2f} ns", flush=True)
            _print_apex(rows)
        elif args.run:
            import importlib

            sys.path.insert(0, str(ROOT / "src"))
            loader = importlib.import_module("02a_gprmax_loader")
            rows = apex_check(
                loader.load_gprmax(merged),
                scene,
                truth,
                loader.load_gprmax(reference),
            )
            record["apex"] = rows
            _print_apex(rows)
        (out_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
