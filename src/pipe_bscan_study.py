"""Generate and run a small set of B-scan variants of src/pipe.in.

The full factorial of 3 frequencies, 3 permittivities, 2 conductivities and
2 depths is 36 models. This script runs 8 of them: a frequency sweep, the
other two permittivities, one lossy soil, the deeper pipe, and one combined
slow/lossy case. Each level of every factor appears at least once.
"""

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gprMax.toolboxes.Utilities.outputfiles_merge import get_output_data

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "pipe.in"
RUN_ROOT = ROOT / "data" / "sim" / "pipe_study"
FIGURE = ROOT / "figures" / "pipe_bscan_variants.png"
N_TRACES = 60
DX = 0.002

# Eight-model subset. Depth is the pipe-centre depth below the soil surface.
CASES = [
    {"tag": "f0.9_e6_s0_d5", "freq": 0.9e9, "eps": 6, "sigma": 0.0, "depth": 0.05},
    {"tag": "f1.5_e6_s0_d5", "freq": 1.5e9, "eps": 6, "sigma": 0.0, "depth": 0.05},
    {"tag": "f2.1_e6_s0_d5", "freq": 2.1e9, "eps": 6, "sigma": 0.0, "depth": 0.05},
    {"tag": "f1.5_e4_s0_d5", "freq": 1.5e9, "eps": 4, "sigma": 0.0, "depth": 0.05},
    {"tag": "f1.5_e12_s0_d5", "freq": 1.5e9, "eps": 12, "sigma": 0.0, "depth": 0.05},
    {"tag": "f1.5_e6_s05_d5", "freq": 1.5e9, "eps": 6, "sigma": 0.05, "depth": 0.05},
    {"tag": "f1.5_e6_s0_d9", "freq": 1.5e9, "eps": 6, "sigma": 0.0, "depth": 0.09},
    {"tag": "f1.5_e12_s05_d9", "freq": 1.5e9, "eps": 12, "sigma": 0.05, "depth": 0.09},
]


def _tokens(line):
    return line.split()


def parse_template(text):
    """Read the geometry the variants keep fixed."""
    surface_y = None
    pipe_y = None
    src_x = None
    rx_x = None
    step = None
    for line in text.splitlines():
        if line.startswith("#box:"):
            surface_y = float(_tokens(line)[5])
        elif line.startswith("#cylinder:"):
            parts = _tokens(line)
            pipe_y = float(parts[2])
            if abs(float(parts[5]) - pipe_y) > 1e-9:
                raise ValueError("pipe.in cylinder is not horizontal in y")
        elif line.startswith("#hertzian_dipole:"):
            src_x = float(_tokens(line)[2])
        elif line.startswith("#rx:"):
            rx_x = float(_tokens(line)[1])
        elif line.startswith("#src_steps:"):
            step = float(_tokens(line)[1])
    missing = [
        name
        for name, value in (
            ("#box", surface_y),
            ("#cylinder", pipe_y),
            ("#hertzian_dipole", src_x),
            ("#rx", rx_x),
            ("#src_steps", step),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"pipe.in is missing {', '.join(missing)}")
    return {
        "surface_y": surface_y,
        "pipe_y": pipe_y,
        "src_x": src_x,
        "rx_x": rx_x,
        "step": step,
    }


def render(template, case, surface_y):
    pipe_y = round((surface_y - case["depth"]) / DX) * DX
    title = (
        f"Pipe B-scan {case['freq'] / 1e9:.1f} GHz, "
        f"eps_r={case['eps']:g}, sigma={case['sigma']:g}, "
        f"depth={case['depth']:.2f} m"
    )
    lines = []
    for line in template.splitlines():
        if line.startswith("#title:"):
            lines.append(f"#title: {title}")
        elif line.startswith("#material:"):
            lines.append(f"#material: {case['eps']:g} {case['sigma']:g} 1 0 half_space")
        elif line.startswith("#waveform:"):
            lines.append(f"#waveform: ricker 1 {case['freq']:.6g} my_ricker")
        elif line.startswith("#cylinder:"):
            parts = _tokens(line)
            lines.append(
                f"#cylinder: {parts[1]} {pipe_y:.3f} {parts[3]} "
                f"{parts[4]} {pipe_y:.3f} {parts[6]} {parts[7]} {parts[8]}"
            )
        else:
            lines.append(line)
    return "\n".join(lines) + "\n", pipe_y


def ground_truth_from_model(text):
    """Target centre and soil permittivity from a gprMax input file.

    Depth is the cylinder centre below the top of the half-space.
    """

    eps_r = None
    surface_y = None
    target_x = None
    target_y = None
    radius = None
    for line in text.splitlines():
        parts = _tokens(line)
        if line.startswith("#material:"):
            eps_r = float(parts[1])
        elif line.startswith("#box:"):
            surface_y = float(parts[5])
        elif line.startswith("#cylinder:"):
            target_x = float(parts[1])
            target_y = float(parts[2])
            if abs(float(parts[5]) - target_y) > 1e-9:
                raise ValueError("cylinder is not horizontal")
            radius = float(parts[7])
    missing = [
        name
        for name, value in (
            ("eps_r", eps_r),
            ("surface", surface_y),
            ("target_x", target_x),
            ("target_y", target_y),
            ("radius", radius),
        )
        if value is None
    ]
    if missing:
        raise ValueError("model is missing " + ", ".join(missing))
    return {
        "eps_r": eps_r,
        "target_x_m": round(target_x, 6),
        "target_depth_m": round(surface_y - target_y, 6),
        "target_y_m": round(target_y, 6),
        "target_radius_m": round(radius, 6),
    }


def write_ground_truth(model_dir, text):
    truth = ground_truth_from_model(text)
    (model_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return truth


def run_case(case, template, geometry):
    model_dir = RUN_ROOT / case["tag"]
    model_dir.mkdir(parents=True, exist_ok=True)
    text, pipe_y = render(template, case, geometry["surface_y"])
    infile = model_dir / "pipe.in"
    infile.write_text(text)
    truth = write_ground_truth(model_dir, text)
    merged = model_dir / "pipe_merged.h5"
    if merged.exists():
        print(f"skip {case['tag']}: {merged.name} already exists", flush=True)
        return merged, pipe_y, truth

    base = model_dir / "pipe"
    command = [
        sys.executable,
        "-m",
        "gprMax",
        str(infile),
        "-n",
        str(N_TRACES),
        "--geometry-fixed",
        "-o",
        str(base),
    ]
    print(f"run {case['tag']}", flush=True)
    subprocess.run(command, check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gprMax.toolboxes.Utilities.outputfiles_merge",
            str(base),
            "--remove-files",
        ],
        check=True,
    )
    if not merged.exists():
        raise FileNotFoundError(merged)
    return merged, pipe_y, truth


def load_bscan(path):
    data, dt = get_output_data(path, 1, "Ez")
    data = np.asarray(data, dtype=float)
    background = data.mean(axis=1, keepdims=True)
    return data - background, dt


def plot_cases(results, geometry):
    step = geometry["step"]
    mid0 = 0.5 * (geometry["src_x"] + geometry["rx_x"])
    position = mid0 + step * np.arange(N_TRACES)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True, constrained_layout=True)
    for ax, (case, path, _pipe_y, _truth) in zip(axes.ravel(), results):
        data, dt = load_bscan(path)
        if data.shape[1] != N_TRACES:
            raise ValueError(f"{path} has {data.shape[1]} traces, expected {N_TRACES}")
        time_ns = np.arange(data.shape[0]) * dt * 1e9
        limit = np.percentile(np.abs(data), 99)
        if limit == 0:
            limit = 1
        ax.imshow(
            data,
            extent=[position[0], position[-1], time_ns[-1], time_ns[0]],
            aspect="auto",
            cmap="seismic",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ax.set_title(
            f"{case['freq'] / 1e9:.1f} GHz,  εr={case['eps']:g}\n"
            f"σ={case['sigma']:g} S/m,  depth {case['depth'] * 100:.0f} cm",
            fontsize=10,
        )
        ax.axvline(0.120, color="k", lw=0.6, ls=":", alpha=0.7)

    for ax in axes[:, 0]:
        ax.set_ylabel("Time [ns]")
    for ax in axes[1, :]:
        ax.set_xlabel("Antenna midpoint [m]")
    fig.suptitle("Metal-pipe B-scans, Ez with the mean trace removed", fontsize=13)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=140)
    plt.close(fig)
    print(f"wrote {FIGURE}", flush=True)


def write_index(results):
    index = RUN_ROOT / "cases.csv"
    with index.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
            "tag",
            "freq_hz",
            "eps_r",
            "sigma_s_m",
            "depth_m",
            "pipe_y_m",
            "target_x_m",
            "target_radius_m",
            "merged",
        ],
        )
        writer.writeheader()
        for case, path, pipe_y, truth in results:
            writer.writerow(
                {
                    "tag": case["tag"],
                    "freq_hz": case["freq"],
                    "eps_r": truth["eps_r"],
                    "sigma_s_m": case["sigma"],
                    "depth_m": truth["target_depth_m"],
                    "pipe_y_m": f"{pipe_y:.3f}",
                    "target_x_m": f"{truth['target_x_m']:.3f}",
                    "target_radius_m": f"{truth['target_radius_m']:.3f}",
                    "merged": Path(path).resolve().relative_to(ROOT).as_posix(),
                }
            )


def main():
    template = TEMPLATE.read_text()
    if not re.search(r"^#waveform:\s+ricker\b", template, re.M):
        raise ValueError("pipe.in waveform must be a ricker source")
    geometry = parse_template(template)
    if abs(geometry["step"] - DX) > 1e-12:
        raise ValueError(f"expected src step {DX}, found {geometry['step']}")
    results = []
    for case in CASES:
        merged, pipe_y, truth = run_case(case, template, geometry)
        results.append((case, merged, pipe_y, truth))
    write_index(results)
    plot_cases(results, geometry)


if __name__ == "__main__":
    main()
