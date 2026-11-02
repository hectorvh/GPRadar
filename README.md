# GPRadar

Simulated and processed ground-penetrating radar (GPR) profiles. gprMax builds the synthetic B-scans. GPRPy is available for field profiles. Short notes on travel time and hyperbolas are in [gpr_basics.md](gpr_basics.md).

## Layout

| Path | Contents |
| --- | --- |
| `src/pipe.in` | Buried metal pipe. 2D model, 2 mm cells, 1.5 GHz Ricker, 60-trace B-scan. |
| `src/pipe_bscan_study.py` | Eight variants of that model, then one comparison figure. |
| `src/gprmax_loader.py` | Reads a gprMax HDF5 output into one scan. |
| `src/preprocessing.py` | Time zero, dewow, background removal, band-pass and gain. |
| `src/plot_preprocessing.py` | Figures of those steps and of the amplitude spectrum. |
| `src/gprpy_reference.py` | Runs GPRPy dewow and AGC and saves the reference arrays. |
| `src/compare_gprpy.py` | Compares dewow and AGC with that GPRPy reference. |
| `notebooks/gpr_preprocessing.ipynb` | Loads one pipe B-scan and shows each processing step. |
| `environment-gprmax.yml` | Conda environment for gprMax output, processing, and tests. |
| `environment-gprpy.yml` | Conda environment for the GPRPy reference script. |
| `data/sim/` | Cylinder A-scan and the pipe-study B-scans. |
| `figures/` | Plots, plus `gprpy_comparison.csv` and `gprpy_comparison.png`. |
| `test/` | Tests for the loader, the processing steps, and the GPRPy comparison. |

## Environments

gprMax and GPRPy are in separate conda environments. Create them from the repository root:

```bash
conda env create -f environment-gprmax.yml
conda env create -f environment-gprpy.yml
```

`environment-gprmax.yml` installs the Python packages used to read gprMax output and run the processing. gprMax 4.0.1 itself is built from source (https://github.com/gprMax/gprMax) because it needs a C compiler with OpenMP (GCC). Install that release into the `gprMax-v4` environment after creating it.

Run gprMax from this repository. The gprMax checkout contains an uncompiled package, and Python will load that instead of the installed one if it is the current directory.

```bash
conda activate gprMax-v4
python -m gprMax path/to/model.in
```

GPRPy opens a window and asks for a profile (`p`) or a CMP/WARR (`c`):

```bash
conda activate gprpy
gprpy
```

## Cylinder A-scan

The getting-started metal cylinder is already in `data/sim/cylinder_Ascan_2D.h5`. The Ez trace is `figures/cylinder_Ascan_2D_Ez.png`. To run it again, pass `examples/gpr/basic/cylinder_Ascan_2D.in` from the gprMax source tree:

```bash
conda activate gprMax-v4
python -m gprMax /path/to/gprMax/examples/gpr/basic/cylinder_Ascan_2D.in \
  -o data/sim/cylinder_Ascan_2D.h5
MPLBACKEND=Agg python -m gprMax.toolboxes.Plotting.plot_Ascan \
  data/sim/cylinder_Ascan_2D.h5 -save --outputs Ez
```

## Pipe B-scans

`src/pipe.in` is the same kind of model as a B-scan: a metal cylinder in a dielectric half-space, with the antenna pair stepped across it. One 60-trace B-scan of `src/pipe.in` takes about 32 s on a 16-thread laptop CPU (gprMax 4.0.1, OpenMP). The study script keeps that geometry and changes the source and the ground. It runs 8 of the 36 combinations of 3 frequencies (0.9, 1.5, 2.1 GHz), 3 permittivities (4, 6, 12), 2 conductivities (0 and 0.05 S/m), and 2 pipe-centre depths (5 cm and 9 cm). Each run is 60 traces. Finished scans are skipped.

```bash
conda activate gprMax-v4
python src/pipe_bscan_study.py
```

Outputs:

- `figures/pipe_bscan_variants.png` — Ez with the mean trace removed
- `data/sim/pipe_study/<tag>/pipe_merged.h5` — one merged B-scan per case
- `data/sim/pipe_study/cases.csv` — the eight cases

## Reading a gprMax file

`load_gprmax` accepts a `.h5` or `.out` file, a directory of per-trace files, or a list of those paths. The result is one object: a time-by-trace array, `dt` in seconds, trace spacing in metres, centre frequency in hertz, and the source description.

```bash
conda activate gprMax-v4
python -c "
import sys
sys.path.insert(0, 'src')
from gprmax_loader import load_gprmax
scan = load_gprmax('data/sim/cylinder_Ascan_2D.h5')
print(scan.data.shape, scan.dt, scan.frequency, scan.source.kind)
"
```

A single A-scan has no trace spacing. A merged B-scan does not keep the waveform, so `frequency` is `None` unless you load the per-trace files from before they were merged.

```bash
conda activate gprMax-v4
python -m pytest test/test_gprmax_loader.py
```

## Preprocessing

The steps in `src/preprocessing.py` run in this order:

1. **Time zero** — shift each trace so the first break is at the start of the record.
2. **Dewow** — subtract a running mean and remove the low-frequency bias from antenna coupling.
3. **Background removal** — subtract the mean trace and remove the horizontal direct wave.
4. **Band-pass** — keep about half to twice the antenna frequency, without shifting the pulse.
5. **Gain** — divide by a running RMS so a weaker late reflection remains visible.

```bash
conda activate gprMax-v4
python -m pytest test/test_preprocessing.py
MPLBACKEND=Agg python src/plot_preprocessing.py
```

`notebooks/gpr_preprocessing.ipynb` loads the 1.5 GHz pipe scan and shows each step with its image. From the repository root:

```bash
conda activate gprMax-v4
jupyter notebook notebooks/gpr_preprocessing.ipynb
```

On this 12 cm profile the hyperbola covers most traces, so mean-trace removal leaves artefacts near the apex. Longer profiles, or subtracting a no-target simulation, avoid this.

## Cross-check against GPRPy

Dewow and AGC were compared with GPRPy 1.0.14 on the USGS Dinner Point Avenue profile (MALA, 4188 traces). GPRPy's result is `data/raw/usgs_forsythe/DinnerPointAve_gprpy_steps.npz`. That file is about 100 MB and is gitignored. Recreate it, then run the comparison:

```bash
conda activate gprpy
python src/gprpy_reference.py
```

```bash
conda activate gprMax-v4
MPLBACKEND=Agg python src/compare_gprpy.py
```

`src/compare_gprpy.py` does not import GPRPy. It reads the `.npz`, runs `dewow` (25 ns) and then `gain_agc` (50 ns), and writes `figures/gprpy_comparison.csv` and `figures/gprpy_comparison.png`. "Interior" drops one window from each end of the trace: 32 samples for dewow, 64 for AGC.

| Step | Region | Max abs. difference | Relative RMS | Correlation |
| --- | --- | --- | --- | --- |
| dewow | full | 1080.6326 | 0.030876 | 0.999541 |
| dewow | interior | 1080.6326 | 0.025966 | 0.999686 |
| AGC | full | 3.7854 | 6.782221 | 0.976026 |
| AGC | interior | 3.6120 | 7.051782 | 0.999535 |

GPRPy's moving window is one sample longer (33 and 65 instead of 32 and 64) and is centred differently, and its edge samples share one frozen window. Dewow still matches: interior correlation 0.999686, relative RMS 0.026. AGC does not match in amplitude. GPRPy divides by the L2 norm of the window; `gain_agc` divides by the RMS, about `sqrt(64)` smaller. The relative RMS of about 7 is that factor. Interior correlation stays 0.999535 because correlation ignores a constant scale. On the full trace the AGC correlation falls to 0.976026 because of the frozen edge gain. The two AGC panels in the figure share a colour scale, so the GPRPy panel looks pale and the difference looks like the `preprocessing.py` panel.
