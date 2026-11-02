# GPRadar

Simulated and processed ground-penetrating radar (GPR) profiles. gprMax builds the synthetic B-scans. GPRPy is available for field profiles. Short notes on travel time and hyperbolas are in [gpr_basics.md](gpr_basics.md). Signal processing is in [gpr_signal_processing.md](gpr_signal_processing.md). Detection with YOLO is in [gpr_yolo.md](gpr_yolo.md).

Programs and notebooks are numbered in the order to run them. The same number is one stage. A letter means those files do not wait on each other. Where a script and a notebook do the same job, the notebook plots in the page and the script writes a figure file.

## Layout

| Path | Contents |
| --- | --- |
| `src/pipe.in` | Buried metal pipe. 2D model, 2 mm cells, 1.5 GHz Ricker, 60-trace B-scan. |
| `notebooks/01a_download_usgs_raw.ipynb` | Downloads Dinner Point Avenue and removes the other lines. |
| `src/01a_pipe_bscan_study.py` | Eight variants of `pipe.in`, then one comparison figure. |
| `notebooks/01b_pipe_bscan_study.ipynb` | The same eight runs, plotted in the notebook. |
| `test/01b_test_ground_truth.py` | Checks the pipe centre and the soil permittivity. |
| `src/02a_gprmax_loader.py` | Reads a gprMax HDF5 output into one scan. |
| `notebooks/02a_gprmax_loader.ipynb` | The same read, then the Dinner Point Avenue comparison. |
| `test/02a_test_gprmax_loader.py` | Checks the loader against the cylinder and a pipe B-scan. |
| `src/02b_preprocessing.py` | Time zero, dewow, background removal, band-pass and gain. |
| `test/02b_test_preprocessing.py` | Checks those processing steps. |
| `src/02c_gprpy_reference.py` | Runs GPRPy dewow and AGC and saves the reference arrays. |
| `src/03a_plot_preprocessing.py` | Figures of the processing steps and of the amplitude spectrum. |
| `notebooks/03a_scipy_gpr_simulated.ipynb` | Loads one pipe B-scan and shows each processing step. |
| `src/03b_compare_gprpy.py` | Compares dewow and AGC with that GPRPy reference. |
| `test/03b_test_gprpy_comparison.py` | Checks the interior correlation with GPRPy. |
| `environment-gprmax.yml` | Conda environment for gprMax output, processing, and tests. |
| `environment-gprpy.yml` | Conda environment for the GPRPy reference script. |
| `data/sim/` | Cylinder A-scan and the pipe-study B-scans. |
| `figures/` | Plots, plus `gprpy_comparison.csv` and `gprpy_comparison.png`. |

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
python src/01a_pipe_bscan_study.py
```

`notebooks/01b_pipe_bscan_study.ipynb` does the same runs and plots the eight B-scans in the notebook instead of writing the PNG. Both write `cases.csv`. From the repository root:

```bash
conda activate gprMax-v4
jupyter notebook notebooks/01b_pipe_bscan_study.ipynb
```

Outputs:

- `figures/pipe_bscan_variants.png` — Ez with the mean trace removed (the script only)
- `data/sim/pipe_study/<tag>/pipe_merged.h5` — one merged B-scan per case
- `data/sim/pipe_study/<tag>/ground_truth.json` — target centre, depth, radius, and permittivity
- `data/sim/pipe_study/cases.csv` — the eight cases

## Reading a gprMax file

`load_gprmax` accepts a `.h5` or `.out` file, a directory of per-trace files, or a list of those paths. The result is one object: a time-by-trace array, `dt` in seconds, trace spacing in metres, centre frequency in hertz, and the source description.

```bash
conda activate gprMax-v4
python -c "
import importlib
import sys
sys.path.insert(0, 'src')
loader = importlib.import_module('02a_gprmax_loader')
scan = loader.load_gprmax('data/sim/cylinder_Ascan_2D.h5')
print(scan.data.shape, scan.dt, scan.frequency, scan.source.kind)
"
```

A single A-scan has no trace spacing. A merged B-scan does not keep the waveform, so `frequency` is `None` unless you load the per-trace files from before they were merged.

`notebooks/02a_gprmax_loader.ipynb` reads the same files, then compares a running-mean dewow and a running-RMS AGC with the GPRPy windows on Dinner Point Avenue. The six panels are plotted in the notebook.

```bash
conda activate gprMax-v4
jupyter notebook notebooks/02a_gprmax_loader.ipynb
python -m pytest test/02a_test_gprmax_loader.py
```

## Preprocessing

The steps in `src/02b_preprocessing.py` run in this order:

1. **Time zero** — shift each trace so the first break is at the start of the record.
2. **Dewow** — subtract a running mean and remove the low-frequency bias from antenna coupling.
3. **Background removal** — subtract the mean trace and remove the horizontal direct wave.
4. **Band-pass** — keep about half to twice the antenna frequency, without shifting the pulse.
5. **Gain** — divide by a running RMS so a weaker late reflection remains visible.

```bash
conda activate gprMax-v4
python -m pytest test/02b_test_preprocessing.py
MPLBACKEND=Agg python src/03a_plot_preprocessing.py
```

`notebooks/03a_scipy_gpr_simulated.ipynb` loads the 1.5 GHz pipe scan and shows each step with its image. From the repository root:

```bash
conda activate gprMax-v4
jupyter notebook notebooks/03a_scipy_gpr_simulated.ipynb
```

On this 12 cm profile the hyperbola covers most traces, so mean-trace removal leaves artefacts near the apex. Longer profiles, or subtracting a no-target simulation, avoid this.

## Field profile

Dinner Point Avenue is the only line this repository keeps from the USGS Forsythe release. `notebooks/01a_download_usgs_raw.ipynb` checks the two published files, `DinnerPointAve.rd3` and `DinnerPointAve.rad`, and deletes the other profiles if they are in `data/raw/usgs_forsythe/`.

```bash
conda activate gprMax-v4
jupyter notebook notebooks/01a_download_usgs_raw.ipynb
```

## Cross-check against GPRPy

Dewow and AGC were compared with GPRPy 1.0.14 on that profile (MALA, 4188 traces). GPRPy's result is `data/raw/usgs_forsythe/DinnerPointAve_gprpy_steps.npz`. That file is about 100 MB and is gitignored. Recreate it, then run the comparison:

```bash
conda activate gprpy
python src/02c_gprpy_reference.py
```

```bash
conda activate gprMax-v4
MPLBACKEND=Agg python src/03b_compare_gprpy.py
```

`src/03b_compare_gprpy.py` does not import GPRPy. It reads the `.npz`, runs `dewow` (25 ns) and then `gain_agc` (50 ns), and writes `figures/gprpy_comparison.csv` and `figures/gprpy_comparison.png`. "Interior" drops one window from each end of the trace: 32 samples for dewow, 64 for AGC.

| Step | Region | Max abs. difference | Relative RMS | Correlation |
| --- | --- | --- | --- | --- |
| dewow | full | 1080.6326 | 0.030876 | 0.999541 |
| dewow | interior | 1080.6326 | 0.025966 | 0.999686 |
| AGC | full | 3.7854 | 6.782221 | 0.976026 |
| AGC | interior | 3.6120 | 7.051782 | 0.999535 |

GPRPy's moving window is one sample longer (33 and 65 instead of 32 and 64) and is centred differently, and its edge samples share one frozen window. Dewow still matches: interior correlation 0.999686, relative RMS 0.026. AGC does not match in amplitude. GPRPy divides by the L2 norm of the window; `gain_agc` divides by the RMS, about `sqrt(64)` smaller. The relative RMS of about 7 is that factor. Interior correlation stays 0.999535 because correlation ignores a constant scale. On the full trace the AGC correlation falls to 0.976026 because of the frozen edge gain. The two AGC panels in the figure share a colour scale, so the GPRPy panel looks pale and the difference looks like the `02b_preprocessing.py` panel.
