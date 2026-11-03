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
| `configs/scenes.yaml` | Field-scale presets: 80 MHz and 400 MHz. |
| `src/01c_scene_generator.py` | Draws a 2D scene, writes the `.in` file, the reference trace, and the travel times. |
| `test/01c_test_scene_generator.py` | Checks the draw, the fixed grid, the reference model, and the visibility rule. |
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
| `data/sim/` | Cylinder A-scan, the pipe-study B-scans, and field-scale scenes. |
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

- `figures/pipe_bscan_variants.png`: Ez with the mean trace removed (the script only)
- `data/sim/pipe_study/<tag>/pipe_merged.h5`: one merged B-scan per case
- `data/sim/pipe_study/<tag>/ground_truth.json`: target centre, depth, radius, and permittivity
- `data/sim/pipe_study/cases.csv`: the eight cases

## Field-scale scenes

`src/01c_scene_generator.py` reads `configs/scenes.yaml` and writes one 2D model, in the same form as `pipe.in`, plus a JSON file of the parameters and the predicted hyperbola. It also writes `reference.in`: the same soil and layers, with no targets and no rocks. That model is one trace, saved as `reference.h5`. Subtracting it removes the direct air wave and the horizontal reflections. The soil is a half-space plus 0, 1, or 2 layers, conductivity 0 to 0.01 S/m. Each scene has 1 to 3 targets, a metal pipe, an air-filled PVC pipe, or an air void, and 0 to 5 rocks. A cylinder is drawn large enough to cover at least one cell. Targets sit inside the fixed profile, with 0.05 m of clearance from each other and from the ends.

Every scene in a preset uses the same cell size, time window, domain, and trace count. The cell is checked against one tenth of the shortest wavelength. That wavelength is the slowest soil velocity divided by 2.5 times the centre frequency: the gprMax Ricker has little energy above that multiple. The check allows 0.1 mm, so a round cell size can land the step and the offset on whole cells. At 80 MHz, lambda_min/10 is 0.037474 m and the cell is 0.0375 m. The time window has to cover the bistatic two-way time to the deepest target bottom the preset allows, at that slowest velocity, plus 25%, plus the Ricker delay. The pulse peaks at sqrt(2)/f. The z axis is one cell, as in `pipe.in`, so the PML on that face is 0. The in-plane PML is 10 cells, and the antennas sit two cells above the soil.

The profile length is fixed from that same bound. The flank half-width is the distance at which the hyperbola leaves the window. Full flanks, plus one hyperbola-width of flat ground on each side, is six half-widths. `field_80mhz` needs 15.22 m, so the profile is 15.30 m: 102 steps of 0.15 m, 103 traces. `field_400mhz` needs 10.33 m. The profile is capped at 5.00 m: 100 steps of 0.05 m, 101 traces.

`field_80mhz` is an 80 MHz Ricker. The offset is 0.4875 m, 13 cells, the nearest node to the 0.48 m Dinner Point Avenue pair. The trace step is 0.15 m, 4 cells. Relative permittivity is 4 to 16. Target tops are 0.3 m to 2.0 m. `field_400mhz` is a 400 MHz Ricker. The offset is 0.1625 m, 26 cells, the nearest node to the 0.16 m separation in a shielded housing such as the GSSI 400 MHz antenna. The trace step is 0.05 m, 8 cells. Relative permittivity is 4 to 12. Target tops are 0.2 m to 1.0 m. Those ranges keep one scene to a few minutes.

| Preset | dx | Time window | Domain (x, y) | Profile | Traces | PML |
| --- | --- | --- | --- | --- | --- | --- |
| field_80mhz | 0.0375 m | 102.10 ns | 16.6125 m, 3.975 m | 15.30 m | 103 | 10 cells, 0.375 m |
| field_400mhz | 0.00625 m | 47.10 ns | 5.300 m, 2.2375 m | 5.00 m | 101 | 10 cells, 0.0625 m |

The bound the window is checked against is 102.09 ns at 80 MHz (permittivity 16, target bottom 2.5 m) and 47.07 ns at 400 MHz (permittivity 12, target bottom 1.5 m).

A target is marked visible when its predicted apex is at least one centre period after the direct air wave and still inside the window. The direct wave arrives at the Ricker delay plus the offset divided by the speed of light. A target that fails that test stays in the model as clutter and gets no label. In 1000 draws, the visible share at 80 MHz was 77.3% of air voids (527/682), 81.5% of metal pipes (546/670), and 78.9% of PVC pipes (526/667). At 400 MHz every target was visible (682 air, 670 metal, 667 PVC).

Seed 0, including the reference trace, on this 16-thread CPU:

| Preset | Wall time | Cells | Iterations | Traces |
| --- | --- | --- | --- | --- |
| field_80mhz | 81.6 s | 46958 | 1156 | 103 |
| field_400mhz | 381.9 s | 303584 | 3197 | 101 |

The 80 MHz run is 80.0 s for the B-scan and 1.6 s for the reference. The 400 MHz run is 377.3 s and 4.7 s. Every scene in a preset has this grid, so 300 scenes is about 6.8 hours at 80 MHz and about 31.8 hours at 400 MHz. That batch was not run. A merged file newer than its `.in`, with the full trace count, is skipped.

```bash
conda activate gprMax-v4
python src/01c_scene_generator.py --preset field_80mhz --seed 0 --n 1 --out data/sim/scenes --run --figure figures/scene_field_80mhz_seed0.png
python src/01c_scene_generator.py --preset field_400mhz --seed 0 --n 1 --out data/sim/scenes --run --figure figures/scene_field_400mhz_seed0.png
```

The figure is the raw B-scan beside the processed one, with the predicted hyperbolas drawn on both. Processing is time zero, dewow, background removal, a band-pass around the centre frequency, then AGC. Dewow and AGC use three periods of the centre frequency, the same multiple as the 2 ns window on the 1.5 GHz pipe: 37.5 ns at 80 MHz and 7.5 ns at 400 MHz. The apex check uses the reference-subtracted traces and takes the envelope maximum within half a period of the prediction.

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

1. **Time zero**: shift each trace so the first break is at the start of the record.
2. **Dewow**: subtract a running mean and remove the low-frequency bias from antenna coupling.
3. **Background removal**: subtract the mean trace and remove the horizontal direct wave.
4. **Band-pass**: keep about half to twice the antenna frequency, without shifting the pulse.
5. **Gain**: divide by a running RMS so a weaker late reflection remains visible.

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
