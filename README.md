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
| `test/test_preprocessing.py` | Tests for the processing steps. |
| `data/sim/` | Cylinder A-scan and the pipe-study B-scans. |
| `figures/` | Plots of those runs, plus one GPRPy example profile. |
| `test/` | Loader tests. |

## Environments

gprMax and GPRPy are in separate conda environments under `~/miniforge3`. Open a new terminal so `conda` is on your path.

| Environment | What it runs | Source checkout |
| --- | --- | --- |
| `gprMax-v4` | gprMax 4.0.1, built with the system GCC and OpenMP | `~/Code/2026/gprMax-v4-src` |
| `gprpy` | GPRPy 1.0.14 | `~/Code/2026/GPRPy` |

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

The getting-started metal cylinder is already in `data/sim/cylinder_Ascan_2D.h5`. The Ez trace is `figures/cylinder_Ascan_2D_Ez.png`. To run it again:

```bash
conda activate gprMax-v4
python -m gprMax ~/Code/2026/gprMax-v4-src/examples/gpr/basic/cylinder_Ascan_2D.in \
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

On this 12 cm profile the hyperbola covers most traces, so mean-trace removal leaves artefacts near the apex. Longer profiles, or subtracting a no-target simulation, avoid this.
