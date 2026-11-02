# GPR basics

GPR sends a short EM pulse into the ground and records what comes back. Everything below comes down to one idea: **time in the record is distance in the ground, scaled by the wave velocity.**

## 1. Wave velocity: v = c / √εr

- c ≈ 0.3 m/ns (speed of light in air).
- εr = relative permittivity of the ground. It controls how fast the wave travels.
- Assumes a low-loss, non-magnetic ground (μr ≈ 1). If conductivity is high (clay, salty water), the wave dies before it travels far and GPR stops working.
- Water dominates εr, so **wetter soil = slower wave = shallower depth for the same time.**

| Material | εr | v (m/ns) |
| --- | --- | --- |
| Air | 1 | 0.30 |
| Dry sand | ~4 | ~0.15 |
| Moist soil | ~9 | ~0.10 |
| Wet soil | ~25 | ~0.06 |
| Water | 80 | ~0.03 |

## 2. Two-way travel time

The radar measures the time to go down **and** come back, so:

- **t = 2d / v**  → **d = v·t / 2**
- Example: εr = 9 → v = 0.1 m/ns. A pipe at 1 m shows up at t = 20 ns.
- Without a velocity estimate, the vertical axis is only time, not depth. Getting v right is half the job.

## 3. Why a point target makes a hyperbola

The antenna does not only "see" straight down; its beam is a wide cone. So the pipe is visible before and after the antenna passes over it.

- Antenna at position x, target at (x₀, d). The distance to the target is √(d² + (x − x₀)²).
- Two-way time: **t(x) = (2/v) · √(d² + (x − x₀)²)**
- Directly above (x = x₀): shortest time, t₀ = 2d/v → the **apex**.
- Moving away: the path gets longer → the arms bend down → **hyperbola**.
- The **shape depends on v**: a fast (dry) ground gives a wide, flat hyperbola; a slow (wet) ground gives a narrow, steep one.

This is the useful part: fitting a hyperbola gives **x₀, d and v at the same time**. It is what `labels.py` (box from geometry) and `hyperbola_fit.py` (depth and velocity from detections) are built on.

Note: a buried pipe is a point in 2D (cross-section). Crossing the pipe at 90° gives the cleanest hyperbola; at an angle, it gets wider.

## 4. A-scan vs B-scan vs C-scan

| | What it is | Axes | Target looks like |
| --- | --- | --- | --- |
| **A-scan** | One trace at one position | amplitude vs time | a wiggle (peak) at t₀ |
| **B-scan** | Many A-scans along a line, side by side (radargram) | position vs time | a hyperbola |
| **C-scan** | Many parallel B-scans over a grid → 3D volume | x, y, time | a hyperboloid; an ellipse/circle in a time slice |

C-scans are usually shown as **time (depth) slices**: plan-view maps at a fixed time. Good for following a pipe across the site.

## 5. Three material properties

A GPR wave is an electromagnetic field. Three properties decide what the ground does with it.

**Electrical conductivity σ.** Some charges are free to move: ions in pore water, or electrons in metals and some minerals. The radar’s electric field makes them drift, and the collisions turn wave energy into heat. That energy is gone. Unit: S/m, usually quoted as mS/m. High σ: clay, saline water, salty or fertilized soil, metal. Low σ: dry sand, ice, granite, fresh dry concrete.

**Dielectric permittivity ε.** Other charges are bound. The field can only shift electrons or rotate water molecules, so the material polarizes, then springs back when the field reverses. The energy is stored and returned, like a capacitor. That storage is the lag that slows the wave, so ε sets velocity (v = c / √εr) and reflections (contrasts in εr).

**Magnetic permeability μ.** μ describes how magnetic moments align with the wave’s magnetic field. For almost all soils, rocks, water and concrete, μr ≈ 1, the same as air, which is why the velocity formula drops μ. Iron oxides (magnetite, some volcanic soils) are the exception: extra loss and a slower wave.

| Property | Charges | Energy | Main effect on GPR |
| --- | --- | --- | --- |
| σ | Free charges flow | Dissipated as heat | Attenuation, depth limit |
| ε | Bound charges displaced | Stored and returned | Velocity, reflections |
| μ | Magnetic moments align | Stored, and some lost | Usually negligible |

### Absorption

Absorption is the gradual loss of signal strength with depth, because the ground turns wave energy into heat. Conductivity is the main source. In a low-loss material:

**α ≈ 1.64 · σ / √εr** (α in dB/m, σ in mS/m)

- Dry sand (σ ≈ 0.1 mS/m, εr ≈ 4): α ≈ 0.08 dB/m. The signal travels tens of metres.
- Wet clay (σ ≈ 50 mS/m, εr ≈ 10): α ≈ 26 dB/m. The signal is gone within a metre or two.

This is the main practical limit. GPR works in sand, ice and concrete, and fails in clay or salt-affected soil.

Two further losses add to conduction:

- **Dielectric relaxation.** Rotating water molecules lag a high-frequency field, so the restoring “spring” also damps. These losses grow with frequency.
- **Geometric spreading and scattering.** The wave spreads over a larger area, and rough or heterogeneous ground sends energy away from the receiver. The signal weakens without becoming heat.

The loss tangent, **tan δ = σ / (ωε)**, compares conduction with storage. When tan δ ≪ 1, storage dominates and the wave propagates. When tan δ ≳ 1, conduction dominates, the field diffuses, and GPR stops working. Higher frequencies penetrate less: more relaxation loss, more scattering, and a shorter wavelength that is used up over fewer metres.

### Polarity

The transmitted pulse has a fixed shape, the wavelet (a Ricker is a positive–negative–positive wiggle). Each reflection is a copy of that wavelet scaled by

**R = (√ε₁ − √ε₂) / (√ε₁ + √ε₂)**

- **R negative**, into a slower material (higher εr): the wavelet flips. Dry sand over wet sand, the water table, a water-filled pipe, and metal (R ≈ −1).
- **R positive**, into a faster material (lower εr): the wavelet keeps its shape. An air void in concrete, an empty plastic pipe, an air-filled cavity or tunnel.

Take the reference polarity from a known event, such as the direct ground wave or a metal plate, and compare other reflections with it. An empty pipe and a water-filled pipe can make equally strong hyperbolas; the flip tells them apart.

Together:

- ε decides how fast the wave travels and where it reflects.
- σ decides how far it travels before it dies.
- μ can usually be ignored.
- Polarity tells which way the ε contrast goes at each reflector.

## 6. Things to remember

- Time → depth needs v. Wrong v = wrong depth.
- ε sets velocity and reflections. σ sets how deep you can see. μr ≈ 1 for ordinary ground.
- Wet ground: slower, more attenuation, less penetration. Clay and salt kill the signal within a metre or two.
- Polarity tells an empty void from a wet or metal target when both make strong hyperbolas.
- Raw hyperbolas are "smeared" targets. Migration collapses them back to a point, but it needs the right velocity.
- Small targets look bigger than they are in unmigrated data.
- Dewow, mean-trace removal and a band-pass clean the record. Gain only makes late arrivals visible; it does not remove noise. The chain is in [gpr_signal_processing.md](gpr_signal_processing.md).
- A manufacturer file is a header plus a block of integer samples. The header holds the trace count, the time window and the step along the line.
- A detector such as YOLO proposes a box around a hyperbola. The fit inside that box is what returns position, depth and velocity. That step is in [gpr_yolo.md](gpr_yolo.md).
- The field profiles are raw USGS lines from Forsythe refuge. They have no depth axis and no labelled targets.

*Source for sections 1–6: Jol, H. M. (Ed.). (2008). Ground penetrating radar theory and applications. elsevier. Ch. 1 (Annan) and Ch. 14 (Yarovoy).*

## 7. Common raw formats

An instrument file is a header plus the samples. The header says how many samples sit in a trace, how many traces are in the line, the time window, and the distance between traces. The samples are integers from the receiver. Nothing in the file is depth: depth still needs a velocity.

| Format | Maker | Files | Samples | What the header carries |
| --- | --- | --- | --- | --- |
| `.h5`, `.out` | gprMax | one HDF5 file | float field (Ez here) | `dt`, cell size, source and receiver positions. A merged B-scan drops the waveform, so the centre frequency is absent |
| `.rd3` + `.rad` | MALA | binary samples and a text header | int16, traces packed one after another | `SAMPLES`, `FREQUENCY` (sampling rate, MHz), `TIMEWINDOW`, `DISTANCE INTERVAL`, `LAST TRACE`, `STOP POSITION`, antenna name and separation, stacks |
| `.rd7` + `.rad` | MALA | same pair | int32 | the same `.rad` header |
| `.DZT` | GSSI | one binary file | uint8 or uint16, or int32 | a header of at least 1024 bytes: samples per trace, bits per sample, scans per metre or per second, range in ns, start time |
| `.DT1` + `.HD` | Sensors & Software | binary traces and a text header | int16, after a 128-byte header on every trace | number of traces, points per trace, time-zero sample, total time window, start and end position, step size, nominal frequency, antenna separation |

`LAST TRACE` in a MALA `.rad` file is the number of traces, not a zero-based index. Dinner Point Avenue has `LAST TRACE` 4188 and `DISTANCE INTERVAL` 0.152091 m, so the last trace is at 4187 × 0.152091 m = 636.805 m, which is `STOP POSITION`. The sampling rate is `FREQUENCY` 1280 MHz, so `dt` = 0.78125 ns. `TIMEWINDOW` 794.53125 ns is 1017 × `dt`.

`load_gprmax` reads the gprMax HDF5 into one array plus `dt`, trace spacing and source metadata. Field profiles in `data/raw/usgs_forsythe/` are MALA `.rd3`/`.rad` and are read with GPRPy.

A few counts that are easy to get wrong:

- Trace count comes from the header (`LAST TRACE`, or the number of traces in the `.HD`). Dividing the file size by the samples per trace is the check, not a second definition.
- `FREQUENCY` in a `.rad` file is the sampling rate, not the antenna frequency. The antenna is the `ANTENNAS` line (80 MHz on Dinner Point Avenue).
- Time in the header is two-way time. Position is distance along the line. Neither is depth.

## 8. The field profiles

The real data are ten common-offset GPR lines from the Edwin B. Forsythe National Wildlife Refuge, Atlantic County and Ocean County, New Jersey. They were collected for a U.S. Geological Survey study of the Kirkwood-Cohansey aquifer and of how sea-level rise might change fresh groundwater discharge to the wetlands. The lines themselves are the data release

Fiore, A. R. (2018). *Raw ground-penetrating radar data, Edwin B. Forsythe National Wildlife Refuge, New Jersey, 2014–15.* U.S. Geological Survey data release, https://doi.org/10.5066/F7JH3KBD

The hydrogeology and the interpreted sections are in Fiore, Voronin, and Wieben, USGS Scientific Investigations Report 2017-5135, https://doi.org/10.3133/sir20175135. This repository keeps the raw release, not those interpretations.

A MALA antenna was towed by hand. Acquisition software was MALA RAMAC GroundVision 1.4.6. The survey dates in the metadata are 24 October 2014, 13 and 14 November 2014, and 7 January 2015. The files in `data/raw/usgs_forsythe/` are the instrument records: a `.rd3` of int16 samples and a `.rad` header for each line. No filter, topographic correction, velocity, or depth conversion has been applied. Depth is still two-way time.

Two antennas were used.

| Lines | Antenna | Samples | Trace spacing | Time window | Antenna separation |
| --- | --- | --- | --- | --- | --- |
| DinnerPointAve (4188 traces), GameFarmRd-1 (2302), GameFarmRd-2 (1937) | MALA HDR 80 MHz, shielded | 1017 | 0.152091 m | 794.53125 ns | 0.48 m |
| ReedyCreek-1 (3689), ReedyCreek-2 (2653), ScottsLandingRd (2388), SouthWildlifeDr (1293), StaffordAve-eastwest (2688), StaffordAve-westeast (2615), WescottAve (1573) | MALA 100 MHz, shielded | 1024 | 0.303838 m | 502.529843 ns | 0.499872 m |

`LAST TRACE` in each `.rad` file is that trace count. Dinner Point Avenue is the line used for the GPRPy comparison in [gpr_signal_processing.md](gpr_signal_processing.md).

Two text files travel with the profiles:

- `markers.txt` notes surface features at a distance along the line, in feet: gates, vehicles, phragmites, ice puddles where the wheel stopped and traces were lost. Those notes explain gaps and noise. They are not picked reflections.
- `TraceCoordinates.txt` gives latitude and longitude (NAD83) for some traces. Positions came from a handheld GPS with a stated resolution of about 10 feet, and tree cover or cloud can make that worse. The coordinate is a place on the line, not a depth.

The survey target was the aquifer framework, not a mapped pipe. A hyperbola in these lines may be a buried object, a contact, or interference (the markers mention a vehicle and possible pipes along Reedy Creek). Nothing in the raw release labels those events. The synthetic pipe scans in `data/sim/` are the records that have a known target position and permittivity.
