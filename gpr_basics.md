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

*Source: Jol, H. M. (Ed.). (2008). Ground penetrating radar theory and applications. elsevier. Ch. 1 (Annan) and Ch. 14 (Yarovoy).*
