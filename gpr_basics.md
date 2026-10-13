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

## 5. Things to remember

- Time → depth needs v. Wrong v = wrong depth.
- Wet ground: slower, more attenuation, less penetration.
- Raw hyperbolas are "smeared" targets. Migration collapses them back to a point, but it needs the right velocity.
- Small targets look bigger than they are in unmigrated data.

*Source: Jol, H. M. (Ed.). (2008). Ground penetrating radar theory and applications. elsevier. Ch. 1 (Annan) and Ch. 14 (Yarovoy).*
