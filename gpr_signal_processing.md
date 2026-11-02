# Signal processing

A B-scan is one array shaped `(samples, traces)`. Each column is one trace. Sample spacing `dt` is in seconds. Travel time, velocity and the hyperbola that this processing leaves in the image are in [gpr_basics.md](gpr_basics.md). Three tools touch the array, and they do different jobs.

**gprMax** solves Maxwell's equations on a grid with the finite-difference time-domain method. The input file states the domain, the cell size, the materials (εr, σ, μr), a source waveform and a receiver. A Ricker pulse is the usual source. Stepping the antenna writes one trace per position; merging those traces makes the B-scan. The stored quantity here is the electric field Ez, in volts per metre, already a floating-point time series. gprMax does not denoise. It is the source of the synthetic profiles in `data/sim/`.

**SciPy** is the numerical library behind `src/02b_preprocessing.py`. It is not a GPR program. A running mean and a running RMS use `scipy.ndimage.uniform_filter1d`. The band-pass uses `scipy.signal.butter` (a Butterworth filter, second-order sections) and `sosfiltfilt`, which runs the filter forwards and backwards so the pulse does not move in time. NumPy holds the arrays. Every step copies the input and keeps its dtype.

**GPRPy** reads field files, plots them, and applies the usual processing chain (dewow, gain, background removal, migration). Here it is the reference implementation, not the code under test. `src/02c_gprpy_reference.py` runs GPRPy's dewow (32 samples, 25 ns) and AGC (64 samples, 50 ns) on the Dinner Point Avenue profile and saves the arrays. `src/03b_compare_gprpy.py` repeats those two steps with this repository's functions. The windows are not identical: GPRPy's interior window is one sample longer and centred on the sample, and its AGC divides by the L2 norm of the window rather than by the RMS. Dewow still agrees. AGC agrees in shape and differs by a nearly constant scale of about √64.

The processing order in `src/02b_preprocessing.py` is:

1. **Time zero.** Find the first break on each trace and shift it to sample 0. Later times then share an origin. This is alignment, not a filter.
2. **Dewow.** Subtract a running mean along time. Antenna coupling and induction add a slow bias, well below the pulse frequency. The window has to be longer than the pulse and shorter than the record, so the reflection stays and the bias goes.
3. **Background removal.** Subtract the mean trace. The direct wave and system ringing arrive at nearly the same time on every trace, so they form a horizontal band. A pipe changes from trace to trace and remains. On a short profile the hyperbola itself is in the mean, and subtracting it leaves an artefact near the apex.
4. **Band-pass.** Keep about half to twice the antenna frequency. Energy below that is leftover wow. Energy above it is noise. Zero phase keeps the arrival time.
5. **Gain.** Multiply by a power of travel time (`gain_tpow`), or divide by a running RMS (`gain_agc`). Spreading and absorption make a deep reflection weaker. Gain restores its visibility. It also raises whatever noise is left in the quiet part of the trace. A small constant (1e-8) keeps a silent sample from being divided by zero.

Steps 2–4 are the denoise. Step 5 is display. Migration, which would collapse the hyperbola to a point, is not in this chain. It needs a velocity, and a wrong velocity moves the target.
