"""GPRPy reference processing for one USGS MALA profile.

Run in the gprpy environment from the repository root:
    conda activate gprpy
    python src/02c_gprpy_reference.py

Saves GPRPy's array after each step, so src/02b_preprocessing.py can be
compared against it. Parameters are in samples (dt = 0.78125 ns):
dewow 32 samples = 25 ns, AGC 64 samples = 50 ns.
"""

import numpy as np
import gprpy.gprpy as gp

PROFILE = "data/raw/usgs_forsythe/DinnerPointAve.rd3"
OUT = "data/raw/usgs_forsythe/DinnerPointAve_gprpy_steps.npz"
DEWOW_SAMPLES = 32
AGC_SAMPLES = 64

mygpr = gp.gprpyProfile()
mygpr.importdata(PROFILE)
steps = {"raw": np.array(mygpr.data, dtype=float)}

mygpr.dewow(DEWOW_SAMPLES)
steps["dewow"] = np.array(mygpr.data, dtype=float)

mygpr.agcGain(AGC_SAMPLES)
steps["agc"] = np.array(mygpr.data, dtype=float)

np.savez(OUT,
         twtt=np.asarray(mygpr.twtt),
         pos=np.asarray(mygpr.profilePos),
         dewow_samples=DEWOW_SAMPLES,
         agc_samples=AGC_SAMPLES,
         **steps)
print("saved", OUT, {name: array.shape for name, array in steps.items()})