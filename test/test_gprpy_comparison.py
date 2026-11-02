"""Dewow and AGC on DinnerPointAve agree with GPRPy away from the trace ends."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compare_gprpy import (
    AGC_WINDOW_NS,
    DEWOW_WINDOW_NS,
    NPZ,
    compare,
    load_reference,
    pearson,
    window_samples,
)

# The reference file is gitignored. Without it the comparison cannot run.
pytestmark = pytest.mark.skipif(
    not NPZ.is_file(),
    reason="run src/gprpy_reference.py in the gprpy env to write the reference npz",
)


def test_interior_correlation_with_gprpy():
    raw, gprpy_dewow, gprpy_agc, _twtt, _pos, _stored = load_reference(NPZ)
    _rows, images = compare(raw, gprpy_dewow, gprpy_agc)
    dewow_correlation = _interior(images["dewow"], window_samples(DEWOW_WINDOW_NS))
    agc_correlation = _interior(images["agc"], window_samples(AGC_WINDOW_NS))

    # GPRPy's interior dewow window is 33 samples and this dewow uses 32,
    # with a different even-window centre. On this profile the interior
    # correlation is 0.9997. The bound is 0.99.
    assert dewow_correlation > 0.99

    # AGC is a different normalisation, not a failed match. GPRPy divides
    # by the L2 norm of a 65-sample window; gain_agc divides by the RMS of
    # a 64-sample window, so amplitudes differ by about sqrt(64) and the
    # relative RMS is about 7. Pearson correlation ignores that scale.
    # Interior correlation on this profile is 0.9995. The full trace falls
    # to about 0.976 because GPRPy freezes one gain over the first and last
    # half-window, which is why those samples are excluded. The bound is
    # 0.99 on the interior only.
    assert agc_correlation > 0.99


def _interior(images, trim):
    reference, mine, _difference = images
    return pearson(mine[trim:-trim], reference[trim:-trim])
