"""双段相位差测频回归"""

import numpy as np

from config import FS, N_SAMPLES, NARROWBAND_FREQ
from single_channel_methods.dual_segment_phase import dual_segment_phase_freq_est


def test_dual_segment_narrowband():
    n = N_SAMPLES
    t = np.arange(n) / FS
    x = np.exp(1j * 2 * np.pi * NARROWBAND_FREQ * t)
    r = dual_segment_phase_freq_est(x, FS)
    assert abs(r['freq'] - NARROWBAND_FREQ) < 15e3