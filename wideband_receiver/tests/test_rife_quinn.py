"""Rife/Quinn 插值回归测试"""

import numpy as np
import pytest

from config import FS, N_SAMPLES, NARROWBAND_FREQ
from single_channel_methods.basic_fft import basic_fft_freq_est
from single_channel_methods.rife_quinn import rife_freq_est, quinn_freq_est, rife_quinn_freq_est


def _tone(freq, n=N_SAMPLES, snr_db=30.0):
    t = np.arange(n) / FS
    x = np.exp(1j * 2 * np.pi * freq * t)
    return x


@pytest.mark.parametrize("func", [rife_freq_est, quinn_freq_est, rife_quinn_freq_est, basic_fft_freq_est])
def test_narrowband_freq_error_below_10khz(func):
    x = _tone(NARROWBAND_FREQ)
    r = func(x, FS)
    err = abs(r['freq'] - NARROWBAND_FREQ)
    assert err < 10e3, f"{func.__name__} error {err} Hz"