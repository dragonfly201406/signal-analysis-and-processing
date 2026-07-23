"""信道化方案回归 —— 单音应出现在峰值列表中"""

import numpy as np

from config import FS, N_SAMPLES, NARROWBAND_FREQ, CHANNEL_DETECTION_TOLERANCE_HZ
from channelization.single_channel_fft import single_channel_fft_process
from channelization.direct_uniform import direct_uniform_channelize
from channelization.non_uniform import nonuniform_channelize


def _tone_at(freq, n=N_SAMPLES * 2):
    t = np.arange(n) / FS
    return np.exp(1j * 2 * np.pi * freq * t)


def _peak_near(peaks, f_true, tol=CHANNEL_DETECTION_TOLERANCE_HZ):
    freqs = [p['freq'] for p in peaks]
    return any(abs(f - f_true) < tol for f in freqs)


def test_single_channel_fft_detects_narrowband():
    x = _tone_at(NARROWBAND_FREQ)
    r = single_channel_fft_process(x, FS)
    assert _peak_near(r['peaks'], NARROWBAND_FREQ)


def test_direct_uniform_detects_narrowband():
    x = _tone_at(NARROWBAND_FREQ)
    r = direct_uniform_channelize(x, FS, n_channels=16)
    assert _peak_near(r['peaks'], NARROWBAND_FREQ)


def test_nonuniform_coarse_full_spectrum_length():
    x = _tone_at(NARROWBAND_FREQ)
    r = nonuniform_channelize(x, FS)
    assert len(r['coarse_freqs']) == len(r['coarse_spectrum'])
    assert _peak_near(r['peaks'], NARROWBAND_FREQ, tol=20e6)