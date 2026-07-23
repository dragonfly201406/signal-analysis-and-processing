"""基础FFT测频 —— 频谱峰值+插值"""

import numpy as np
from scipy.fft import fft, fftfreq


def basic_fft_freq_est(x: np.ndarray, fs: float,
                       n_fft: int = None) -> dict:
    """基础FFT测频: 峰值搜索 + 抛物线插值

    Args:
        x: 输入信号(复数)
        fs: 采样率(Hz)
        n_fft: FFT点数(None则用len(x))

    Returns:
        dict: {'freq': 估计频率, 'fft_peak_idx': 峰值索引,
               'fft_bin': 量化频率, 'amplitude': 峰值幅度}
    """
    N = len(x) if n_fft is None else n_fft
    X = fft(x, N)
    mag = np.abs(X)

    # 取正频率部分
    half = N // 2
    mag_pos = mag[:half]
    freqs = fftfreq(N, 1 / fs)[:half]

    # 峰值搜索
    peak_idx = np.argmax(mag_pos)
    f_bin = freqs[peak_idx]

    # 抛物线插值 (三点)
    if 0 < peak_idx < half - 1:
        y1, y2, y3 = mag_pos[peak_idx - 1], mag_pos[peak_idx], mag_pos[peak_idx + 1]
        denom = 2 * (y1 - 2 * y2 + y3)
        if abs(denom) > 1e-15:
            delta = (y1 - y3) / denom
            f_est = f_bin + delta * (freqs[1] - freqs[0])
        else:
            f_est = f_bin
    else:
        f_est = f_bin

    return {
        'freq': f_est,
        'fft_peak_idx': peak_idx,
        'fft_bin': f_bin,
        'amplitude': mag_pos[peak_idx],
        'method': 'Basic FFT'
    }
