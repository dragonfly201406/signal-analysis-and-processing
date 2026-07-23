"""Rife/Quinn改进FFT测频 —— 精细频率估计 (修正版)"""

import numpy as np
from scipy.fft import fft, fftfreq


def rife_freq_est(x: np.ndarray, fs: float, n_fft: int = None) -> dict:
    """Rife法: 利用最大谱线与次大谱线幅度比值精细估计

    适用于复数信号 exp(j*2*pi*f*t)

    Args:
        x: 输入信号(复数)
        fs: 采样率
        n_fft: FFT点数

    Returns:
        dict: 测频结果
    """
    N = len(x) if n_fft is None else n_fft
    X = fft(x, N)
    mag = np.abs(X)
    half = N // 2
    mag_pos = mag[:half]
    freqs = fftfreq(N, 1 / fs)[:half]
    df = freqs[1] - freqs[0]

    peak_idx = np.argmax(mag_pos)
    f_bin = freqs[peak_idx]

    # Rife插值: 使用最大谱线和次大谱线的幅度比
    if peak_idx == 0 or peak_idx >= half - 1:
        delta = 0
    else:
        r_ratio = mag_pos[peak_idx + 1] / (mag_pos[peak_idx] + 1e-30)
        l_ratio = mag_pos[peak_idx - 1] / (mag_pos[peak_idx] + 1e-30)
        if r_ratio >= l_ratio:
            delta = r_ratio / (1 + r_ratio)
        else:
            delta = -l_ratio / (1 + l_ratio)

    f_est = f_bin + delta * df

    return {
        'freq': f_est,
        'fft_peak_idx': peak_idx,
        'fft_bin': f_bin,
        'delta': delta,
        'amplitude': mag_pos[peak_idx],
        'method': 'Rife'
    }


def quinn_freq_est(x: np.ndarray, fs: float, n_fft: int = None) -> dict:
    """Quinn法: 利用FFT复数谱线比值精细估计 (复杂信号版)

    Quinn插值使用峰值相邻谱线的复数比值来估计分数偏移。
    对于复数信号 exp(j*2*pi*f*t)，公式为:
      α₁ = Re{X[k-1]/X[k]},  α₂ = Re{X[k+1]/X[k]}
      δ₁ = α₁ / (1 - α₁)
      δ₂ = -α₂ / (1 - α₂)

    Args:
        x: 输入信号(复数)
        fs: 采样率
        n_fft: FFT点数

    Returns:
        dict: 测频结果
    """
    N = len(x) if n_fft is None else n_fft
    X = fft(x, N)
    half = N // 2
    mag_pos = np.abs(X[:half])
    freqs = fftfreq(N, 1 / fs)[:half]
    df = freqs[1] - freqs[0]

    peak_idx = np.argmax(mag_pos)
    f_bin = freqs[peak_idx]

    if 0 < peak_idx < half - 1:
        # 左比值
        alpha1 = np.real(X[peak_idx - 1] / (X[peak_idx] + 1e-30))
        # 右比值
        alpha2 = np.real(X[peak_idx + 1] / (X[peak_idx] + 1e-30))

        # Quinn 插值公式 (适用于复数信号)
        delta1 = alpha1 / (1 - alpha1) if abs(1 - alpha1) > 1e-15 else 0
        delta2 = -alpha2 / (1 - alpha2) if abs(1 - alpha2) > 1e-15 else 0

        # 综合决策 (Quinn准则)
        if delta1 > 0 and delta2 > 0:
            delta = delta2
        elif delta1 < 0 and delta2 < 0:
            delta = delta1
        else:
            delta = (delta1 + delta2) / 2
    else:
        delta = 0

    f_est = f_bin + delta * df

    return {
        'freq': f_est,
        'fft_peak_idx': peak_idx,
        'fft_bin': f_bin,
        'delta': delta,
        'amplitude': mag_pos[peak_idx],
        'method': 'Quinn'
    }


def rife_quinn_freq_est(x: np.ndarray, fs: float, n_fft: int = None) -> dict:
    """综合Rife-Quinn法: Rife幅度插值 + Quinn相位插值加权平均

    根据SNR自适应加权: 高SNR偏向Quinn(相位精度高)，低SNR偏向Rife(幅度稳健)

    Args:
        x: 输入信号(复数)
        fs: 采样率
        n_fft: FFT点数

    Returns:
        dict: 测频结果
    """
    N = len(x) if n_fft is None else n_fft
    X = fft(x, N)
    half = N // 2
    mag_pos = np.abs(X[:half])
    freqs = fftfreq(N, 1 / fs)[:half]
    df = freqs[1] - freqs[0]

    peak_idx = np.argmax(mag_pos)
    f_bin = freqs[peak_idx]

    # Rife分量 (基于幅度)
    if peak_idx == 0 or peak_idx >= half - 1:
        delta_r = 0
    else:
        r_ratio = mag_pos[peak_idx + 1] / (mag_pos[peak_idx] + 1e-30)
        l_ratio = mag_pos[peak_idx - 1] / (mag_pos[peak_idx] + 1e-30)
        if r_ratio >= l_ratio:
            delta_r = r_ratio / (1 + r_ratio)
        else:
            delta_r = -l_ratio / (1 + l_ratio)

    # Quinn分量 (基于复数相位)
    if 0 < peak_idx < half - 1:
        alpha1 = np.real(X[peak_idx - 1] / (X[peak_idx] + 1e-30))
        alpha2 = np.real(X[peak_idx + 1] / (X[peak_idx] + 1e-30))
        delta1 = alpha1 / (1 - alpha1) if abs(1 - alpha1) > 1e-15 else 0
        delta2 = -alpha2 / (1 - alpha2) if abs(1 - alpha2) > 1e-15 else 0
        if delta1 > 0 and delta2 > 0:
            delta_q = delta2
        elif delta1 < 0 and delta2 < 0:
            delta_q = delta1
        else:
            delta_q = (delta1 + delta2) / 2
    else:
        delta_q = 0

    # 自适应加权: SNR高 → Quinn权重大
    peak_mag = mag_pos[peak_idx]
    noise_floor = np.median(mag_pos)
    snr_indicator = peak_mag / (noise_floor + 1e-30)

    if snr_indicator > 10:
        w_r, w_q = 0.3, 0.7
    elif snr_indicator > 3:
        w_r, w_q = 0.5, 0.5
    else:
        w_r, w_q = 0.7, 0.3

    delta = delta_r * w_r + delta_q * w_q
    f_est = f_bin + delta * df

    return {
        'freq': f_est,
        'fft_peak_idx': peak_idx,
        'fft_bin': f_bin,
        'delta_rife': delta_r,
        'delta_quinn': delta_q,
        'w_rife': w_r,
        'w_quinn': w_q,
        'amplitude': mag_pos[peak_idx],
        'method': 'Rife-Quinn'
    }
