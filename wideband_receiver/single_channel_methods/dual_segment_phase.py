"""双段FFT相位差法测频 —— 利用两段FFT的相位差提高精度"""

import numpy as np
from scipy.fft import fft, fftfreq


def dual_segment_phase_freq_est(x: np.ndarray, fs: float,
                                 overlap: float = 0.5,
                                 n_fft: int = None) -> dict:
    """双段FFT相位差法测频 (修正版)

    将信号分成两段(可重叠)，分别做FFT，利用两个FFT峰值处相位差
    结合时延Delta估计频率偏差。

    关键修正: 扣除峰值bin索引引入的已知相位偏移 2*pi*k*start2/N1

    Args:
        x: 输入信号
        fs: 采样率
        overlap: 两段重叠率(0~1)
        n_fft: FFT点数

    Returns:
        dict: 测频结果
    """
    N = len(x)
    if n_fft is None:
        n_fft = N

    # 分成两段
    half_len = N // 2
    overlap_len = int(half_len * overlap)
    start2 = max(1, half_len - overlap_len)
    if start2 + half_len > N:
        start2 = N - half_len

    x1 = x[:half_len]
    x2 = x[start2:start2 + half_len]
    N1 = half_len

    # FFT
    n_fft_actual = min(n_fft, N1)
    X1 = fft(x1, n_fft_actual)
    X2 = fft(x2, n_fft_actual)

    half_fft = n_fft_actual // 2
    mag1 = np.abs(X1[:half_fft])
    freqs = fftfreq(n_fft_actual, 1 / fs)[:half_fft]
    df = freqs[1] - freqs[0]

    peak_idx = np.argmax(mag1)
    f_bin = freqs[peak_idx]
    k = peak_idx  # bin index

    # 提取相位
    z1 = X1[peak_idx]
    z2 = X2[peak_idx]
    phi1 = np.angle(z1)
    phi2 = np.angle(z2)

    # 原始相位差
    d_phi = phi2 - phi1

    # 扣除由bin index k引入的已知相位偏移
    # 对于复信号: phi2 - phi1 = 2*pi*f*start2/fs (理论上)
    # 其中 f = (k+δ)*df, 所以:
    # phi2 - phi1 = 2*pi*k*df*start2/fs + 2*pi*δ*df*start2/fs
    #              = 2*pi*k*start2/N1 + 2*pi*δ*start2/N1
    phi_bin = 2 * np.pi * k * start2 / N1
    phi_delta = d_phi - phi_bin

    # 解缠绕到 [-pi, pi]
    phi_delta = np.mod(phi_delta + np.pi, 2 * np.pi) - np.pi

    # 由剩余相位差估计分数偏移 δ
    # phi_delta = 2*pi*δ*start2/N1  (模2π)
    # δ = phi_delta * N1 / (2*pi*start2)
    delta = phi_delta * N1 / (2 * np.pi * start2)

    # 处理频率模糊: 可能存在的整数周期模糊
    # 2*pi*δ_true*start2/N1 = phi_delta + 2*pi*m
    # δ_true = δ + m * N1/start2
    # 模糊量 = N1/start2 (bin单位)
    ambiguity = N1 / start2

    # 选择使 total δ 最接近0的m值 (因为f_bin是最近的量化频率)
    m = round(-delta / ambiguity)
    delta_corrected = delta + m * ambiguity

    # 最终频率
    f_est = f_bin + delta_corrected * df
    delta_f = delta_corrected * df

    return {
        'freq': f_est,
        'fft_peak_idx': peak_idx,
        'fft_bin': f_bin,
        'delta_f': delta_f,
        'phase_diff_raw': d_phi,
        'phi_bin_offset': phi_bin,
        'phi_delta': phi_delta,
        'delta_frac': delta,
        'delta_corrected': delta_corrected,
        'start2': start2,
        'amplitude': mag1[peak_idx],
        'method': 'Dual-Segment Phase'
    }
