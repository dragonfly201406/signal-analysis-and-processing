"""单信道 FFT 信道化方案 —— 对整个带宽做一次大FFT"""

import numpy as np
import time
from scipy.fft import fft, fftfreq
from config import FS, N_FFT_CHANNEL, MASK_ALPHA


def single_channel_fft_process(x: np.ndarray, fs: float = FS,
                                n_fft: int = None) -> dict:
    """单信道FFT处理: 整带宽一次FFT + 峰值检测

    对复数信号进行全频谱分析 [-fs/2, fs/2]。

    Args:
        x: 输入信号 (复数基带)
        fs: 采样率
        n_fft: FFT点数 (默认 N_FFT_CHANNEL，不足则零填充)

    Returns:
        dict: 处理结果
    """
    t_start = time.perf_counter()

    N = n_fft if n_fft is not None else N_FFT_CHANNEL
    X = fft(x, N)
    mag = np.abs(X)
    # 复数信号需要用fftshift查看完整频谱 [-fs/2, fs/2]
    mag_shifted = np.fft.fftshift(mag)
    freqs = np.fft.fftshift(fftfreq(N, 1 / fs))
    df = freqs[1] - freqs[0]

    # 信号检测
    noise_floor = np.median(mag_shifted)
    threshold = noise_floor * (1 + MASK_ALPHA)

    # 寻找所有超过门限的峰值
    peaks = []
    i = 1
    while i < N - 1:
        if (mag_shifted[i] > mag_shifted[i - 1] and
            mag_shifted[i] > mag_shifted[i + 1] and
            mag_shifted[i] > threshold):
            # 抛物线插值
            y1, y2, y3 = mag_shifted[i - 1], mag_shifted[i], mag_shifted[i + 1]
            denom = 2 * (y1 - 2 * y2 + y3)
            if abs(denom) > 1e-15:
                delta = (y1 - y3) / denom
                f_est = freqs[i] + delta * df
            else:
                f_est = freqs[i]

            peaks.append({
                'freq': f_est,
                'amplitude': mag_shifted[i],
                'bin_idx': i
            })
            i += 2
        else:
            i += 1

    # 按幅度排序
    peaks.sort(key=lambda p: p['amplitude'], reverse=True)

    t_elapsed = time.perf_counter() - t_start

    return {
        'peaks': peaks,
        'n_peaks': len(peaks),
        'spectrum': mag_shifted,
        'freqs': freqs,
        'df': df,
        'n_fft': N,
        'time_sec': t_elapsed,
        'method': 'Single-Channel FFT'
    }