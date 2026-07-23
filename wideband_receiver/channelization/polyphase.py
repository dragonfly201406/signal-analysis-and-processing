"""多相滤波信道化方案 —— 演示版 PFB (多相分支 + 块 FFT)

说明: 本实现为教学/对比用的简化多相滤波器组，与 scipy 标准 PFB 在边界处理上
可能略有差异；回归测试见 tests/test_channelization.py。
"""

import numpy as np
import time
from scipy import signal
from scipy.fft import fft, fftfreq
from config import FS, POLYPHASE_DECIM, POLYPHASE_TAPS, MASK_ALPHA


def _polyphase_filter_bank(x: np.ndarray, prototype: np.ndarray, M: int) -> np.ndarray:
    """多相分析: 输出 shape (M, n_blocks)，每块 M 路分支滤波后做 FFT 得信道化结果"""
    n_taps = len(prototype)
    taps_per_phase = n_taps // M
    poly_coeffs = np.array([prototype[p::M] for p in range(M)])
    n_blocks = len(x) // M
    if n_blocks < 1:
        return np.zeros((M, 0), dtype=complex)

    channel_out = np.zeros((M, n_blocks), dtype=complex)
    for n in range(n_blocks):
        base = (n + 1) * M
        for p in range(M):
            end_idx = base - p
            start_idx = end_idx - taps_per_phase
            if start_idx < 0:
                seg = np.zeros(taps_per_phase, dtype=complex)
                valid = x[0:end_idx]
                seg[-len(valid):] = valid[::-1]
            else:
                seg = x[start_idx:end_idx][::-1]
            if len(seg) < taps_per_phase:
                seg = np.pad(seg, (0, taps_per_phase - len(seg)))
            channel_out[p, n] = np.dot(seg[:taps_per_phase], poly_coeffs[p])

    return channel_out


def polyphase_channelize(x: np.ndarray, fs: float = FS,
                          decim: int = POLYPHASE_DECIM,
                          n_taps: int = POLYPHASE_TAPS) -> dict:
    """多相滤波信道化 (简化 PFB + 各信道 FFT 寻峰)"""
    t_start = time.perf_counter()

    M = decim
    N = len(x)
    n_taps = max((n_taps // M) * M, M * 4)
    prototype = signal.firwin(n_taps, 1.0 / M, window=('kaiser', 8.0))

    channel_out = _polyphase_filter_bank(x, prototype, M)
    channel_spectra = []
    all_peaks = []
    channel_bw = fs / M

    for ch in range(M):
        ch_out = channel_out[ch, :]
        n_fft = min(128, len(ch_out))
        if n_fft < 4:
            continue
        X = fft(ch_out, n_fft)
        mag = np.abs(X)
        mag_shifted = np.fft.fftshift(mag)
        freqs_local = np.fft.fftshift(fftfreq(n_fft, 1 / (fs / M)))
        df = freqs_local[1] - freqs_local[0]
        f_center = (ch + 0.5) * channel_bw
        freqs_global = freqs_local + f_center

        noise_floor = np.median(mag_shifted)
        threshold = noise_floor * (1 + MASK_ALPHA)

        for i in range(1, n_fft - 1):
            if (mag_shifted[i] > mag_shifted[i - 1] and
                mag_shifted[i] > mag_shifted[i + 1] and
                mag_shifted[i] > threshold):
                y1, y2, y3 = mag_shifted[i - 1], mag_shifted[i], mag_shifted[i + 1]
                denom = 2 * (y1 - 2 * y2 + y3)
                if abs(denom) > 1e-15:
                    delta = (y1 - y3) / denom
                    f_est = freqs_global[i] + delta * df
                else:
                    f_est = freqs_global[i]
                all_peaks.append({
                    'freq': f_est,
                    'amplitude': mag_shifted[i],
                    'channel': ch,
                    'bin_idx': i
                })

        channel_spectra.append({
            'channel': ch,
            'f_center': f_center,
            'spectrum': mag_shifted,
            'freqs': freqs_global
        })

    all_peaks.sort(key=lambda p: p['amplitude'], reverse=True)
    t_elapsed = time.perf_counter() - t_start

    return {
        'peaks': all_peaks,
        'n_peaks': len(all_peaks),
        'n_channels': M,
        'channel_bw': channel_bw,
        'channel_spectra': channel_spectra,
        'filter_taps': n_taps,
        'time_sec': t_elapsed,
        'method': 'Polyphase Channelization (simplified PFB)'
    }