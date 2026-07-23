"""直接均匀信道化方案 —— 均匀分割子信道后独立FFT"""

import numpy as np
import time
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from config import FS, N_CHANNELS, N_FFT_CHANNEL, MASK_ALPHA


def direct_uniform_channelize(x: np.ndarray, fs: float = FS,
                               n_channels: int = N_CHANNELS,
                               n_fft_per_channel: int = None) -> dict:
    """直接均匀信道化

    将带宽均匀分割为 n_channels 个子信道，每个子信道
    进行下变频、抗混叠滤波、抽取、独立FFT处理。

    Args:
        x: 输入信号
        fs: 采样率
        n_channels: 子信道数
        n_fft_per_channel: 每个子信道FFT点数

    Returns:
        dict: 处理结果
    """
    t_start = time.perf_counter()

    N = len(x)
    channel_bw = fs / n_channels

    if n_fft_per_channel is None:
        n_fft_per_channel = max(256, N // n_channels)

    all_peaks = []
    channel_spectra = []

    filter_order = 128
    cutoff = 1.0 / n_channels
    lpf = scipy_signal.firwin(filter_order + 1, cutoff, window=('kaiser', 8.0))
    n = np.arange(N)

    for ch in range(n_channels):
        f_center = (ch + 0.5) * channel_bw
        x_bb = x * np.exp(-1j * 2 * np.pi * f_center * n / fs)

        # filtfilt: 零相位抗混叠滤波，避免 lfilter 群延迟未补偿问题
        padlen = min(3 * (len(lpf) - 1), max(0, N - 1))
        if padlen > 0 and N > padlen:
            x_filtered = scipy_signal.filtfilt(lpf, 1.0, x_bb, padlen=padlen)
        else:
            x_filtered = scipy_signal.lfilter(lpf, 1.0, x_bb)

        dec_factor = n_channels
        x_dec = x_filtered[::dec_factor]

        n_fft = min(len(x_dec), n_fft_per_channel)
        if n_fft < 4:
            continue
        X = fft(x_dec, n_fft)
        mag = np.abs(X)
        mag_shifted = np.fft.fftshift(mag)
        freqs_local = np.fft.fftshift(fftfreq(n_fft, 1 / (fs / dec_factor)))
        df = freqs_local[1] - freqs_local[0]
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
        'n_channels': n_channels,
        'channel_bw': channel_bw,
        'channel_spectra': channel_spectra,
        'time_sec': t_elapsed,
        'method': 'Direct Uniform Channelization'
    }