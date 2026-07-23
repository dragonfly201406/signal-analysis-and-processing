"""非均匀信道化方案 —— 根据信号带宽自适应分配子信道"""

import numpy as np
import time
from scipy.fft import fft, fftfreq
from config import (FS, N_FFT_CHANNEL, MASK_ALPHA,
                    NONUNIFORM_THRESHOLD, NONUNIFORM_CHANNEL_BW)


def _find_active_regions(freqs: np.ndarray, energy: np.ndarray,
                         threshold: float) -> list:
    """在能量序列上找连续活跃区间 (freqs 与 energy 等长)"""
    half = len(freqs)
    active_regions = []
    i = 0
    while i < half:
        if energy[i] > threshold:
            start = i
            while i < half and energy[i] > threshold * 0.8:
                i += 1
            end = i
            bw = freqs[end - 1] - freqs[start]
            if bw <= 0:
                bw = abs(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
            f_center = (freqs[start] + freqs[end - 1]) / 2
            active_regions.append({
                'f_start': freqs[start],
                'f_end': freqs[end - 1],
                'f_center': f_center,
                'bw': bw,
                'start_bin': start,
                'end_bin': end
            })
        else:
            i += 1
    return active_regions


def nonuniform_channelize(x: np.ndarray, fs: float = FS) -> dict:
    """非均匀信道化

    策略：
    1. 先进行一次粗FFT感知频谱占用 (全频 [-fs/2, fs/2)，与单信道FFT一致)
    2. 根据频谱能量分布自适应分配信道带宽
    3. 窄带区域用窄信道(高分辨)，稀疏区域用宽信道

    Args:
        x: 输入信号
        fs: 采样率

    Returns:
        dict: 处理结果
    """
    t_start = time.perf_counter()

    N = len(x)
    # 1. 粗FFT感知 (fftshift 全频谱)
    n_coarse = min(max(2048, N_FFT_CHANNEL // 2), max(N, 256))
    X_coarse = fft(x, n_coarse)
    mag_coarse = np.abs(X_coarse)
    mag_coarse_shifted = np.fft.fftshift(mag_coarse)
    freqs_coarse = np.fft.fftshift(fftfreq(n_coarse, 1 / fs))
    df_coarse = freqs_coarse[1] - freqs_coarse[0]

    # 2. 能量检测 + 自适应信道划分
    noise_floor = np.median(mag_coarse_shifted)
    energy = mag_coarse_shifted / (noise_floor + 1e-30)

    active_regions = _find_active_regions(
        freqs_coarse, energy, NONUNIFORM_THRESHOLD
    )

    # 3. 没有活跃区域时退化为均匀信道化 (正频监视带 0 ~ fs/2)
    if len(active_regions) == 0:
        bw = fs / 2
        ch_bw = bw / 8
        active_regions = []
        for ch in range(8):
            f0 = ch * ch_bw
            f1 = (ch + 1) * ch_bw
            active_regions.append({
                'f_start': f0,
                'f_end': f1,
                'f_center': (ch + 0.5) * ch_bw,
                'bw': ch_bw,
                'start_bin': 0,
                'end_bin': 0
            })

    # 4. 对每个活跃区域做精细处理
    all_peaks = []
    channel_info = []

    for reg_idx, region in enumerate(active_regions):
        f_start = region['f_start']
        f_end = region['f_end']
        bw = max(region['bw'], NONUNIFORM_CHANNEL_BW / 10)

        f_center = (f_start + f_end) / 2
        x_bb = x * np.exp(-1j * 2 * np.pi * f_center * np.arange(N) / fs)

        target_resolution = min(bw / 100, 5e3)
        n_fft_local = max(256, int(fs / target_resolution))
        n_fft_local = min(n_fft_local, N)

        dec_factor = max(1, int(fs / (2 * max(bw, 1.0))))
        if dec_factor > N:
            dec_factor = max(1, N // 2)
        x_dec = x_bb[::dec_factor] if dec_factor > 1 else x_bb

        n_fft = min(len(x_dec), n_fft_local)
        if n_fft < 8:
            continue
        X = fft(x_dec, n_fft)
        mag = np.abs(X)
        mag_shifted = np.fft.fftshift(mag)
        freqs_local = np.fft.fftshift(fftfreq(n_fft, 1 / (fs / dec_factor)))
        df = freqs_local[1] - freqs_local[0]

        freqs_global = freqs_local + f_center

        threshold = np.median(mag_shifted) * (1 + MASK_ALPHA)

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
                    'region': reg_idx,
                    'bin_idx': i
                })

        channel_info.append({
            'region': reg_idx,
            'f_center': f_center,
            'bw': bw,
            'spectrum': mag_shifted,
            'freqs': freqs_global,
            'n_fft': n_fft,
            'dec_factor': dec_factor
        })

    all_peaks.sort(key=lambda p: p['amplitude'], reverse=True)

    t_elapsed = time.perf_counter() - t_start

    return {
        'peaks': all_peaks,
        'n_peaks': len(all_peaks),
        'n_regions': len(active_regions),
        'regions': active_regions,
        'channel_info': channel_info,
        'coarse_spectrum': mag_coarse_shifted,
        'coarse_freqs': freqs_coarse,
        'time_sec': t_elapsed,
        'method': 'Non-Uniform Channelization'
    }