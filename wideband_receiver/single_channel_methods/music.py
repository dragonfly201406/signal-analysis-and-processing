"""MUSIC超分辨测频算法 —— 子空间分解超分辨频率估计"""

import numpy as np
from scipy.fft import fft, fftfreq
from config import FREQ_SEARCH_GRANULARITY


def music_freq_est(x: np.ndarray, fs: float,
                   n_sources: int = 3,
                   n_subspace: int = None) -> dict:
    """MUSIC超分辨测频

    利用自相关矩阵的特征分解，将信号子空间与噪声子空间正交分解，
    通过谱峰搜索实现超分辨频率估计。

    使用FFT预先定位候选频段，大幅缩小搜索范围。

    Args:
        x: 输入信号(复数)
        fs: 采样率
        n_sources: 信号源数
        n_subspace: 子空间维度(默认 min(64, len(x)//8))

    Returns:
        dict: 包含频率估计列表和伪谱
    """
    N = len(x)
    if n_subspace is None:
        n_subspace = min(64, N // 8)
    n_subspace = max(n_subspace, n_sources + 2)
    if n_subspace >= N:
        n_subspace = N - 1
    n_subspace = min(n_subspace, N - n_sources - 1)

    # 构造自相关矩阵 (前后向平滑)
    R = np.zeros((n_subspace, n_subspace), dtype=complex)

    # 前向
    for i in range(N - n_subspace):
        R += np.outer(x[i:i + n_subspace], np.conj(x[i:i + n_subspace]))

    # 后向 (前后向平均)
    x_conj = np.conj(x[::-1])
    for i in range(N - n_subspace):
        R += np.outer(x_conj[i:i + n_subspace], np.conj(x_conj[i:i + n_subspace]))

    R /= 2 * (N - n_subspace)

    # 特征分解
    try:
        eigvals, eigvecs = np.linalg.eigh(R)
    except np.linalg.LinAlgError:
        return {'freqs': [], 'music_spectrum': np.array([]),
                'freq_search': np.array([]), 'n_sources': n_sources,
                'eigvals': np.array([]), 'method': 'MUSIC'}

    # 降序排列
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 噪声子空间
    noise_subspace = eigvecs[:, n_sources:]

    # 预搜索: 用FFT粗定位候选频率
    n_fft_coarse = min(2048, N)
    Xc = fft(x, n_fft_coarse)
    mag_c = np.abs(Xc[:n_fft_coarse // 2])
    freqs_c = fftfreq(n_fft_coarse, 1 / fs)[:n_fft_coarse // 2]

    # 找到主要谱峰 (最多n_sources个)
    threshold = np.median(mag_c) + 2.0 * np.std(mag_c)
    candidate_centers = []
    i = 1
    while i < len(mag_c) - 1:
        if mag_c[i] > mag_c[i - 1] and mag_c[i] > mag_c[i + 1] and mag_c[i] > threshold:
            candidate_centers.append(freqs_c[i])
            i += 2
        else:
            i += 1

    if len(candidate_centers) == 0:
        candidate_centers = [freqs_c[np.argmax(mag_c)]]

    # 限制候选数
    candidate_centers = candidate_centers[:n_sources]

    # 对每个候选频率进行MUSIC精细搜索
    search_band = fs / n_fft_coarse * 4  # 搜索带宽 = 4个FFT bin
    search_step = float(FREQ_SEARCH_GRANULARITY)

    all_peaks = []
    all_pseudo = []
    all_freqs = []

    for f_center in candidate_centers:
        f_lo = max(0, f_center - search_band / 2)
        f_hi = min(fs / 2, f_center + search_band / 2)
        f_search = np.arange(f_lo, f_hi, search_step)

        if len(f_search) < 5:
            continue

        # 向量化MUSIC伪谱计算（避免逐点循环）
        # 构造导向矩阵 A (n_subspace × n_search)
        n_f = len(f_search)
        A = np.exp(1j * 2 * np.pi * np.outer(np.arange(n_subspace), f_search / fs))

        # En * En^H: (n_subspace × n_subspace)
        EnEnH = noise_subspace @ np.conj(noise_subspace.T)

        # 伪谱: 1 / diag(A^H * EnEnH * A)
        # = 1 / sum_{i,j} conj(A[i]) * EnEnH[i,j] * A[j]
        # 向量化: 1 / sum(abs(A^H * En)^2, axis=0)
        AH_En = np.conj(A).T @ noise_subspace  # (n_f × n_noise)
        pseudo = 1.0 / (np.sum(np.abs(AH_En) ** 2, axis=1) + 1e-30)

        # 在伪谱中找峰
        local_threshold = np.mean(pseudo) + 1.0 * np.std(pseudo)
        for j in range(1, n_f - 1):
            if (pseudo[j] > pseudo[j - 1] and
                    pseudo[j] > pseudo[j + 1] and
                    pseudo[j] > local_threshold):
                y1, y2, y3 = pseudo[j - 1], pseudo[j], pseudo[j + 1]
                denom = 2 * (y1 - 2 * y2 + y3)
                if abs(denom) > 1e-15:
                    delta = (y1 - y3) / denom
                    f_peak = f_search[j] + delta * search_step
                else:
                    f_peak = f_search[j]
                all_peaks.append(f_peak)

        all_pseudo.extend(pseudo.tolist())
        all_freqs.extend(f_search.tolist())

    # 去重: 合并接近的峰值
    if len(all_peaks) > 0:
        all_peaks = sorted(all_peaks)
        merged = [all_peaks[0]]
        for p in all_peaks[1:]:
            if p - merged[-1] > search_step * 3:
                merged.append(p)
        all_peaks = merged[:n_sources]
    else:
        all_peaks = []

    return {
        'freqs': all_peaks,
        'music_spectrum': np.array(all_pseudo) if all_pseudo else np.array([]),
        'freq_search': np.array(all_freqs) if all_freqs else np.array([]),
        'n_sources': n_sources,
        'eigvals': eigvals,
        'method': 'MUSIC'
    }
