"""相位差分法测频 —— 利用瞬时相位的差分估计频率

注意: |f|/fs 较大时相位增量易折叠，宽带高频场景不适用；评估框架会标注 applicability。
"""

import numpy as np
from config import TIME_DOMAIN_FREQ_RATIO_MAX


def phase_difference_freq_est(x: np.ndarray, fs: float) -> dict:
    """相位差分法测频

    对复数信号的瞬时相位做差分，平均后得到频率估计。

    Args:
        x: 输入信号(复数)
        fs: 采样率

    Returns:
        dict: 测频结果
    """
    # 瞬时相位
    phi = np.angle(x)

    # 相位差分
    d_phi = np.diff(phi)

    # 解缠绕
    d_phi = np.mod(d_phi + np.pi, 2 * np.pi) - np.pi

    # 平均差分相位 => 频率
    # f = mean(d_phi) / (2*pi) * fs
    mean_d_phi = np.mean(d_phi)
    f_est = mean_d_phi * fs / (2 * np.pi)

    # 频率标准差(反映估计稳定性)
    f_std = np.std(d_phi) * fs / (2 * np.pi)

    # 信噪比估计(基于相位差分方差)
    snr_est = -20 * np.log10(np.std(d_phi)) if np.std(d_phi) > 0 else 40

    applicable = abs(f_est) / fs <= TIME_DOMAIN_FREQ_RATIO_MAX

    return {
        'freq': f_est,
        'freq_std': f_std,
        'mean_phase_diff': mean_d_phi,
        'snr_estimate': snr_est,
        'time_domain_applicable': applicable,
        'method': 'Phase Difference'
    }
