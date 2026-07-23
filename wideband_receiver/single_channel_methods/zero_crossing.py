"""过零检测法测频 —— 利用时域过零点统计频率

注意: 高频复指数下实部/虚部过零率与 f 关系复杂，|f|/fs 大时不适用。
"""

import numpy as np
from config import TIME_DOMAIN_FREQ_RATIO_MAX


def zero_crossing_freq_est(x: np.ndarray, fs: float) -> dict:
    """过零检测法测频

    对实部信号的过零点计数，利用单位时间内过零点数估计频率。
    对于复数信号，分别处理I/Q路后取平均。

    Args:
        x: 输入信号(复数)
        fs: 采样率

    Returns:
        dict: 测频结果
    """
    N = len(x)
    duration = N / fs

    # 对实部和虚部分别做过零检测
    x_real = np.real(x)
    x_imag = np.imag(x)

    def _zero_crossing_rate(sig: np.ndarray) -> int:
        """计算过零次数: 检测所有符号变化"""
        # signbit: True for negative, False for positive/zero
        signs = np.signbit(sig)
        # diff: True where sign changes (both directions)
        return int(np.sum(np.diff(signs) != 0))

    zc_real = _zero_crossing_rate(x_real)
    zc_imag = _zero_crossing_rate(x_imag)

    # 每秒过零次数 = 2 * 频率 (对于正弦信号)
    # 但差分后的sign: 每个周期前后沿各一次，共2次零交叉
    # 对于实部cos信号，每秒过零次数 = 2*f
    freq_real = zc_real / (2 * duration)
    freq_imag = zc_imag / (2 * duration)

    f_est = (freq_real + freq_imag) / 2

    # 分段统计稳定性
    n_segments = max(4, N // 128)
    n_segments = min(n_segments, N // 4)
    seg_len = N // n_segments
    freqs_seg = []

    for i in range(n_segments):
        seg = x_real[i * seg_len:min((i + 1) * seg_len, N)]
        if len(seg) < 4:
            continue
        zc = _zero_crossing_rate(seg)
        seg_dur = len(seg) / fs
        if seg_dur > 0 and zc > 0:
            freqs_seg.append(zc / (2 * seg_dur))

    f_std = np.std(freqs_seg) if len(freqs_seg) > 1 else 0.0

    applicable = abs(f_est) / fs <= TIME_DOMAIN_FREQ_RATIO_MAX

    return {
        'freq': f_est,
        'freq_std': f_std,
        'zero_crossing_real': zc_real,
        'zero_crossing_imag': zc_imag,
        'n_segments': len(freqs_seg),
        'time_domain_applicable': applicable,
        'method': 'Zero-Crossing'
    }
