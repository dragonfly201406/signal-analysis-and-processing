"""信号生成器 —— 产生3类测试信号"""

import numpy as np
from typing import List, Tuple, Optional
from config import (
    FS, NARROWBAND_FREQ, MEDIUM_FREQ_1, MEDIUM_FREQ_2, WIDEBAND_FREQ
)


def _add_awgn(x: np.ndarray, snr_db: float) -> np.ndarray:
    """加性高斯白噪声 (复数信号)"""
    if snr_db >= 50:
        return x
    signal_power = np.mean(np.abs(x) ** 2)
    noise_power = signal_power * 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(len(x)) + 1j * np.random.randn(len(x))
    )
    return x + noise


def generate_narrowband(fs: float = FS, duration: float = 1e-6,
                        snr_db: float = 10.0, n_samples: Optional[int] = None,
                        freq: float = NARROWBAND_FREQ) -> Tuple[np.ndarray, np.ndarray]:
    """生成窄带信号

    Args:
        fs: 采样率
        duration: 信号时长(秒)
        snr_db: 信噪比(dB)
        n_samples: 采样点数(覆盖duration)
        freq: 信号频率

    Returns:
        (t, x): 时间轴和复数信号
    """
    if n_samples is None:
        n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    x = np.exp(1j * 2 * np.pi * freq * t)
    x = _add_awgn(x, snr_db)
    return t, x


def generate_dual_medium(fs: float = FS, duration: float = 1e-6,
                         snr_db: float = 10.0, n_samples: Optional[int] = None,
                         freq1: float = MEDIUM_FREQ_1,
                         freq2: float = MEDIUM_FREQ_2,
                         amp_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """生成中等带宽双信号

    Args:
        fs: 采样率
        duration: 信号时长(秒)
        snr_db: 信噪比(dB)
        n_samples: 采样点数
        freq1, freq2: 两个信号的频率
        amp_ratio: 幅度比

    Returns:
        (t, x): 时间轴和复数信号
    """
    if n_samples is None:
        n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    x1 = np.exp(1j * 2 * np.pi * freq1 * t)
    x2 = amp_ratio * np.exp(1j * 2 * np.pi * freq2 * t)
    x = x1 + x2
    x = _add_awgn(x, snr_db)
    return t, x


def generate_wideband(fs: float = FS, duration: float = 1e-6,
                      snr_db: float = 10.0, n_samples: Optional[int] = None,
                      freq: float = WIDEBAND_FREQ) -> Tuple[np.ndarray, np.ndarray]:
    """生成宽带空闲信号（稀疏监测）
    用较低带宽的线性调频或纯单音模拟

    Args:
        fs: 采样率
        duration: 信号时长(秒)
        snr_db: 信噪比(dB)
        n_samples: 采样点数
        freq: 信号中心频率

    Returns:
        (t, x): 时间轴和复数信号
    """
    if n_samples is None:
        n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs
    x = np.exp(1j * 2 * np.pi * freq * t)
    x = _add_awgn(x, snr_db)
    return t, x


def generate_multi_band_signal(fs: float = FS, duration: float = 1e-6,
                                snr_db: float = 10.0,
                                n_samples: Optional[int] = None
                                ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """生成包含三类信号的全频段信号

    Returns:
        (t, x, info): 时间轴、复合信号、信号信息字典
    """
    if n_samples is None:
        n_samples = int(fs * duration)
    t = np.arange(n_samples) / fs

    x = np.zeros(n_samples, dtype=complex)

    # 窄带信号
    x += np.exp(1j * 2 * np.pi * NARROWBAND_FREQ * t)

    # 中等带宽双信号
    x += np.exp(1j * 2 * np.pi * MEDIUM_FREQ_1 * t)
    x += np.exp(1j * 2 * np.pi * MEDIUM_FREQ_2 * t)

    # 宽带稀疏信号
    x += np.exp(1j * 2 * np.pi * WIDEBAND_FREQ * t)

    x = _add_awgn(x, snr_db)

    info = {
        'narrowband': {'freq': NARROWBAND_FREQ, 'type': 'narrowband'},
        'medium1': {'freq': MEDIUM_FREQ_1, 'type': 'medium'},
        'medium2': {'freq': MEDIUM_FREQ_2, 'type': 'medium'},
        'wideband': {'freq': WIDEBAND_FREQ, 'type': 'wideband'},
    }

    return t, x, info