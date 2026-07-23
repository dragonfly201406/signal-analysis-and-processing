"""仿真结果可视化"""

import os
from typing import Dict, List, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import FIGURES_DIR, NARROWBAND_FREQ


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_single_method_rmse(single_results: Dict[str, List[Dict]], out_dir: str = FIGURES_DIR) -> str:
    """窄带各方法 RMSE 柱状图 (取 SNR=10dB 附近)"""
    _ensure_dir(out_dir)
    nb = single_results.get('narrowband', [])
    methods, rmses = [], []
    for mr in nb:
        srs = mr.get('snr_results', [])
        sr = next((s for s in srs if s['snr_db'] == 10), srs[-1] if srs else None)
        if sr is None:
            continue
        methods.append(mr['method'])
        rmses.append(sr['rmse_hz'])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    ax.bar(x, rmses, color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right')
    ax.set_ylabel('RMSE (Hz)')
    ax.set_title('Single-channel methods (narrowband, SNR=10 dB)')
    ax.set_yscale('log')
    fig.tight_layout()
    path = os.path.join(out_dir, 'narrowband_rmse.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_channelization_detection(channel_results: List[Dict], out_dir: str = FIGURES_DIR) -> str:
    """信道化粗检测率 (SNR=10dB)"""
    _ensure_dir(out_dir)
    names, rates = [], []
    for cr in channel_results:
        srs = cr.get('snr_results', [])
        sr = next((s for s in srs if s['snr_db'] == 10), srs[0] if srs else None)
        if sr is None:
            continue
        names.append(cr['scheme'])
        rates.append(sr['overall_detection_rate'] * 100)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, rates, color='coral')
    ax.set_ylabel('Detection rate (%)')
    ax.set_title('Channelization coarse detection (SNR=10 dB)')
    plt.xticks(rotation=20, ha='right')
    fig.tight_layout()
    path = os.path.join(out_dir, 'channelization_detection.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_spectrum_example(x: np.ndarray, fs: float, out_dir: str = FIGURES_DIR) -> str:
    """示例频谱图"""
    from scipy.fft import fft, fftfreq

    _ensure_dir(out_dir)
    N = len(x)
    X = fft(x, N)
    mag = np.fft.fftshift(np.abs(X))
    freqs = np.fft.fftshift(fftfreq(N, 1 / fs)) / 1e6

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, 20 * np.log10(mag + 1e-12))
    ax.axvline(NARROWBAND_FREQ / 1e6, color='g', ls='--', alpha=0.5, label='NB ref')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Multi-band test signal spectrum')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, 'spectrum_example.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path