"""评估辅助: 频率估计提取、双音匹配等"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from config import TIME_DOMAIN_FREQ_RATIO_MAX


def time_domain_method_applicable(estimated_freq_hz: float, fs: float) -> bool:
    """相位差分/过零法在 |f|/fs 过大时不适用"""
    return abs(estimated_freq_hz) / fs <= TIME_DOMAIN_FREQ_RATIO_MAX


def extract_freq_estimates(result: dict) -> List[float]:
    """从测频函数返回值提取频率列表 (Hz)"""
    if 'freqs' in result and isinstance(result['freqs'], (list, np.ndarray)):
        freqs = list(result['freqs'])
        return [float(f) for f in freqs if f is not None]
    if 'freq' in result and result['freq'] is not None:
        return [float(result['freq'])]
    return []


def match_estimate_to_true(estimates: List[float], true_freq: float,
                           max_error_hz: float = 5e6) -> Optional[float]:
    """在估计列表中选取最接近 true_freq 的一个"""
    if not estimates:
        return None
    errs = [abs(e - true_freq) for e in estimates]
    i = int(np.argmin(errs))
    if errs[i] > max_error_hz:
        return None
    return estimates[i]


def dual_tone_metrics(estimates_list: List[List[float]],
                      true_f1: float, true_f2: float,
                      match_tol_hz: float = 5e6,
                      resolve_spacing_hz: float = 20e6) -> Dict:
    """双音蒙特卡洛: 两频 RMSE、双峰检出率、间隔可分辨率"""
    e1, e2 = [], []
    both_detected = 0
    resolved = 0
    n = len(estimates_list)

    for ests in estimates_list:
        m1 = match_estimate_to_true(ests, true_f1, match_tol_hz)
        m2 = match_estimate_to_true(ests, true_f2, match_tol_hz)
        if m1 is not None and m2 is not None and abs(m2 - m1) > 1e5:
            both_detected += 1
            e1.append(m1)
            e2.append(m2)
            if abs(m2 - m1) >= 0.5 * resolve_spacing_hz:
                resolved += 1

    e1 = np.array(e1) if e1 else np.array([])
    e2 = np.array(e2) if e2 else np.array([])

    def _rmse(arr, true):
        if len(arr) == 0:
            return float('nan')
        return float(np.sqrt(np.mean((arr - true) ** 2)))

    return {
        'rmse_hz_f1': _rmse(e1, true_f1),
        'rmse_hz_f2': _rmse(e2, true_f2),
        'rmse_hz_combined': float(np.nanmean([
            _rmse(e1, true_f1), _rmse(e2, true_f2)
        ])) if len(e1) and len(e2) else float('nan'),
        'dual_detection_rate': both_detected / n if n else 0.0,
        'resolution_rate': resolved / n if n else 0.0,
        'n_mc': n,
        'n_valid_pairs': len(e1),
    }