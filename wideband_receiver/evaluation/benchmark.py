"""评估与基准测试框架"""

import numpy as np
import time
from typing import Callable, Dict, List
from tabulate import tabulate
from evaluation.metrics import rmse, benchmark_function, success_rate
from evaluation.helpers import (
    extract_freq_estimates, match_estimate_to_true,
    dual_tone_metrics, time_domain_method_applicable,
)
from config import (
    SNR_DB_VALUES, N_MONTE_CARLO, N_SAMPLES, FS,
    NARROWBAND_FREQ, MEDIUM_FREQ_1, MEDIUM_FREQ_2, WIDEBAND_FREQ,
    NARROWBAND_ACCURACY, WIDEBAND_ACCURACY,
    MEDIUM_RESOLVE_SPACING, CHANNEL_DETECTION_TOLERANCE_HZ,
    TIME_DOMAIN_FREQ_RATIO_MAX,
)
from utils.signal_generator import (
    generate_narrowband, generate_dual_medium, generate_wideband,
    generate_multi_band_signal
)

TIME_DOMAIN_METHODS = {'Phase Difference', 'Zero-Crossing'}


def evaluate_single_channel_method(method_func: Callable,
                                    method_name: str,
                                    signal_type: str = 'narrowband',
                                    snr_db_values: List[float] = None,
                                    n_mc: int = N_MONTE_CARLO) -> Dict:
    """评估单个测频方法在不同SNR下的性能"""
    if snr_db_values is None:
        snr_db_values = SNR_DB_VALUES

    results = {'method': method_name, 'signal_type': signal_type, 'snr_results': []}

    for snr_db in snr_db_values:
        estimates = []
        estimates_dual = []
        times = []
        true_freq = NARROWBAND_FREQ
        true_f2 = MEDIUM_FREQ_2

        for _ in range(n_mc):
            if signal_type == 'narrowband':
                _, x = generate_narrowband(snr_db=snr_db, n_samples=N_SAMPLES)
                true_freq = NARROWBAND_FREQ
            elif signal_type == 'medium':
                _, x = generate_dual_medium(snr_db=snr_db, n_samples=N_SAMPLES)
                true_freq = MEDIUM_FREQ_1
            elif signal_type == 'wideband':
                _, x = generate_wideband(snr_db=snr_db, n_samples=N_SAMPLES)
                true_freq = WIDEBAND_FREQ
            else:
                _, x, _ = generate_multi_band_signal(snr_db=snr_db, n_samples=N_SAMPLES)
                true_freq = NARROWBAND_FREQ

            t0 = time.perf_counter()
            result = method_func(x, FS)
            t1 = time.perf_counter()

            est_list = extract_freq_estimates(result)
            if signal_type == 'medium':
                estimates_dual.append(est_list)
                f_est = match_estimate_to_true(est_list, true_freq)
                f_est = f_est if f_est is not None else (est_list[0] if est_list else 0.0)
            else:
                if est_list:
                    f_est = est_list[0]
                elif 'freq' in result:
                    f_est = result['freq']
                else:
                    f_est = 0.0

            estimates.append(f_est)
            times.append((t1 - t0) * 1e6)

        estimates = np.array(estimates)

        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        if std_est > 0:
            valid_mask = np.abs(estimates - mean_est) < 3 * std_est
        else:
            valid_mask = np.ones_like(estimates, dtype=bool)
        estimates_valid = estimates[valid_mask]

        snr_result = {
            'snr_db': snr_db,
            'rmse_hz': float(rmse(estimates_valid, true_freq)),
            'mean_time_us': float(np.mean(times)),
            'std_time_us': float(np.std(times)),
            'mean_estimate': float(np.mean(estimates_valid)),
            'n_valid': int(np.sum(valid_mask)),
            'success_rate_narrow': float(
                success_rate(estimates_valid, true_freq, NARROWBAND_ACCURACY)
            ),
            'success_rate_wide': float(
                success_rate(estimates_valid, true_freq, WIDEBAND_ACCURACY)
            ),
        }

        if signal_type == 'medium':
            dual = dual_tone_metrics(
                estimates_dual, MEDIUM_FREQ_1, MEDIUM_FREQ_2,
                match_tol_hz=CHANNEL_DETECTION_TOLERANCE_HZ,
                resolve_spacing_hz=MEDIUM_RESOLVE_SPACING,
            )
            snr_result['medium_dual'] = dual
            snr_result['rmse_hz'] = dual['rmse_hz_combined']

        if method_name in TIME_DOMAIN_METHODS:
            ref_f = true_freq
            snr_result['time_domain_applicable'] = time_domain_method_applicable(ref_f, FS)
            snr_result['freq_ratio'] = abs(ref_f) / FS
            if not snr_result['time_domain_applicable']:
                snr_result['note'] = (
                    f'|f|/fs={snr_result["freq_ratio"]:.3f} > '
                    f'{TIME_DOMAIN_FREQ_RATIO_MAX}，时域法仅供参考'
                )

        results['snr_results'].append(snr_result)

    return results


def evaluate_channelization_scheme(scheme_func: Callable,
                                    scheme_name: str,
                                    snr_db_values: List[float] = None,
                                    n_mc: int = 10) -> Dict:
    """评估信道化方案性能 (粗检测率，非测频 RMSE)"""
    if snr_db_values is None:
        snr_db_values = SNR_DB_VALUES

    results = {
        'scheme': scheme_name,
        'detection_tolerance_hz': CHANNEL_DETECTION_TOLERANCE_HZ,
        'snr_results': [],
    }
    det_tol = CHANNEL_DETECTION_TOLERANCE_HZ

    for snr_db in snr_db_values:
        detections = []
        times = []
        n_peaks_list = []

        for _ in range(n_mc):
            _, x, _ = generate_multi_band_signal(snr_db=snr_db, n_samples=N_SAMPLES * 2)

            t0 = time.perf_counter()
            scheme_result = scheme_func(x, FS)
            t1 = time.perf_counter()

            times.append((t1 - t0) * 1e6)
            peaks = scheme_result.get('peaks', [])
            n_peaks_list.append(len(peaks))
            detections.append({
                'n_peaks': len(peaks),
                'freqs': [p['freq'] for p in peaks[:12]]
            })

        avg_n_peaks = np.mean(n_peaks_list)
        true_freqs = [NARROWBAND_FREQ, MEDIUM_FREQ_1, MEDIUM_FREQ_2, WIDEBAND_FREQ]
        detect_rates = []
        for f_true in true_freqs:
            detected = 0
            for det in detections:
                if any(abs(f - f_true) < det_tol for f in det['freqs']):
                    detected += 1
            detect_rates.append(detected / len(detections))

        snr_result = {
            'snr_db': snr_db,
            'mean_time_us': float(np.mean(times)),
            'std_time_us': float(np.std(times)),
            'avg_n_peaks': float(avg_n_peaks),
            'detection_rates': {
                'narrowband': float(detect_rates[0]),
                'medium1': float(detect_rates[1]),
                'medium2': float(detect_rates[2]),
                'wideband': float(detect_rates[3])
            },
            'overall_detection_rate': float(np.mean(detect_rates))
        }
        results['snr_results'].append(snr_result)

    return results


def format_results_table(results_list: List[Dict],
                         title: str = "评估结果") -> str:
    """将评估结果格式化为 Markdown 表格"""
    lines = [f"\n## {title}\n"]

    for result in results_list:
        method = result.get('method') or result.get('scheme', 'Unknown')
        lines.append(f"\n### {method}\n")
        lines.append("| SNR(dB) | RMSE(Hz) | 均值时延(us) | 成功率(%) |")
        lines.append("|---------|----------|-------------|----------|")

        for sr in result['snr_results']:
            lines.append(
                f"| {sr['snr_db']:+3d} | {sr['rmse_hz']:.2e} | "
                f"{sr['mean_time_us']:.1f} | "
                f"{sr.get('success_rate_narrow', 0):.1f} |"
            )
            if 'medium_dual' in sr:
                d = sr['medium_dual']
                lines.append(
                    f"| (双音) | f1 RMSE {d['rmse_hz_f1']:.2e} | "
                    f"f2 {d['rmse_hz_f2']:.2e} | 双峰率 {d['dual_detection_rate']:.1%} |"
                )

    return '\n'.join(lines)