"""
超宽带测频接收机信道化方案设计与仿真 —— 主入口

对以下内容进行完整仿真与评估:
1. 四种信道化方案对比
2. 八种单信道测频算法对比 (含 Rife/Quinn 等变体)
3. 生成 JSON 结果、图表与分析报告
"""

import numpy as np
import time
import os
import json
from datetime import datetime
from tabulate import tabulate

from config import *
from utils.signal_generator import (
    generate_narrowband, generate_dual_medium, generate_wideband,
    generate_multi_band_signal
)

from single_channel_methods.basic_fft import basic_fft_freq_est
from single_channel_methods.rife_quinn import rife_freq_est, quinn_freq_est, rife_quinn_freq_est
from single_channel_methods.dual_segment_phase import dual_segment_phase_freq_est
from single_channel_methods.phase_difference import phase_difference_freq_est
from single_channel_methods.zero_crossing import zero_crossing_freq_est
from single_channel_methods.music import music_freq_est

from channelization.single_channel_fft import single_channel_fft_process
from channelization.direct_uniform import direct_uniform_channelize
from channelization.polyphase import polyphase_channelize
from channelization.non_uniform import nonuniform_channelize

from evaluation.metrics import benchmark_function
from evaluation.benchmark import evaluate_single_channel_method, evaluate_channelization_scheme
from evaluation.visualization import (
    plot_single_method_rmse, plot_channelization_detection, plot_spectrum_example,
)
from report.generate_report import generate_analysis_report


def _json_sanitize(obj):
    """将 numpy 类型转为 JSON 可序列化"""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_quick_test(methods, signal_type, snr_values, n_mc, label):
    """快速运行单信道方法测试"""
    results = []
    for name, func in methods:
        is_music = ('MUSIC' in name)
        mc = min(n_mc, 5) if is_music else n_mc
        result = evaluate_single_channel_method(
            func, name, signal_type=signal_type,
            snr_db_values=snr_values, n_mc=mc
        )
        results.append(result)
        r_idx = len(snr_values) - 2 if len(snr_values) >= 2 else 0
        r = result['snr_results'][r_idx]
        extra = ''
        if signal_type == 'medium' and 'medium_dual' in r:
            d = r['medium_dual']
            extra = f" | DualDet={d['dual_detection_rate']:.1%}"
        print(f"  {name:20s} | RMSE={r['rmse_hz']:.2e} Hz | "
              f"Time={r['mean_time_us']:.1f} us | "
              f"SR={r.get('success_rate_narrow', 0):.1f}%{extra}")
    return results


def run_single_method_comparison():
    """运行单信道测频方法对比"""
    print("=" * 70)
    print("任务一: 单信道测频方法对比")
    print("=" * 70)

    fast_methods = [
        ('Basic FFT', basic_fft_freq_est),
        ('Rife', rife_freq_est),
        ('Quinn', quinn_freq_est),
        ('Rife-Quinn', rife_quinn_freq_est),
        ('Dual-Seg Phase', dual_segment_phase_freq_est),
        ('Phase Diff', phase_difference_freq_est),
        ('Zero-Crossing', zero_crossing_freq_est),
    ]

    music_methods = [('MUSIC', lambda x, fs: music_freq_est(x, fs, n_sources=N_MUSIC_SOURCES))]

    print("\n--- 窄带信号测试 (f=120.123MHz) ---")
    narrowband_results = run_quick_test(
        fast_methods, 'narrowband', SNR_DB_VALUES, N_MC_SINGLE, '窄带')
    for name, func in music_methods:
        r = evaluate_single_channel_method(func, name, signal_type='narrowband',
                                           snr_db_values=[10, 20], n_mc=3)
        narrowband_results.append(r)
        print(f"  {name:20s} | RMSE={r['snr_results'][0]['rmse_hz']:.2e} Hz | "
              f"Time={r['snr_results'][0]['mean_time_us']:.1f} us")

    print("\n--- 中带宽双信号测试 (500.456MHz, 520.789MHz) ---")
    medium_results = run_quick_test(
        fast_methods, 'medium', [5, 10, 20], N_MC_SINGLE, '中带宽')
    for name, func in music_methods:
        r = evaluate_single_channel_method(func, name, signal_type='medium',
                                           snr_db_values=[10, 20], n_mc=3)
        medium_results.append(r)
        if r['snr_results']:
            d = r['snr_results'][0].get('medium_dual', {})
            print(f"  {name:20s} | RMSE={r['snr_results'][0]['rmse_hz']:.2e} Hz | "
                  f"DualDet={d.get('dual_detection_rate', 0):.1%}")

    print("\n--- 宽带信号测试 (1480.321MHz) ---")
    wideband_results = run_quick_test(
        fast_methods, 'wideband', SNR_DB_VALUES, N_MC_SINGLE, '宽带')
    for name, func in music_methods:
        r = evaluate_single_channel_method(func, name, signal_type='wideband',
                                           snr_db_values=[10, 20], n_mc=3)
        wideband_results.append(r)
        if r['snr_results']:
            note = r['snr_results'][0].get('note', '')
            print(f"  {name:20s} | RMSE={r['snr_results'][0]['rmse_hz']:.2e} Hz | "
                  f"Time={r['snr_results'][0]['mean_time_us']:.1f} us {note}")

    return {
        'narrowband': narrowband_results,
        'medium': medium_results,
        'wideband': wideband_results
    }


def run_channelization_comparison():
    """运行四种信道化方案对比"""
    print("\n" + "=" * 70)
    print("任务二: 四种信道化方案对比 (粗检测, 容差 "
          f"{CHANNEL_DETECTION_TOLERANCE_HZ/1e6:.0f} MHz)")
    print("=" * 70)

    schemes = [
        ('Single-Channel FFT', single_channel_fft_process),
        ('Direct Uniform', direct_uniform_channelize),
        ('Polyphase', lambda x, fs: polyphase_channelize(x, fs, decim=POLYPHASE_DECIM, n_taps=POLYPHASE_TAPS)),
        ('Non-Uniform', nonuniform_channelize),
    ]

    print("\n--- 全频段信号检测性能 ---")
    all_results = []
    for name, func in schemes:
        result = evaluate_channelization_scheme(
            func, name,
            snr_db_values=[5, 10, 20],
            n_mc=N_MC_CHANNEL
        )
        all_results.append(result)
        r = result['snr_results'][1] if len(result['snr_results']) >= 2 else result['snr_results'][0]
        print(f"  {name:25s} | Time={r['mean_time_us']:.1f} us | "
              f"Avg Peaks={r['avg_n_peaks']:.1f} | "
              f"Detect Rate={r['overall_detection_rate']:.1%}")

    return all_results


def run_single_method_benchmark():
    """单测频方法精细基准测试"""
    print("\n" + "=" * 70)
    print("任务三: 单测频方法精细基准 - 窄带信号 SNR=10dB")
    print("=" * 70)

    _, x = generate_narrowband(snr_db=10.0, n_samples=N_SAMPLES)

    methods = [
        ('Basic FFT', basic_fft_freq_est, 20),
        ('Rife', rife_freq_est, 20),
        ('Quinn', quinn_freq_est, 20),
        ('Rife-Quinn', rife_quinn_freq_est, 20),
        ('Dual-Seg Phase', dual_segment_phase_freq_est, 20),
        ('Phase Diff', phase_difference_freq_est, 20),
        ('Zero-Crossing', zero_crossing_freq_est, 20),
        ('MUSIC', lambda x, fs=FS: music_freq_est(x, fs, n_sources=N_MUSIC_SOURCES), 5),
    ]

    print(f"{'Method':<18} {'Estimate(MHz)':<16} {'Error(kHz)':<14} {'Time(us)':<12}")
    print("-" * 60)

    bench_results = []
    for name, func, n_runs in methods:
        bm = benchmark_function(func, x, FS, n_runs=n_runs)
        last = bm['last_result']
        if 'freqs' in last and isinstance(last['freqs'], list) and len(last['freqs']) > 0:
            f_est = last['freqs'][0]
        else:
            f_est = last.get('freq', 0)
        err_khz = (f_est - NARROWBAND_FREQ) / 1e3
        print(f"  {name:<18} {f_est/1e6:<12.6f}    {err_khz:<+10.3f}    {bm['mean_us']:<8.1f}")
        bench_results.append({
            'method': name,
            'freq_est_hz': f_est,
            'error_khz': err_khz,
            'time_us': bm['mean_us']
        })

    return bench_results


def run_channelization_benchmark():
    """信道化方案精细基准测试"""
    print("\n" + "=" * 70)
    print("任务四: 信道化方案精细基准 - 全频段混合信号 SNR=10dB")
    print("=" * 70)

    _, x, info = generate_multi_band_signal(snr_db=10.0, n_samples=N_SAMPLES * 2)

    schemes = [
        ('Single-Channel FFT', lambda x: single_channel_fft_process(x, FS)),
        ('Direct Uniform(16ch)', lambda x: direct_uniform_channelize(x, FS, n_channels=N_CHANNELS)),
        ('Polyphase(16ch)', lambda x: polyphase_channelize(x, FS, decim=POLYPHASE_DECIM, n_taps=POLYPHASE_TAPS)),
        ('Non-Uniform', lambda x: nonuniform_channelize(x, FS)),
    ]

    true_freqs = [NARROWBAND_FREQ, MEDIUM_FREQ_1, MEDIUM_FREQ_2, WIDEBAND_FREQ]
    det_tol = CHANNEL_DETECTION_TOLERANCE_HZ
    results = []

    print(f"{'Scheme':<24} {'Peaks':<8} {'Time(us)':<12} {'Detected':<28}")
    print("-" * 72)

    for name, func in schemes:
        bm = benchmark_function(func, x, n_runs=5)
        result = bm['last_result']
        peaks = result.get('peaks', [])
        peak_freqs = [p['freq'] for p in peaks[:8]]

        detected = [any(abs(f - tf) < det_tol for f in peak_freqs) for tf in true_freqs]
        detected_str = f"N:{detected[0]} M1:{detected[1]} M2:{detected[2]} W:{detected[3]}"

        print(f"  {name:<24} {len(peaks):<8} {bm['mean_us']:<12.1f} {detected_str:<28}")

        results.append({
            'scheme': name,
            'n_peaks': len(peaks),
            'time_us': bm['mean_us'],
            'detected': detected,
            'detection_tolerance_hz': det_tol,
            'peak_freqs_mhz': [f / 1e6 for f in peak_freqs[:6]]
        })

    return results


def print_summary_tables(single_method_results, channelization_results,
                          bench_results, channel_bench):
    """打印综合对比汇总表"""
    print("\n" + "=" * 70)
    print("综合对比汇总")
    print("=" * 70)

    print("\n表1: 单信道测频方法 (窄带, SNR=10dB)")
    header = ["方法", "RMSE(Hz)", "耗时(us)", "成功率(<5kHz)"]
    rows = []
    for mr in single_method_results['narrowband']:
        sr = next((s for s in mr['snr_results'] if s['snr_db'] == 10),
                  mr['snr_results'][-1])
        rows.append([
            mr['method'],
            f"{sr['rmse_hz']:.2e}",
            f"{sr['mean_time_us']:.1f}",
            f"{sr.get('success_rate_narrow', 0):.1f}%"
        ])
    print(tabulate(rows, headers=header, tablefmt='grid'))

    print("\n表2: 信道化粗检测 (全频段, SNR=10dB)")
    header = ["方案", "耗时(us)", "平均峰值", "检测率"]
    rows = []
    for cr in channelization_results:
        sr = next((s for s in cr['snr_results'] if s['snr_db'] == 10),
                  cr['snr_results'][0])
        rows.append([
            cr['scheme'],
            f"{sr['mean_time_us']:.1f}",
            f"{sr['avg_n_peaks']:.1f}",
            f"{sr['overall_detection_rate']:.1%}"
        ])
    print(tabulate(rows, headers=header, tablefmt='grid'))

    print("\n表3: 精细基准 (窄带 SNR=10dB)")
    header = ["方法", "估计(MHz)", "误差(kHz)", "耗时(us)"]
    rows = [[b['method'], f"{b['freq_est_hz']/1e6:.6f}",
             f"{b['error_khz']:.3f}", f"{b['time_us']:.1f}"]
            for b in bench_results]
    print(tabulate(rows, headers=header, tablefmt='grid'))

    print("\n表4: 理论频率分辨率")
    rows = [
        [f"单信道FFT({N_FFT_CHANNEL}点)", f"{FS/N_FFT_CHANNEL/1e3:.2f} kHz"],
        [f"直接均匀({N_CHANNELS}ch×128点)", f"{FS/N_CHANNELS/128/1e3:.2f} kHz"],
        [f"多相滤波({N_CHANNELS}ch×128点)", f"{FS/N_CHANNELS/128/1e3:.2f} kHz"],
        ["非均匀自适应", "依赖信号带宽，窄带区可达 kHz 级"],
    ]
    print(tabulate(rows, headers=["方案", "分辨率"], tablefmt='grid'))


def save_results(single_results, channel_results, bench_results, channel_bench,
                 figure_paths, results_dir: str = RESULTS_DIR):
    """保存完整仿真结果到 JSON"""
    os.makedirs(results_dir, exist_ok=True)
    payload = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': {
            'fs_hz': FS,
            'bandwidth_hz': BANDWIDTH,
            'n_samples': N_SAMPLES,
            'n_fft_channel': N_FFT_CHANNEL,
            'n_mc_single': N_MC_SINGLE,
            'n_mc_channel': N_MC_CHANNEL,
            'channel_detection_tolerance_hz': CHANNEL_DETECTION_TOLERANCE_HZ,
            'narrowband_freq_hz': NARROWBAND_FREQ,
            'medium_freq_1_hz': MEDIUM_FREQ_1,
            'medium_freq_2_hz': MEDIUM_FREQ_2,
            'wideband_freq_hz': WIDEBAND_FREQ,
        },
        'single_channel': single_results,
        'channelization': channel_results,
        'benchmark_single': bench_results,
        'benchmark_channelization': channel_bench,
        'figures': figure_paths,
    }
    path = os.path.join(results_dir, 'simulation_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_sanitize(payload), f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(results_dir, 'simulation_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(_json_sanitize(payload['config'] | {'timestamp': payload['timestamp']}),
                  f, indent=2)

    print(f"\n完整结果: {path}")
    print(f"配置摘要: {summary_path}")
    return path


def main():
    """主入口"""
    print("=" * 70)
    print("超宽带测频接收机信道化方案设计与仿真")
    print(f"采样率: {FS/1e9:.2f} GHz | 带宽: {BANDWIDTH/1e9:.2f} GHz")
    print(f"窄带: {NARROWBAND_FREQ/1e6:.3f} MHz | "
          f"中带: {MEDIUM_FREQ_1/1e6:.3f}/{MEDIUM_FREQ_2/1e6:.3f} MHz | "
          f"宽带: {WIDEBAND_FREQ/1e6:.3f} MHz")
    print(f"蒙特卡洛: {N_MC_SINGLE}次(单信道)/{N_MC_CHANNEL}次(信道化) | "
          f"采样: {N_SAMPLES}点 | FFT: {N_FFT_CHANNEL}点")
    print("=" * 70)

    t_total = time.perf_counter()

    single_results = run_single_method_comparison()
    channel_results = run_channelization_comparison()
    bench_results = run_single_method_benchmark()
    channel_bench = run_channelization_benchmark()

    print_summary_tables(single_results, channel_results, bench_results, channel_bench)

    figure_paths = []
    try:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        figure_paths.append(plot_single_method_rmse(single_results))
        figure_paths.append(plot_channelization_detection(channel_results))
        _, x_demo, _ = generate_multi_band_signal(snr_db=15.0, n_samples=N_SAMPLES)
        figure_paths.append(plot_spectrum_example(x_demo, FS))
        print(f"\n图表已保存至 {FIGURES_DIR}/")
    except Exception as e:
        print(f"\n[警告] 图表生成失败: {e}")

    report_path = generate_analysis_report(
        single_results, channel_results, bench_results, channel_bench, figure_paths
    )
    print(f"分析报告: {report_path}")

    save_results(single_results, channel_results, bench_results, channel_bench, figure_paths)

    elapsed = time.perf_counter() - t_total
    print(f"\n总耗时: {elapsed:.1f} 秒")
    print("=" * 70)


if __name__ == '__main__':
    main()