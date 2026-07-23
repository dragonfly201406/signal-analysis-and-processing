"""根据仿真结果生成/更新分析报告 Markdown"""

import os
from datetime import datetime
from typing import Any, Dict, List

from tabulate import tabulate
from evaluation.benchmark import format_results_table
from config import (
    REPORT_DIR, FIGURES_DIR, FS, BANDWIDTH, N_SAMPLES,
    CHANNEL_DETECTION_TOLERANCE_HZ, TIME_DOMAIN_FREQ_RATIO_MAX,
)


def generate_analysis_report(
    single_results: Dict[str, List[Dict]],
    channel_results: List[Dict],
    bench_results: List[Dict],
    channel_bench: List[Dict],
    figure_paths: List[str],
    output_path: str = None,
) -> str:
    """生成分析报告 Markdown 文件"""
    if output_path is None:
        os.makedirs(REPORT_DIR, exist_ok=True)
        output_path = os.path.join(REPORT_DIR, 'analysis_report.md')

    lines = [
        '# 超宽带测频接收机信道化方案设计与仿真 —— 分析报告',
        '',
        f'*自动生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*',
        '',
        '## 一、仿真说明',
        '',
        '本报告由 `main.py` 运行后调用 `report/generate_report.py` 自动生成。',
        '信道化指标为 **粗检测率**（峰值列表是否含真实频率，容差 '
        f'{CHANNEL_DETECTION_TOLERANCE_HZ/1e6:.0f} MHz），**非**测频 RMSE。',
        '',
        '信号模型: 复数序列 `exp(j2πft)` 等效基带仿真（无显式 IQ 下变频链）。',
        f'时域法（相位差分/过零）在 |f|/fs > {TIME_DOMAIN_FREQ_RATIO_MAX} 时标注为仅供参考。',
        '',
        '### 系统参数',
        f'- 采样率: {FS/1e9:.2f} GHz',
        f'- 带宽: {BANDWIDTH/1e9:.2f} GHz',
        f'- 采样点数: {N_SAMPLES}',
        '',
        '## 二、单信道测频方法',
        '',
    ]

    # 窄带表
    lines.append('### 窄带 (SNR=10 dB)')
    rows = []
    for mr in single_results.get('narrowband', []):
        srs = mr['snr_results']
        sr = next((s for s in srs if s['snr_db'] == 10), srs[-1] if srs else None)
        if not sr:
            continue
        rows.append([
            mr['method'],
            f"{sr['rmse_hz']:.2e}",
            f"{sr['mean_time_us']:.1f}",
            f"{sr.get('success_rate_narrow', 0):.1f}%",
        ])
    if rows:
        lines.append(tabulate(
            rows,
            headers=['方法', 'RMSE(Hz)', '耗时(us)', '成功率(<5kHz)'],
            tablefmt='pipe',
        ))
    lines.append('')

    lines.append(format_results_table(single_results.get('medium', []), '中带宽双音'))
    lines.append('')

    # 精细基准
    lines.append('## 三、精细基准 (窄带 SNR=10 dB)')
    rows = [[b['method'], f"{b['freq_est_hz']/1e6:.6f}",
             f"{b['error_khz']:.3f}", f"{b['time_us']:.1f}"] for b in bench_results]
    if rows:
        lines.append(tabulate(
            rows, headers=['方法', '估计(MHz)', '误差(kHz)', '耗时(us)'], tablefmt='pipe'
        ))
    lines.append('')

    lines.append('## 四、信道化方案 (粗检测)')
    rows = []
    for cr in channel_results:
        srs = cr['snr_results']
        sr = next((s for s in srs if s['snr_db'] == 10), srs[0] if srs else None)
        if not sr:
            continue
        rows.append([
            cr['scheme'],
            f"{sr['mean_time_us']:.1f}",
            f"{sr['avg_n_peaks']:.1f}",
            f"{sr['overall_detection_rate']:.1%}",
        ])
    if rows:
        lines.append(tabulate(
            rows, headers=['方案', '耗时(us)', '平均峰值数', '总检测率'], tablefmt='pipe'
        ))
    lines.append('')

    lines.append('## 五、全频段混合信号基准')
    rows = []
    for b in channel_bench:
        det = b.get('detected', [])
        rows.append([
            b['scheme'], b['n_peaks'], f"{b['time_us']:.1f}",
            ' '.join(f'{int(d)}' for d in det) if det else '-',
        ])
    if rows:
        lines.append(tabulate(
            rows, headers=['方案', '峰值数', '耗时(us)', '检测 N M1 M2 W'], tablefmt='pipe'
        ))
    lines.append('')

    if figure_paths:
        lines.append('## 六、可视化')
        for p in figure_paths:
            rel = os.path.relpath(p, REPORT_DIR).replace('\\', '/')
            lines.append(f'![figure]({rel})')
        lines.append('')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return output_path