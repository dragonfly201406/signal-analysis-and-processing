# 超宽带测频接收机信道化方案设计与仿真

## 项目概述
本项目对超宽带测频接收机的四种信道化方案进行仿真与对比分析，覆盖 2GHz 带宽，包含窄带、中等带宽和宽带空闲三类信号。

**仿真模型说明**：测试信号为复数序列 `exp(j2πft)` 的等效基带表示（未建模显式 IQ 下变频与射频前端）。FFT 类方法在正频率段搜索；相位差分/过零检测在 `|f|/fs` 较大时不可靠，评估结果中会标注。

## 信号模型
| 信号类型 | 频率 | 精度要求 |
|---------|------|---------|
| 窄带信号 | 120.123 MHz | < 5 kHz |
| 中等带宽信号 | 500.456 MHz, 520.789 MHz | 分辨 20MHz 间隔 |
| 宽带空闲信号 | 1480.321 MHz | < 100 kHz |

## 四种信道化方案
1. **单信道 FFT 方案** — 整带宽单次 FFT（默认 `N_FFT_CHANNEL` 点零填充）
2. **直接均匀信道化** — 均匀分割子信道，`filtfilt` 抗混叠后独立 FFT
3. **多相滤波信道化** — 简化 PFB 演示实现
4. **非均匀信道化** — 粗 FFT（`fftshift` 全频）感知后自适应分区

信道化评估指标为 **粗检测率**（峰值是否落在真实频率 ±`CHANNEL_DETECTION_TOLERANCE_HZ`），不是测频 RMSE。

## 八种单信道测频算法
1. **基础 FFT 测频** — 频谱峰值插值  
2. **Rife 改进 FFT**  
3. **Quinn 改进 FFT**  
4. **Rife-Quinn 综合**  
5. **双段 FFT 相位差法**  
6. **相位差分法**（高频慎用）  
7. **过零检测法**（高频慎用）  
8. **MUSIC 超分辨测频** — 多峰；中带宽场景评估双峰检出率  

## 使用方式
```bash
pip install -r requirements.txt
python main.py
```

运行后将生成：
- `results/simulation_results.json` — 完整仿真数据  
- `results/simulation_summary.json` — 配置摘要  
- `results/figures/*.png` — RMSE、检测率、频谱示例图  
- `report/analysis_report.md` — 自动更新的分析报告  

### 单元测试
```bash
pip install pytest
pytest tests/ -q
```

## 目录结构
```
wideband_receiver/
├── config.py
├── main.py
├── utils/
│   └── signal_generator.py
├── channelization/
├── single_channel_methods/
├── evaluation/
│   ├── metrics.py
│   ├── benchmark.py
│   ├── helpers.py
│   └── visualization.py
├── report/
│   ├── generate_report.py
│   └── analysis_report.md
├── tests/
└── results/
```