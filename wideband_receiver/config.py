"""全局配置参数"""

# 系统参数
FS = 4e9  # 采样率 4 GHz (满足2GHz带宽奈奎斯特采样)
BANDWIDTH = 2e9  # 总带宽 2 GHz
N_FFT_CHANNEL = 4096  # 单信道FFT点数(零填充时可大于 N_SAMPLES)
MASK_ALPHA = 0.3  # 信道检测门限系数

# 信号参数
NARROWBAND_FREQ = 120.123e6  # 窄带信号频率
NARROWBAND_ACCURACY = 5e3  # 窄带测频精度要求 < 5kHz

MEDIUM_FREQ_1 = 500.456e6  # 中带宽信号1
MEDIUM_FREQ_2 = 520.789e6  # 中带宽信号2
MEDIUM_SPACING = 20e6  # 双信号间隔 20MHz
MEDIUM_RESOLVE_SPACING = 20e6  # 双音可分辨间隔要求

WIDEBAND_FREQ = 1480.321e6  # 宽带稀疏信号
WIDEBAND_ACCURACY = 100e3  # 宽带测频精度要求 < 100kHz

# 相位差分/过零等时域法在 f/fs 较大时不可靠的阈值
TIME_DOMAIN_FREQ_RATIO_MAX = 0.35  # |f|/fs 超过此值则标注不适用

# 仿真参数
SNR_DB_VALUES = [0, 5, 10, 20]  # 信噪比范围
N_MONTE_CARLO = 15  # 蒙特卡洛次数
N_SAMPLES = 2048  # 采样点数
N_MUSIC_SOURCES = 3  # MUSIC算法信号源数
FREQ_SEARCH_GRANULARITY = 1000  # 频率搜索粒度(Hz)，MUSIC 局部搜索步长
N_MC_SINGLE = 15  # 单信道方法蒙特卡洛次数
N_MC_CHANNEL = 5  # 信道化方案蒙特卡洛次数

# 信道化粗检测: 是否出现在峰值列表(非测频 RMSE)
CHANNEL_DETECTION_TOLERANCE_HZ = 10e6

# 信道化参数
N_CHANNELS = 16  # 均匀信道化信道数
POLYPHASE_DECIM = 16  # 多相抽取因子
POLYPHASE_TAPS = 128  # 多相滤波器抽头数
NONUNIFORM_THRESHOLD = 0.3  # 非均匀信道化能量检测门限
NONUNIFORM_CHANNEL_BW = 125e6  # 非均匀信道化最小信道带宽

# 结果与报告
RESULTS_DIR = 'results'
REPORT_DIR = 'report'
FIGURES_DIR = 'results/figures'