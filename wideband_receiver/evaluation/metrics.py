"""评估指标 —— RMSE、计算耗时等"""

import numpy as np
import time


def rmse(estimates: np.ndarray, true_value: float) -> float:
    """均方根误差

    Args:
        estimates: 多次估计的频率值数组
        true_value: 真实频率值

    Returns:
        RMSE (Hz)
    """
    return np.sqrt(np.mean((estimates - true_value) ** 2))


def mean_absolute_error(estimates: np.ndarray, true_value: float) -> float:
    """平均绝对误差"""
    return np.mean(np.abs(estimates - true_value))


def bias(estimates: np.ndarray, true_value: float) -> float:
    """偏差(系统误差)"""
    return np.mean(estimates) - true_value


def std_dev(estimates: np.ndarray) -> float:
    """标准差(随机误差)"""
    return np.std(estimates)


def success_rate(estimates: np.ndarray, true_value: float,
                 tolerance: float) -> float:
    """成功率: 估计误差在tolerance内的比例"""
    errors = np.abs(estimates - true_value)
    return np.mean(errors < tolerance) * 100


def benchmark_function(func, *args, n_runs: int = 10, **kwargs) -> dict:
    """对函数进行性能基准测试

    Args:
        func: 被测函数
        *args, **kwargs: 函数参数
        n_runs: 运行次数

    Returns:
        dict: 时间统计
    """
    # 预热
    _ = func(*args, **kwargs)

    times = []
    results = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        results.append(result)

    times = np.array(times) * 1e6  # 转换为微秒

    return {
        'mean_us': np.mean(times),
        'std_us': np.std(times),
        'min_us': np.min(times),
        'max_us': np.max(times),
        'p50_us': np.percentile(times, 50),
        'p95_us': np.percentile(times, 95),
        'last_result': results[-1]
    }


def frequency_resolution(fs: float, n_fft: int) -> float:
    """FFT频率分辨率"""
    return fs / n_fft
