"""
baseline.py — 基线与噪声估计
职责：从原始光谱中估计逐像素基线和全局噪声标准差
暴露接口：
    estimate_baseline(intensity, config) -> (baseline, noise_std)
        intensity  : np.ndarray, shape (N,)
        config     : Config 对象（只读取基线相关字段）
        返回：
            baseline  : np.ndarray shape (N,)，逐像素基线
            noise_std : float，噪声标准差（用于 SNR 和边界判定）

设计说明：
    基线平坦假设下，采用滚动百分位数法：
    - 将光谱分成若干小窗口，每窗口取低百分位数作为局部基线估计
    - 再用线性插值还原到逐像素分辨率
    - noise_std 用无峰区域（低于基线 + 粗阈值）的像素估计
"""

import numpy as np
from typing import Tuple


def estimate_baseline(
    intensity: np.ndarray,
    window_size: int = 100,
    percentile: float = 10.0,
) -> Tuple[np.ndarray, float]:
    """
    滚动百分位数基线估计。

    Parameters
    ----------
    intensity    : 原始强度数组，shape (N,)
    window_size  : 滚动窗口宽度（像素），默认 100
                   建议 >> 最宽峰宽，<< 探测器总长
    percentile   : 取窗口内第几百分位数作为基线，默认 10
                   基线平坦时 10 已足够；有缓变背景时可适当提高到 20

    Returns
    -------
    baseline  : np.ndarray shape (N,)，逐像素基线
    noise_std : float，无峰区域的噪声标准差

    Notes
    -----
    - window_size 应远大于单峰宽度，否则峰脚会压低百分位数估计
    - 对于 2048 px 探测器，window_size=100 通常是合理起点
    """
    intensity = np.asarray(intensity, dtype=float)
    n = len(intensity)

    # ── Step 1：滚动窗口百分位数，得到稀疏控制点 ────────────────────────────
    half = window_size // 2
    centers = np.arange(half, n - half, half)  # 每半窗步进一个控制点

    # 边界补充首尾控制点
    if centers[0] != 0:
        centers = np.concatenate([[0], centers])
    if centers[-1] != n - 1:
        centers = np.concatenate([centers, [n - 1]])

    baseline_ctrl = np.array([
        np.percentile(intensity[max(0, c - half): min(n, c + half)], percentile)
        for c in centers
    ])

    # ── Step 2：线性插值还原到逐像素 ─────────────────────────────────────────
    baseline = np.interp(np.arange(n), centers, baseline_ctrl)

    # ── Step 3：估计噪声标准差（只用"无峰"区域）────────────────────────────
    # 粗判：强度低于 baseline + 3×粗噪声 的区域认为是纯背景
    # 先用全局 MAD 做粗噪声估计，再精化
    residual = intensity - baseline
    mad = np.median(np.abs(residual - np.median(residual)))
    rough_noise = mad * 1.4826  # MAD → sigma 换算系数（高斯假设）

    background_mask = residual < 3.0 * rough_noise
    if background_mask.sum() < 10:
        # 极端情况：几乎全是峰，退回全局 MAD
        noise_std = rough_noise
    else:
        noise_std = residual[background_mask].std()
        if noise_std < 1e-6:
            noise_std = rough_noise  # 防止除零

    return baseline, float(noise_std)


def estimate_local_noise(
    intensity   : np.ndarray,
    baseline    : np.ndarray,
    window_size : int = 100,
    n_sigma_mask: float = 3.0,
) -> np.ndarray:
    """
    逐像素局部噪声标准差估计，供边界判定使用。

    背景：全局 noise_std 假设整张光谱的背景噪声水平一致，但实际光谱中
    不同区域的背景可能因杂散光、相邻强峰远翼等原因有结构性差异（不是
    纯随机噪声，而是局部背景偏高或有轻微波动）。用同一个全局阈值判定
    所有峰的边界，会让背景较"脏"区域的峰更难收敛。

    做法：
    在每个像素位置，取以它为中心的局部窗口，先用 residual=intensity-baseline
    剔除疑似峰区域（residual 远高于局部粗估计噪声的点），只用纯背景点算
    局部标准差。窗口滑动，逐像素输出局部噪声值。

    Parameters
    ----------
    intensity    : 原始强度数组
    baseline     : estimate_baseline() 返回的逐像素基线
    window_size  : 局部窗口宽度（像素），默认与基线窗口一致，100
    n_sigma_mask : 剔除疑似峰区域的阈值（× 全局粗噪声）

    Returns
    -------
    local_noise : np.ndarray shape (N,)，逐像素局部噪声标准差
                  噪声水平上升的区域（如强峰附近的脏背景）该值会更大
    """
    intensity = np.asarray(intensity, dtype=float)
    baseline  = np.asarray(baseline,  dtype=float)
    n = len(intensity)
    residual = intensity - baseline

    # 全局粗噪声，用于初步区分"峰" vs "背景"
    mad = np.median(np.abs(residual - np.median(residual)))
    rough_noise = mad * 1.4826
    if rough_noise < 1e-6:
        rough_noise = max(np.std(residual), 1e-6)

    half = window_size // 2
    local_noise = np.empty(n, dtype=float)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        seg = residual[lo:hi]

        # 剔除疑似峰区域，只用纯背景点估计局部噪声
        bg_mask = seg < n_sigma_mask * rough_noise
        if bg_mask.sum() >= 10:
            local_noise[i] = seg[bg_mask].std()
        else:
            local_noise[i] = rough_noise  # 窗口内几乎全是峰，退回全局粗估计

        if local_noise[i] < 1e-6:
            local_noise[i] = rough_noise

    return local_noise


def snr(intensity: np.ndarray, baseline: np.ndarray, noise_std: float) -> np.ndarray:
    """
    逐像素 SNR，供 peak_finder 使用。
    SNR[i] = (intensity[i] - baseline[i]) / noise_std
    """
    return (np.asarray(intensity, dtype=float) - baseline) / noise_std