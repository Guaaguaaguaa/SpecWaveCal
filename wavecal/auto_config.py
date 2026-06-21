"""
auto_config.py — 仪器自适应参数推导
职责：从实际光谱数据（而不是仪器规格/通道数/采样间隔）里测出"这台仪器的峰
      实际有多宽"，据此自动推导 Config 里跟像素尺度相关的寻峰窗口参数。

为什么不用通道数/采样间隔去反推：
    通道数、采样间隔只能告诉你"理论上峰大概有多少像素"，实际还受光学
    PSF展宽、狭缝宽度、探测器响应等因素影响，人工换算容易引入误差，
    而且每接入一台新仪器都要重新算一遍、重新判断该用哪一套参数——这正是
    项目这几轮一直在踩的坑（min_peak_sep/max_half_width 等按旧仪器经验
    写死，换仪器就失效）。

设计思路（自举式两遍扫描）：
    第一遍：用几乎不设限的宽松参数跑一次 find_peaks，只是为了"看看这台
            仪器的峰长什么样"，不追求干净结果。
    筛选：  从结果里挑出明显可信的峰（SNR远高于阈值、边界自然收敛、
            无双峰粘连/不对称/邻峰过近等任何标记）——这些大概率是真实、
            孤立、形态正常的谱线。
    统计：  取这批峰 half_width 的中位数，作为这台仪器的"典型峰宽"。
    推导：  典型峰宽反推出 min_half_width / max_half_width / min_peak_sep /
            baseline_window_size / snr_scale_reference_width，返回一份
            新的 Config。

    找不到足够数量的"明显可信"峰时直接报错（硬熔断），不做猜测性兜底——
    宁可让人工介入检查数据质量，也不要悄悄返回一份可能不靠谱的 Config。

暴露接口：
    auto_tune_config(intensity_or_lamps, base_config=None) -> Config
"""

import os
import numpy as np
from dataclasses import replace
from typing import Dict, List, Optional, Union

from .config import Config
from .logger import Logger
from .baseline import estimate_baseline, estimate_local_noise
from .peak_finder import find_peaks, PeakResult


class AutoTuneError(Exception):
    """自动调参失败（明显可信的峰数量不足，无法可靠估计典型峰宽）。"""
    pass


def _explore_log_path(base_log_path: str, name: str) -> str:
    """
    派生探索阶段专用日志路径，和 multi_lamp.py 里"stem_LAMPNAME.ext"的
    命名方式保持一致风格，不直接传空字符串给 Logger（没有 logger.py 源码，
    不确定它对空路径的处理方式，稳妥起见沿用已验证可用的命名模式）。
    """
    base = base_log_path or "log.txt"
    stem, ext = os.path.splitext(base)
    ext = ext or ".txt"
    suffix = f"_autotune_explore_{name}" if name else "_autotune_explore"
    return f"{stem}{suffix}{ext}"


def _explore_clean_peaks(
    intensity        : np.ndarray,
    base_config      : Config,
    high_confidence_snr_multiplier: float,
    name             : str = "",
) -> List[PeakResult]:
    """对单条光谱做一次宽松探索，返回其中"明显可信"的峰列表。"""
    intensity = np.asarray(intensity, dtype=float)

    baseline, noise_std = estimate_baseline(
        intensity,
        window_size = base_config.baseline_window_size,
        percentile  = base_config.baseline_percentile,
    )
    local_noise = estimate_local_noise(
        intensity, baseline, window_size=base_config.baseline_window_size,
    )

    log_path = _explore_log_path(base_config.log_path, name)
    with Logger(log_path) as log:
        peaks = find_peaks(
            intensity            = intensity,
            baseline              = baseline,
            noise_std             = noise_std,
            logger                 = log,
            local_noise            = local_noise,
            min_snr                = base_config.min_snr,
            # 探索阶段故意几乎不设限——目的是看清这台仪器的峰本来长什么样，
            # 不是要在这一步就拿到生产可用的候选峰列表
            min_peak_sep            = 1,
            boundary_n_sigma        = base_config.boundary_n_sigma,
            boundary_consec         = base_config.boundary_consec,
            saddle_n_sigma           = base_config.saddle_n_sigma,
            min_half_width           = 1,
            max_half_width           = max(20, len(intensity) // 4),
            snr_scale_reference_width = base_config.snr_scale_reference_width,
            skewness_threshold        = base_config.skewness_threshold,
            core_height_ratio         = base_config.core_height_ratio,
        )

    snr_floor = high_confidence_snr_multiplier * base_config.min_snr
    clean = [p for p in peaks if p.passed_all and p.snr >= snr_floor]
    return clean


def auto_tune_config(
    intensity_or_lamps : Union[np.ndarray, Dict[str, np.ndarray]],
    base_config         : Optional[Config] = None,
    min_clean_peaks      : int   = 5,
    high_confidence_snr_multiplier: float = 3.0,
) -> Config:
    """
    从实际数据自动推导一份适配该仪器的 Config。

    Parameters
    ----------
    intensity_or_lamps : 单条光谱强度数组，或多灯联合场景下的
                          {"KR": array, "AR": array, "NM": array} 字典——
                          多灯场景会把各灯找到的"明显可信"峰汇总到一起统计，
                          数据点更多，典型峰宽估计更稳健（多灯本来就是
                          同一台仪器拍的，色散特性应该一致）。
    base_config         : 提供其余不需要自动推导的参数（min_snr、
                          skewness_threshold、calibration_degree 等）的
                          基准 Config，None 则用默认值。返回结果在此基础上
                          覆盖 detector_size / min_half_width / max_half_width /
                          min_peak_sep / baseline_window_size /
                          snr_scale_reference_width 这几项。
    min_clean_peaks     : 至少需要这么多个"明显可信"的峰才能可靠估计典型
                          峰宽，不够则报 AutoTuneError（硬熔断，不做猜测性
                          兜底）
    high_confidence_snr_multiplier: "明显可信"的SNR门槛 = 此值 × min_snr

    Returns
    -------
    Config：detector_size 取实际数据长度；寻峰窗口相关参数按测得的典型
            峰宽自动推导；其余参数沿用 base_config。

    Raises
    ------
    AutoTuneError : 明显可信的峰数量不足，无法可靠估计
    """
    base_config = base_config or Config()

    if isinstance(intensity_or_lamps, dict):
        lamp_items = list(intensity_or_lamps.items())
        detector_size = len(next(iter(intensity_or_lamps.values())))
    else:
        lamp_items = [("", intensity_or_lamps)]
        detector_size = len(intensity_or_lamps)

    all_clean: List[PeakResult] = []
    per_lamp_counts = {}
    for name, intensity in lamp_items:
        clean = _explore_clean_peaks(intensity, base_config, high_confidence_snr_multiplier, name=name)
        per_lamp_counts[name or "(单灯)"] = len(clean)
        all_clean.extend(clean)

    if len(all_clean) < min_clean_peaks:
        raise AutoTuneError(
            f"自动调参失败：明显可信的峰只有 {len(all_clean)} 个"
            f"（各来源: {per_lamp_counts}），少于要求的 {min_clean_peaks} 个，"
            f"无法可靠估计这台仪器的典型峰宽。请检查数据质量，或手动指定 Config。"
        )

    hw_typical = float(np.median([p.half_width for p in all_clean]))

    min_half_width = max(1, round(hw_typical * 0.4))
    max_half_width = max(min_half_width + 2, round(hw_typical * 4))
    min_peak_sep   = max(min_half_width, round(hw_typical * 1.5))
    baseline_window_size = max(20, round(max_half_width * 4))
    snr_scale_reference_width = round(max(hw_typical, min_half_width), 2)

    print(
        f"[auto_tune_config] 典型峰宽(中位数) hw_typical={hw_typical:.2f}px"
        f"（基于 {len(all_clean)} 个明显可信的峰，{per_lamp_counts}）→ "
        f"min_half_width={min_half_width}  max_half_width={max_half_width}  "
        f"min_peak_sep={min_peak_sep}  baseline_window_size={baseline_window_size}  "
        f"snr_scale_reference_width={snr_scale_reference_width}"
    )

    return replace(
        base_config,
        detector_size = detector_size,
        min_half_width = min_half_width,
        max_half_width = max_half_width,
        min_peak_sep    = min_peak_sep,
        baseline_window_size = baseline_window_size,
        snr_scale_reference_width = snr_scale_reference_width,
    )