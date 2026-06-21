"""
multi_lamp.py — 多灯联合定标编排器
职责：对 KR / AR / NM 三灯分别寻峰，合并候选峰后统一 RANSAC 匹配 + 拟合

使用场景：
    某些仪器单独一盏灯提供的定标谱线数量不够（比如只有3~4条），需要
    多盏灯（如 KR / AR / NM）联合提供足够的定标谱线（如11条）才能定标。
    这与具体是哪台仪器、多少通道无关——只要某次定标用了多个灯文件，
    就走这条路径；几通道、典型峰宽多少由 auto_tune_config() 从实际数据
    里测，不在这里假设。

    流程：
    1. 各灯独立寻峰 → 收集全部候选峰质心
    2. 合并候选峰 + 去重
    3. 用合并后的候选峰列表 + 全部 11 条参考波长跑一次 RANSAC
    4. 三次多项式拟合

输入：
    lamp_intensities: {"KR": array, "AR": array, "NM": array}
    每个灯的光谱强度数组（需已翻转，保证波长随像素增加）

输出：
    MultiLampResult:
        - calibration: CalibrationResult
        - ransac_result: RansacMatchResult（联合 RANSAC 结果，
          ransac_result.excluded_centroids 中可查看被预过滤排除的候选峰）
        - all_pairs: [(px, wl), ...]（全部匹配对）

设计变更记录（本版本）：
    寻峰阶段被 peak_finder.py 标记为 SUSPECTED_ARTIFACT（疑似伪峰：窄峰+
    边际信噪比组合）的候选峰，现在会在送入 RANSAC 之前被排除（见
    EXCLUDE_FROM_RANSAC_REASONS），不再和正常候选峰混在一起参与种子抽样
    和内点投票。注意：ransac_kwargs 不要再传 excluded_centroids，本函数
    已经会显式传递，重复传会报 TypeError。
"""

import os
import numpy as np
from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional, Tuple

from .pipeline import run_explorer
from .config import Config
from .peak_finder import FlagReason
from .ransac_matcher import ransac_match_wavelengths, RansacMatchResult
from .calibration import calibrate, CalibrationResult


# 哪些 fail_reasons 标记意味着"不该进入RANSAC候选池"。
# 目前只有 SUSPECTED_ARTIFACT（窄峰+边际信噪比组合，置信度不足）——其余
# 标记（如 BOUNDARY_HIGH_SADDLE 疑似双峰粘连、WIDTH_TOO_WIDE 等）仍然
# 保留在候选池里交给 RANSAC 的鲁棒匹配机制自行判断，不在这里抢先剔除。
# 以后如果需要把更多标记也纳入预过滤，加进这个列表即可，不用改下面的逻辑。
EXCLUDE_FROM_RANSAC_REASONS = [FlagReason.SUSPECTED_ARTIFACT]


@dataclass
class MultiLampResult:
    """多灯联合定标结果。"""
    calibration    : CalibrationResult
    ransac_result  : RansacMatchResult
    all_pairs      : List[Tuple[float, float]] = field(default_factory=list)


def calibrate_multi_lamp(
    lamp_intensities: Dict[str, np.ndarray],
    combined_wavelengths: List[float],
    config          : Optional[Config] = None,
    ransac_kwargs   : Optional[dict] = None,
) -> MultiLampResult:
    """
    多灯联合定标主入口。

    Parameters
    ----------
    lamp_intensities    : {"KR": array, "AR": array, "NM": array}
    combined_wavelengths : 全部参考波长（三灯合并），如 11 条
    config              : 全局 Config，None 则用默认值
    ransac_kwargs       : 传给 ransac_match_wavelengths 的额外参数

    Returns
    -------
    MultiLampResult
    """
    cfg = config or Config()
    rkwargs = ransac_kwargs or {}

    # ── 各自寻峰，合并候选峰（每灯独立日志） ──────────────────────────────
    all_candidates: List[float] = []
    all_excluded  : List[float] = []   # 标记为 SUSPECTED_ARTIFACT 等的候选峰

    for lamp_name, intensity in lamp_intensities.items():
        # 每灯独立日志：从 base log_path 派生，如 "log.txt" → "KR_log.txt"
        base = cfg.log_path
        stem, ext = os.path.splitext(base)
        lamp_log = f"{stem}_{lamp_name}{ext}" if stem else f"{lamp_name}{ext}"
        lamp_cfg = replace(cfg, log_path=lamp_log, quality_csv_path="")

        pipeline_result = run_explorer(intensity, lamp_cfg, save_csv=False)
        lamp_peaks = pipeline_result.peaks   # 完整 PeakResult，带 fail_reasons
        candidates = [p.centroid for p in lamp_peaks]

        if len(candidates) == 0:
            raise RuntimeError(
                f"[{lamp_name}] 灯未找到任何候选峰。"
            )
        all_candidates.extend(candidates)

        lamp_excluded = [
            p.centroid for p in lamp_peaks
            if any(reason in r for r in p.fail_reasons
                   for reason in EXCLUDE_FROM_RANSAC_REASONS)
        ]
        if lamp_excluded:
            print(f"[{lamp_name}] 预过滤排除 {len(lamp_excluded)} 个候选峰"
                  f"（{', '.join(EXCLUDE_FROM_RANSAC_REASONS)}）: "
                  f"{[round(c, 3) for c in lamp_excluded]}")
        all_excluded.extend(lamp_excluded)

    # 去重（保留 3 位小数精度，与下面 unique 的精度一致，避免容差错位）
    unique = sorted(set(round(c, 3) for c in all_candidates))
    excluded_unique = sorted(set(round(c, 3) for c in all_excluded))

    if len(unique) < 3:
        raise RuntimeError(
            f"三灯联合仅找到 {len(unique)} 个不重复候选峰"
            + (f"（其中 {len(excluded_unique)} 个已被预过滤排除）" if excluded_unique else "")
            + f"，不足以进行 RANSAC 匹配。"
        )

    # ── 联合 RANSAC ───────────────────────────────────────────────────────
    min_inl = max(4, int(np.ceil(len(combined_wavelengths) * 0.5)))
    ransac_result = ransac_match_wavelengths(
        candidate_centroids = unique,
        true_wavelengths    = combined_wavelengths,
        min_inliers         = min_inl,
        excluded_centroids  = excluded_unique,
        **rkwargs,
    )

    # ── 多项式拟合 ────────────────────────────────────────────────────────
    pairs = ransac_result.calibration_pairs
    calibration = calibrate(
        centroids   = [p[0] for p in pairs],
        wavelengths = [p[1] for p in pairs],
        degree      = cfg.calibration_degree,
    )

    return MultiLampResult(
        calibration   = calibration,
        ransac_result = ransac_result,
        all_pairs     = pairs,
    )