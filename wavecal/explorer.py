"""
explorer.py — 峰质量探索接口
职责：输出所有有效峰的详细质量报告，辅助用户人工选取定标特征峰
使用场景：
    - 首次建立定标方案时运行，确定特征峰后几乎不再用
    - 换新仪器/新光源时重新评估
    - 调试 peak_finder 参数时检查中间结果
暴露接口：
    peak_quality_report(peaks, intensity, baseline, noise_std)
        -> list[PeakQuality]         详细质量信息列表
    print_quality_table(qualities)   终端打印质量表
    suggest_calibration_peaks(qualities, n, coverage_weight)
        -> list[PeakQuality]         按综合评分推荐 n 个定标峰
    save_quality_csv(qualities, path) 保存为 CSV 供外部查看
"""

import numpy as np
import csv
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .peak_finder import (
    PeakResult,
    extract_peak_window,
    compute_centroid,
    compute_sigma,
    compute_skewness,
)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class PeakQuality:
    """单峰的完整质量档案，供人工选峰和自动评分使用。"""
    # 基本位置
    peak_pixel  : int
    centroid    : float
    left        : int
    right       : int

    # 峰形指标
    half_width  : float     # (right - left) / 2，px
    fwhm_est    : float     # 2.355 × sigma（高斯近似），px
    skewness    : float
    snr         : float
    peak_height : float     # 扣基线后峰顶强度

    # 孤立度（距离最近相邻峰的像素数）
    isolation_left  : float   # 距左侧最近峰的距离（px），无相邻峰则为 inf
    isolation_right : float   # 距右侧最近峰的距离（px），无相邻峰则为 inf
    isolation_min   : float   # 两侧最小值

    # 综合评分（0~100，越高越适合作定标峰）
    quality_score : float = 0.0

    # 质量检测状态（来自 peak_finder 的标记，不影响是否进入此列表）
    passed_all   : bool = True
    fail_reasons : List[str] = field(default_factory=list)


# ── 主入口 ────────────────────────────────────────────────────────────────────

def peak_quality_report(
    peaks      : List[PeakResult],
    I_corr     : np.ndarray,          # 已扣基线的强度数组
    snr_threshold_calibration: float = 50.0,   # 定标峰推荐的更高 SNR 门槛
    skew_threshold_calibration: float = 0.3,   # 定标峰推荐的更严对称性门槛
) -> List[PeakQuality]:
    """
    对 find_peaks() 返回的有效峰列表做深度质量分析。

    Parameters
    ----------
    peaks      : find_peaks() 的输出
    I_corr     : 扣基线后的强度数组（baseline.py 的 intensity - baseline）
    snr_threshold_calibration  : 用于评分，不用于过滤
    skew_threshold_calibration : 用于评分，不用于过滤

    Returns
    -------
    list[PeakQuality]，按综合评分降序排列
    """
    I_corr = np.asarray(I_corr, dtype=float)
    centroids = np.array([p.centroid for p in peaks])

    qualities: List[PeakQuality] = []

    for i, p in enumerate(peaks):
        # FWHM 估计用核心窗口，避免裙边（伴峰残留）让 sigma 虚高失真
        pixels, I_win = extract_peak_window(I_corr, p.core_left, p.core_right)
        I_win_clipped = np.clip(I_win, 0, None)

        # FWHM 估计（高斯近似）
        sigma    = compute_sigma(pixels, I_win_clipped, p.centroid)
        fwhm_est = 2.355 * sigma

        # 孤立度
        iso_left  = float(p.centroid - centroids[i - 1]) if i > 0 \
                    else float("inf")
        iso_right = float(centroids[i + 1] - p.centroid) if i < len(peaks) - 1 \
                    else float("inf")
        iso_min   = min(iso_left, iso_right)

        # 综合评分
        score = _quality_score(
            snr       = p.snr,
            skewness  = p.skewness,
            isolation = iso_min,
            snr_thr   = snr_threshold_calibration,
            skew_thr  = skew_threshold_calibration,
        )

        qualities.append(PeakQuality(
            peak_pixel      = p.peak_pixel,
            centroid        = p.centroid,
            left            = p.left,
            right           = p.right,
            half_width      = p.half_width,
            fwhm_est        = fwhm_est,
            skewness        = p.skewness,
            snr             = p.snr,
            peak_height     = p.peak_height,
            isolation_left  = iso_left,
            isolation_right = iso_right,
            isolation_min   = iso_min,
            quality_score   = score,
            passed_all      = p.passed_all,
            fail_reasons    = list(p.fail_reasons),
        ))

    qualities.sort(key=lambda q: q.quality_score, reverse=True)
    return qualities


def suggest_calibration_peaks(
    qualities       : List[PeakQuality],
    n               : int   = 8,
    coverage_weight : float = 0.4,
    detector_size   : int   = 2048,
) -> List[PeakQuality]:
    """
    从质量列表中推荐 n 个定标特征峰，兼顾质量评分和空间覆盖均匀性。

    Parameters
    ----------
    qualities       : peak_quality_report() 的输出
    n               : 期望推荐数量
    coverage_weight : 覆盖均匀性在最终评分中的权重（0=纯质量，1=纯覆盖）
    detector_size   : 探测器像素数，用于归一化覆盖评分

    Returns
    -------
    推荐峰列表，按像素位置升序排列
    """
    if len(qualities) <= n:
        result = sorted(qualities, key=lambda q: q.centroid)
        return result

    # 贪心选取：每次选"质量评分 + 覆盖贡献"最高的点
    selected: List[PeakQuality] = []
    remaining = list(qualities)

    # 先选评分最高的一个作为起点
    selected.append(remaining.pop(0))

    while len(selected) < n and remaining:
        best_idx  = -1
        best_total = -1.0

        for idx, q in enumerate(remaining):
            # 覆盖贡献：距已选点最近距离，越大越好
            min_dist = min(abs(q.centroid - s.centroid) for s in selected)
            cov_score = min_dist / detector_size  # 归一化到 [0,1]

            total = (1 - coverage_weight) * (q.quality_score / 100.0) \
                  + coverage_weight * cov_score

            if total > best_total:
                best_total = total
                best_idx   = idx

        selected.append(remaining.pop(best_idx))

    selected.sort(key=lambda q: q.centroid)
    return selected


# ── 报告输出 ──────────────────────────────────────────────────────────────────

def print_quality_table(
    qualities   : List[PeakQuality],
    highlight   : Optional[List[PeakQuality]] = None,
) -> None:
    """
    终端打印质量表。highlight 中的峰会标注 ★（推荐定标峰）。

    列说明：
        centroid  质心像素位置
        hw        半宽（px）
        fwhm      高斯近似FWHM（px）
        skew      三阶矩对称性
        SNR       信噪比
        iso       最近相邻峰距离（px）
        score     综合评分
        通过      是/否，是否通过全部质量检测（所有峰均保留在列表中，
                  此列仅供人工参考，不代表该峰已被排除）
        flag      ★ = 推荐定标峰
    """
    highlight_centroids = set()
    if highlight:
        highlight_centroids = {round(q.centroid, 3) for q in highlight}

    sep  = "-" * 92
    hdr  = (f"  {'centroid':>10}  {'hw':>5}  {'fwhm':>6}  "
            f"{'skew':>7}  {'SNR':>8}  {'iso':>6}  {'score':>6}  {'通过':>4}  flag")

    print(sep)
    print("  峰质量报告（按综合评分降序，全部候选峰均列出，不做剔除）")
    print(sep)
    print(hdr)
    print(sep)

    for q in qualities:
        flag = "★" if round(q.centroid, 3) in highlight_centroids else ""
        iso_str = f"{q.isolation_min:.0f}" if q.isolation_min != float("inf") \
                  else "∞"
        status = "是" if q.passed_all else "否"
        print(
            f"  {q.centroid:10.3f}  {q.half_width:5.1f}  {q.fwhm_est:6.2f}  "
            f"{q.skewness:+7.4f}  {q.snr:8.1f}  {iso_str:>6}  "
            f"{q.quality_score:6.1f}  {status:>4}  {flag}"
        )
        if not q.passed_all and q.fail_reasons:
            print(f"      └─ {'; '.join(q.fail_reasons)}")
    print(sep)
    n_passed = sum(1 for q in qualities if q.passed_all)
    print(f"  共 {len(qualities)} 个候选峰（完全通过 {n_passed} 个，存在标记 "
          f"{len(qualities) - n_passed} 个）")
    print(sep)


def save_quality_csv(
    qualities : List[PeakQuality],
    path      : str = "peak_quality.csv",
) -> None:
    """
    保存质量报告为 CSV，方便在 Excel / Python 中进一步筛选。
    新增 passed_all / fail_reasons 两列，标记该峰是否通过全部质量检测
    及具体原因（所有峰均保留在表中，此标记仅供人工参考）。
    """
    fields = [
        "peak_pixel", "centroid", "left", "right",
        "half_width", "fwhm_est", "skewness", "snr", "peak_height",
        "isolation_left", "isolation_right", "isolation_min", "quality_score",
        "passed_all", "fail_reasons",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for q in qualities:
            row = {k: getattr(q, k) for k in fields}
            for k in ("isolation_left", "isolation_right", "isolation_min"):
                if row[k] == float("inf"):
                    row[k] = ""
            row["fail_reasons"] = "; ".join(row["fail_reasons"]) if row["fail_reasons"] else ""
            writer.writerow(row)
    print(f"[explorer] 质量报告已保存: {path}")


# ── 内部评分函数 ──────────────────────────────────────────────────────────────

def _quality_score(
    snr       : float,
    skewness  : float,
    isolation : float,
    snr_thr   : float = 50.0,
    skew_thr  : float = 0.3,
) -> float:
    """
    综合质量评分，满分 100。
    三个维度加权：
        SNR 分    (40%)：log 压缩，SNR≥snr_thr 得满分
        对称性分  (40%)：|skew| 越小越好，超过 skew_thr 线性扣分
        孤立度分  (20%)：距最近邻峰 ≥ 50px 得满分，线性衰减

    这套权重参考了 ESO HARPS pipeline 定标峰筛选的思路：
    SNR 和对称性同等重要，孤立度作为辅助条件。
    """
    # SNR 分（40分）：log 尺度，SNR=snr_thr 时得 40 分
    snr_score = min(40.0, 40.0 * np.log1p(snr) / np.log1p(snr_thr))

    # 对称性分（40分）：|skew|=0 得 40 分，|skew|≥skew_thr 得 0 分
    sym_score = max(0.0, 40.0 * (1.0 - abs(skewness) / skew_thr))

    # 孤立度分（20分）：isolation ≥ 50px 得满分
    iso_ref   = 50.0
    iso_score = min(20.0, 20.0 * min(isolation, iso_ref) / iso_ref)

    return round(snr_score + sym_score + iso_score, 2)