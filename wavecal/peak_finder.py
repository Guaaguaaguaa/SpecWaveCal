"""
peak_finder.py — 寻峰、边界分析、对称性检验、质心定位
职责：从基线修正后的光谱中提取所有候选峰，返回结构化结果

设计原则（重要变更）：
    本模块不再自动剔除任何候选峰。所有质量检测（SNR、宽度、边界收敛质量、
    对称性、峰间距）都只打标签、记录具体原因，不影响该峰是否进入返回列表。
    每个候选峰都会被计算质心并返回，是否采用交由人工依据
    PeakResult.passed_all 和 PeakResult.fail_reasons 自行判断。

    原因：实际数据质量参差不齐，自动硬性剔除在生产中容易因为单一阈值
    不合理而连锁影响后续定标流程（误伤本可以用的峰，或者掩盖了
    "仪器分辨率不足导致双峰粘连"这类需要人工介入的真实问题）。

设计变更记录（本版本）：
    1. Step 1 粗寻峰不再用 scipy 的 distance=min_peak_sep 做硬性剔除——
       旧版若两峰间距小于 min_peak_sep，矮的一个会直接消失，不进入返回
       列表，日志也无痕迹。这在采样间隔粗的仪器上（真实谱线像素间距被
       压缩）会系统性吞掉真实强峰。现在改为：取全部局部极大值，min_peak_sep
       范围内是否存在更高邻居只作为 FlagReason.NEAR_TALLER_NEIGHBOR 标记。
    2. 新增窄峰自适应SNR判断：半宽小于 snr_scale_reference_width 时，
       要求的 SNR 按比例提高（峰越窄，可信所需的SNR越高），同时满足
       "窄于min_half_width 且 SNR也不达标"才标记为
       FlagReason.SUSPECTED_ARTIFACT（疑似伪峰），单纯窄但SNR足够高的
       真实窄峰不会被误标。snr_scale_reference_width 与 min_half_width
       是两个独立参数，不互相联动。

暴露接口：
    find_peaks(intensity, baseline, noise_std, local_noise, config, logger)
        -> list[PeakResult]

每个 PeakResult 包含：
    peak_pixel    : float — 粗定位峰顶（整数像素）
    centroid      : float — 质心亚像素位置（核心窗口计算，不受裙边影响）
    left          : int   — 完整窗口左边界（含裙边，供拟合/导出使用）
    right         : int   — 完整窗口右边界（含裙边，供拟合/导出使用）
    core_left     : int   — 核心窗口左边界（用于质心/对称性计算）
    core_right    : int   — 核心窗口右边界（用于质心/对称性计算）
    half_width    : float — 完整窗口半宽（像素）
    snr           : float — 峰顶 SNR
    skewness      : float — 归一化三阶矩（核心窗口内计算）
    peak_height   : float — 扣基线后峰顶强度
    passed_all    : bool  — 是否通过全部质量检测
    fail_reasons  : list[str] — 未通过项的具体原因（可能为空、可能多条）
    boundary_quality: str — 边界收敛质量描述（"自然收敛"/"强制截断"/"鞍点过高(疑似双峰粘连)"）

设计说明（双层窗口，沿用之前版本）：
    完整窗口 (left, right)：边界拓展的完整结果，含裙边
    核心窗口 (core_left, core_right)：峰高某比例以上的区域，排除裙边干扰，
                                       质心和 skewness 只在此窗口内计算

设计说明（边界强制截断，本次新增）：
    边界搜索不再因"谷底过高"或"未自然收敛"而放弃这个峰。规则：
    1. 强度持续下降，遇到真正的谷底（下降转上升的拐点）→ 用谷底作为边界，
       同时记录谷底相对阈值的高低（决定 boundary_quality 标记）
    2. 一路搜索到 max_half_width 仍未出现拐点（持续下降或持续上升不降）
       → 强制截断在 max_half_width 处，标记为"强制截断"

拓展接口（供 explorer.py / 分辨率模块使用）：
    extract_peak_window(I_corr, left, right) -> (pixels, I_corr_window)
    compute_centroid(pixels, I_win) -> float
    compute_skewness(pixels, I_win, centroid, sigma) -> float
    compute_sigma(pixels, I_win, centroid) -> float
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.signal import find_peaks as scipy_find_peaks

from .logger import Logger


# ── 失败原因常量（与 logger.RejectReason 风格一致，但语义改为"标记"而非"舍弃"）──

class FlagReason:
    LOW_SNR            = "SNR不足"
    WIDTH_TOO_NARROW   = "半宽过窄（可能为噪声尖刺）"
    WIDTH_TOO_WIDE     = "半宽过宽（边界异常）"
    ASYMMETRIC         = "峰形不对称（|skewness| 超过阈值）"
    NEGATIVE_WING      = "扣基线后出现大范围负值（基线估计异常）"
    BOUNDARY_FORCED    = "边界未自然收敛，已强制截断"
    BOUNDARY_HIGH_SADDLE = "边界谷底明显高于基线（疑似双峰粘连/仪器分辨率不足）"
    NEAR_TALLER_NEIGHBOR = "min_peak_sep范围内存在更高峰（原方案会被distance直接剔除，现仅标记）"
    SUSPECTED_ARTIFACT   = "疑似伪峰（窄峰+边际信噪比组合，置信度不足）"


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class PeakResult:
    peak_pixel      : int
    centroid        : float
    left            : int
    right           : int
    core_left       : int
    core_right      : int
    half_width      : float
    snr             : float
    skewness        : float
    peak_height     : float
    passed_all      : bool = True
    fail_reasons    : List[str] = field(default_factory=list)
    boundary_quality: str = "自然收敛"


# ── 主入口 ────────────────────────────────────────────────────────────────────

def find_peaks(
    intensity   : np.ndarray,
    baseline    : np.ndarray,
    noise_std   : float,
    logger      : Logger,
    local_noise : Optional[np.ndarray] = None,
    # ── 基线/SNR 参数 ──────────────────────
    min_snr         : float = 5,
    # ── 峰间距 ────────────────────────────
    min_peak_sep    : int   = 10,
    # ── 边界收敛判定 ──────────────────────
    boundary_n_sigma: float = 3.0,
    boundary_consec : int   = 3,
    saddle_n_sigma  : float = 1.5,
    # ── 硬限制（仅用于标记，不剔除）─────────
    min_half_width  : int   = 3,
    max_half_width  : int   = 40,
    # ── 窄峰自适应SNR（与min_half_width解耦：min_half_width是"过窄"的硬性
    #    标记线，可随仪器调低；snr_scale_reference_width是"足够宽到无需
    #    额外信噪比加成"的参考宽度，默认沿用旧版min_half_width=3的物理含义，
    #    不随min_half_width联动，避免仪器调窄后连带放松了伪峰判定）────────
    snr_scale_reference_width: float = 3.0,
    # ── 对称性 ────────────────────────────
    skewness_threshold: float = 0.5,
    # ── 核心窗口 ──────────────────────────
    core_height_ratio  : float = 0.05,
) -> List[PeakResult]:
    """
    完整寻峰流程：粗定位 → 边界拓展（不剔除）→ 质量打标 → 质心计算

    所有候选峰都会出现在返回列表中。每个峰是否满足各项质量标准记录在
    passed_all / fail_reasons / boundary_quality 字段中，由调用方
    （人工或下游流程）自行决定是否采用。

    Parameters
    ----------
    intensity, baseline, noise_std : 来自 baseline.estimate_baseline()
    local_noise                    : 来自 baseline.estimate_local_noise()
                                      为 None 时退化为全局 noise_std
    logger                          : Logger 实例，记录每个峰的完整检测过程
    其余参数均有默认值，可通过 config 覆盖

    Returns
    -------
    list[PeakResult]，按 centroid 升序排列，包含全部候选峰（不剔除）
    """
    intensity = np.asarray(intensity, dtype=float)
    baseline  = np.asarray(baseline,  dtype=float)
    n = len(intensity)

    if local_noise is None:
        local_noise = np.full(n, noise_std, dtype=float)
    else:
        local_noise = np.asarray(local_noise, dtype=float)

    I_corr = intensity - baseline
    boundary_thresh_arr = boundary_n_sigma * local_noise

    # ── Step 1：粗寻峰（不再用 distance 做硬性剔除，只取局部极大值）──────────
    #    旧版用 scipy 的 distance=min_peak_sep 参数：两个局部极大值间距小于
    #    min_peak_sep 时，矮的一个会被直接丢弃且不留任何痕迹（不进入返回
    #    列表，日志里也查无此峰）。这与本模块"不自动剔除候选峰"的设计原则
    #    矛盾，且在采样间隔较粗的仪器上（真实谱线间距压缩到 min_peak_sep
    #    像素以内）会系统性吞掉真实强峰。现在改为：min_peak_sep 范围内
    #    是否存在更高的邻居，只作为标记（NEAR_TALLER_NEIGHBOR），峰本身
    #    依然保留在返回列表中。
    min_height = min_snr * noise_std
    raw_peaks, _ = scipy_find_peaks(
        I_corr,
        height=min_height,
    )

    logger.info(f"粗寻峰完成（不做间距强制剔除），候选峰数量: {len(raw_peaks)}")

    # 向量化计算每个候选峰 min_peak_sep 像素范围内是否存在更高的邻居
    peak_heights_arr = I_corr[raw_peaks]
    if len(raw_peaks) > 0:
        pixel_diff  = np.abs(raw_peaks[:, None] - raw_peaks[None, :])
        nearby_mask = (pixel_diff > 0) & (pixel_diff <= min_peak_sep)
        taller_nearby = np.array([
            nearby_mask[i].any() and peak_heights_arr[nearby_mask[i]].max() > peak_heights_arr[i]
            for i in range(len(raw_peaks))
        ])
    else:
        taller_nearby = np.array([], dtype=bool)

    results: List[PeakResult] = []

    for idx, p in enumerate(raw_peaks):
        fail_reasons: List[str] = []
        peak_height = float(I_corr[p])
        peak_snr    = peak_height / noise_std

        if taller_nearby[idx]:
            fail_reasons.append(FlagReason.NEAR_TALLER_NEIGHBOR)

        # ── Step 2：边界拓展（不剔除，强制给出边界 + 质量标记）─────────────────
        left, left_quality = _find_boundary(
            I_corr, p, direction=-1,
            threshold_arr=boundary_thresh_arr,
            consec=boundary_consec, n_pixels=n,
            saddle_n_sigma=saddle_n_sigma,
            max_half_width=max_half_width,
        )
        right, right_quality = _find_boundary(
            I_corr, p, direction=+1,
            threshold_arr=boundary_thresh_arr,
            consec=boundary_consec, n_pixels=n,
            saddle_n_sigma=saddle_n_sigma,
            max_half_width=max_half_width,
        )

        # 边界质量综合判断：两侧中较差的一个决定整体标记
        boundary_quality = _combine_boundary_quality(left_quality, right_quality)
        if boundary_quality != "自然收敛":
            if "强制截断" in boundary_quality:
                fail_reasons.append(FlagReason.BOUNDARY_FORCED)
            if "鞍点过高" in boundary_quality:
                fail_reasons.append(FlagReason.BOUNDARY_HIGH_SADDLE)

        half_width = (right - left) / 2.0

        # ── Step 3：宽度 × 自适应SNR 联合判断（不剔除）─────────────────────────
        #    Step 1 的 height=min_snr*noise_std 已保证所有候选峰 SNR>=min_snr，
        #    单独的"SNR不足"判断在这里已经恒为假，不再重复判断。
        #    但 min_snr 这个统一底线对"窄峰"区分度不够：半宽只有1~2px的峰，
        #    参与判断的像素点很少，纯噪声偶然凑出一个临界SNR的概率明显更高，
        #    需要更高的SNR才能视为可信信号，而不是宽峰那一档"过线即可"的标准。
        #    snr_scale_reference_width 与 min_half_width 解耦：后者是"过窄"
        #    的硬性标记线（可随仪器调低，比如新仪器调到1），前者是"足够宽、
        #    无需额外信噪比加成"的参考宽度，默认保持3.0，不随min_half_width
        #    联动——否则把min_half_width调低后，这里的窄峰伪峰判定也会跟着
        #    失效（窄峰会重新被视为"宽度正常"，绕过本该有的高SNR要求）。
        if half_width < snr_scale_reference_width:
            required_snr = min_snr * (snr_scale_reference_width / max(half_width, 0.1))
        else:
            required_snr = min_snr

        # 注意：这里不能用 is_narrow(=half_width<min_half_width) 来gate这个判断，
        # 否则 min_half_width 被调低后（比如新仪器调到1），半宽1.0~2.9px的伪峰会
        # 因为"不算窄"而绕过自适应SNR要求，重新被判定为正常峰——这正是之前的bug。
        # SUSPECTED_ARTIFACT 必须只取决于 snr_scale_reference_width，与 min_half_width
        # 彻底解耦。
        if peak_snr < required_snr:
            fail_reasons.append(
                f"{FlagReason.SUSPECTED_ARTIFACT}"
                f"(half_width={half_width:.2f}, snr={peak_snr:.2f}<要求{required_snr:.2f})"
            )
        elif half_width < min_half_width:
            # SNR已经达到该宽度下的可信要求，但仍窄于硬性下限 → 单纯标记宽度
            # 异常，不再质疑其真实性
            fail_reasons.append(
                f"{FlagReason.WIDTH_TOO_NARROW}(half_width={half_width:.2f}<{min_half_width})"
            )
        if half_width > max_half_width:
            fail_reasons.append(
                f"{FlagReason.WIDTH_TOO_WIDE}(half_width={half_width:.2f}>{max_half_width})"
            )

        # ── Step 4：提取完整窗口，检查负值异常（标记，不剔除）───────────────────
        pixels_full, I_win_full = extract_peak_window(I_corr, left, right)
        negative_fraction = (I_win_full < 0).mean() if len(I_win_full) > 0 else 0.0
        if negative_fraction > 0.3:
            fail_reasons.append(
                f"{FlagReason.NEGATIVE_WING}(占比={negative_fraction:.1%})"
            )

        # ── Step 4b：核心窗口，质心/对称性计算（始终执行，不受标记影响）─────────
        core_left, core_right = _find_core_window(
            I_corr, p, left, right, peak_height, core_height_ratio,
        )
        pixels_core, I_win_core = extract_peak_window(I_corr, core_left, core_right)
        I_win_core_clipped = np.clip(I_win_core, 0, None)

        centroid = compute_centroid(pixels_core, I_win_core_clipped)
        sigma    = compute_sigma(pixels_core, I_win_core_clipped, centroid)
        skew     = compute_skewness(pixels_core, I_win_core_clipped, centroid, sigma)

        # ── Step 5：对称性标记（不剔除）───────────────────────────────────────
        if abs(skew) > skewness_threshold:
            fail_reasons.append(
                f"{FlagReason.ASYMMETRIC}(skew={skew:+.4f}, 阈值={skewness_threshold})"
            )

        passed_all = (len(fail_reasons) == 0)

        # ── 记录完整检测过程到 log（不论是否通过）──────────────────────────────
        logger.info(
            f"候选峰 px={p}  centroid={centroid:.3f}  SNR={peak_snr:.2f}  "
            f"skew={skew:+.4f}  half_width={half_width:.1f}  "
            f"boundary=[{left},{right}]({boundary_quality})  "
            f"通过={'是' if passed_all else '否'}"
            + (f"  原因: {'; '.join(fail_reasons)}" if fail_reasons else "")
        )

        results.append(PeakResult(
            peak_pixel      = int(p),
            centroid        = centroid,
            left            = left,
            right           = right,
            core_left       = core_left,
            core_right      = core_right,
            half_width      = half_width,
            snr             = peak_snr,
            skewness        = skew,
            peak_height     = peak_height,
            passed_all      = passed_all,
            fail_reasons    = fail_reasons,
            boundary_quality= boundary_quality,
        ))

    results.sort(key=lambda r: r.centroid)
    n_passed = sum(1 for r in results if r.passed_all)
    logger.info(f"候选峰总数: {len(results)}  完全通过: {n_passed}  "
                f"存在标记: {len(results) - n_passed}")
    return results


# ── 拓展接口（可被 explorer / 分辨率模块单独调用）────────────────────────────

def extract_peak_window(
    I_corr: np.ndarray,
    left: int,
    right: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 [left, right] 闭区间的像素坐标和扣基线强度。"""
    pixels = np.arange(left, right + 1, dtype=float)
    I_win  = I_corr[left: right + 1].copy()
    return pixels, I_win


def compute_centroid(pixels: np.ndarray, I_win: np.ndarray) -> float:
    """强度加权质心，要求 I_win >= 0。"""
    total = I_win.sum()
    if total <= 0:
        return float(pixels[len(pixels) // 2])
    return float(np.dot(I_win, pixels) / total)


def compute_sigma(
    pixels: np.ndarray,
    I_win: np.ndarray,
    centroid: float,
) -> float:
    """强度加权二阶矩（标准差），用于 skewness 归一化。"""
    total = I_win.sum()
    if total <= 0:
        return 1.0
    var = np.dot(I_win, (pixels - centroid) ** 2) / total
    return float(np.sqrt(max(var, 1e-12)))


def compute_skewness(
    pixels  : np.ndarray,
    I_win   : np.ndarray,
    centroid: float,
    sigma   : Optional[float] = None,
) -> float:
    """归一化三阶矩 skewness。正→右拖尾，负→左拖尾，0→完全对称"""
    if sigma is None:
        sigma = compute_sigma(pixels, I_win, centroid)
    total = I_win.sum()
    if total <= 0 or sigma < 1e-12:
        return 0.0
    return float(np.dot(I_win, (pixels - centroid) ** 3) / (total * sigma ** 3))


# ── 内部工具 ──────────────────────────────────────────────────────────────────

def _combine_boundary_quality(left_q: str, right_q: str) -> str:
    """合并左右两侧的边界质量描述，取较差的一侧作为整体标记。"""
    priority = {"自然收敛": 0, "鞍点过高(疑似双峰粘连)": 1, "强制截断": 2}
    worse = max([left_q, right_q], key=lambda q: priority.get(q, 0))
    if left_q == right_q:
        return left_q
    return f"左:{left_q} / 右:{right_q}" if left_q != "自然收敛" and right_q != "自然收敛" \
           else worse


def _find_core_window(
    I_corr           : np.ndarray,
    peak_idx         : int,
    left             : int,
    right            : int,
    peak_height      : float,
    core_height_ratio: float,
) -> Tuple[int, int]:
    """
    在完整窗口 [left, right] 内，找到峰高 core_height_ratio 比例以上的
    核心区域，排除裙边。逻辑与之前版本一致。
    """
    core_thresh = peak_height * core_height_ratio

    core_left = left
    for idx in range(peak_idx, left - 1, -1):
        if I_corr[idx] < core_thresh:
            core_left = idx + 1
            break
    else:
        core_left = left

    core_right = right
    for idx in range(peak_idx, right + 1):
        if I_corr[idx] < core_thresh:
            core_right = idx - 1
            break
    else:
        core_right = right

    core_left  = max(left,  min(core_left,  peak_idx))
    core_right = min(right, max(core_right, peak_idx))

    return core_left, core_right


def _find_boundary(
    I_corr        : np.ndarray,
    peak_idx      : int,
    direction     : int,
    threshold_arr : np.ndarray,
    consec        : int,
    n_pixels      : int,
    max_half_width: int,
    saddle_n_sigma: float = 1.5,
) -> Tuple[int, str]:
    """
    从峰顶向一侧逐像素行走，寻找边界。不再返回 None ——任何情况下都
    给出一个具体的边界像素位置，同时返回质量描述供上层打标签。

    返回的质量描述三种之一：
        "自然收敛"              ：标准回落（连续 consec 点低于阈值）
        "鞍点过高(疑似双峰粘连)"  ：找到了谷底，但谷底明显高于基线阈值
                                    （说明两峰未真正分开，或仪器分辨率不足）
        "强制截断"               ：搜索到 max_half_width 仍未找到任何
                                    拐点或自然回落，强制在此处截断

    Parameters
    ----------
    direction      : -1 向左搜索，+1 向右搜索
    threshold_arr  : 逐像素阈值数组 = boundary_n_sigma × local_noise
    consec         : 连续满足点数阈值
    max_half_width : 强制截断的最大搜索范围（像素）
    saddle_n_sigma : 谷底判定的宽容倍数，用于区分"自然收敛"与"鞍点过高"
    """
    consec_count = 0
    first_hit    = None
    prev_val     = I_corr[peak_idx]
    descending   = True

    limit = max_half_width  # 强制截断的硬上限，避免无限搜索
    steps = 0

    idx = peak_idx + direction
    while 0 <= idx < n_pixels and steps < limit:
        val    = I_corr[idx]
        thresh = threshold_arr[idx]

        # ── 标准回落判断 ─────────────────────────────────────────────────────
        if val <= thresh:
            if consec_count == 0:
                first_hit = idx
            consec_count += 1
            if consec_count >= consec:
                return first_hit, "自然收敛"
            descending = True

        else:
            if val > prev_val:
                if descending:
                    saddle_thresh = saddle_n_sigma * thresh
                    saddle_idx = idx - direction
                    if prev_val <= saddle_thresh:
                        return saddle_idx, "自然收敛"
                    else:
                        # 谷底过高：不再放弃，而是采用这个谷底作为边界，
                        # 但标记为"鞍点过高"，提示人工核查是否双峰粘连
                        return saddle_idx, "鞍点过高(疑似双峰粘连)"
            consec_count = 0
            first_hit    = None
            descending   = (val < prev_val)

        prev_val = val
        idx += direction
        steps += 1

    # 搜索到上限仍未找到任何收敛点或拐点 → 强制截断在当前位置
    forced_idx = idx - direction  # 回退到循环内最后一次有效的位置
    forced_idx = max(0, min(n_pixels - 1, forced_idx))
    return forced_idx, "强制截断"