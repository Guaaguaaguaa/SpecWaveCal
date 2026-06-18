"""
calibration.py — 波长定标
职责：根据有效峰质心像素位置和用户提供的参考波长，拟合像素→波长多项式
暴露接口：
    CalibrationResult                        — 定标结果数据类
    calibrate(centroids, wavelengths, degree) -> CalibrationResult
    pixel_to_wavelength(pixels, result)      -> np.ndarray
    wavelength_to_pixel(wavelengths, result) -> np.ndarray
    print_calibration_report(result)         — 打印定标质量报告

锚点匹配接口（通用几何匹配，与具体光源无关）：
    AnchorMatch                                       — 单个锚点匹配结果数据类
    match_anchors_to_peaks(anchor_pixels, peak_centroids,
                            match_tolerance, peak_quality_flags)  -> list[AnchorMatch]
        将一组理论锚点像素位置，与一组候选峰质心做最近邻匹配。

        设计变更（务实优先策略）：peak_finder.py 不再自动剔除任何候选峰，
        所有候选峰（包括存在质量标记的）都会参与匹配。任何锚点找不到落在
        容差范围内的候选峰，立即抛出 AnchorMatchError，不做任何静默兜底。
        若传入 peak_quality_flags，匹配结果会附带该峰是否通过质量检测的
        信息，供报告环节展示，最终是否采用交由人工判断。

    verify_anchor_ratios(anchor_pixels, matched_centroids, tolerance)
        -> None（通过则无返回，不通过抛出 AnchorRatioError）
        验证已匹配的锚点之间，实际质心间距与锚点理论间距的比例是否一致。
        用于排除"匹配到了，但其实匹配错了邻近峰"的情况——同一台仪器上
        不同特征峰之间的相对像素间距应保持稳定比例关系。

拓展接口（供 explorer / 验证使用）：
    residuals(centroids, wavelengths, result) -> np.ndarray
    evaluate_fit_quality(result)             -> dict
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.optimize import curve_fit


# ── 异常类型 ──────────────────────────────────────────────────────────────────

class AnchorMatchError(Exception):
    """锚点附近找不到任何落在容差范围内的合格候选峰时抛出。"""
    pass


class AnchorRatioError(Exception):
    """锚点间距比例交叉验证失败时抛出（疑似匹配到了邻近的错误峰）。"""
    pass


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class CalibrationResult:
    """
    波长定标结果，包含多项式系数和质量指标。
    多项式方向：pixel -> wavelength
        wavelength = sum(coeffs[i] * pixel^i)  (numpy.polyval 约定：高次在前)
    """
    coeffs      : np.ndarray    # np.polyfit 输出，高次在前，长度 = degree+1
    degree      : int
    rms_nm      : float         # 残差 RMS（nm）
    rms_px      : float         # 残差 RMS（像素），需色散率才能算，用线性近似
    max_resid_nm: float         # 最大绝对残差（nm）
    centroids   : np.ndarray    # 参与拟合的质心像素（已排序）
    ref_wavelengths: np.ndarray # 对应参考波长（nm）
    fitted_wavelengths: np.ndarray  # 拟合值
    residuals_nm: np.ndarray    # 残差 = fitted - ref（nm）
    dispersion_nm_per_px: float # 中心像素处色散率（nm/px），线性近似


@dataclass
class AnchorMatch:
    """单个锚点与候选峰质心的匹配结果。"""
    anchor_index    : int      # 锚点在原始列表中的序号（用于报错定位）
    anchor_pixel    : float    # 理论锚点像素位置
    matched_centroid: float    # 匹配到的真实质心像素位置
    distance        : float    # |matched_centroid - anchor_pixel|
    passed_all      : bool = True       # 匹配到的候选峰是否通过全部质量检测
    fail_reasons    : List[str] = field(default_factory=list)  # 未通过项原因


# ── 锚点匹配（通用几何匹配，与具体光源无关）──────────────────────────────────

def match_anchors_to_peaks(
    anchor_pixels   : List[float],
    peak_centroids  : List[float],
    match_tolerance : float = 5.0,
    peak_passed_flags: Optional[List[bool]] = None,
    peak_fail_reasons : Optional[List[List[str]]] = None,
) -> List[AnchorMatch]:
    """
    将理论锚点位置与候选峰质心做最近邻匹配。

    设计说明（务实优先策略）：调用方应对全谱跑完整寻峰流程（find_peaks），
    其结果不再自动剔除任何候选峰，而是给每个峰打上质量标记。本函数对
    全部候选峰（不论是否通过质量检测）一视同仁地参与匹配，但若提供了
    peak_passed_flags / peak_fail_reasons，会把匹配到的那个峰的质量状态
    一并带入 AnchorMatch，供后续报告展示，最终是否采用交由人工判断。

    任何锚点在容差范围内找不到候选峰，立即抛出 AnchorMatchError 并终止，
    不做任何静默兜底（例如退化为直接找最大值），以保证流程的可追溯性。

    Parameters
    ----------
    anchor_pixels      : 理论锚点像素位置列表（光源注册表中的预期位置）
    peak_centroids     : 候选峰质心列表（find_peaks 的完整输出，不论是否
                          通过质量检测都应包含在内）
    match_tolerance    : 匹配容差（像素），候选峰与锚点的距离超过此值不算匹配
    peak_passed_flags  : 与 peak_centroids 一一对应的布尔列表，标记该峰是否
                          通过全部质量检测；为 None 时所有匹配结果的
                          passed_all 默认为 True（即不做状态标注）
    peak_fail_reasons  : 与 peak_centroids 一一对应的失败原因列表；
                          为 None 时所有匹配结果的 fail_reasons 为空列表

    Returns
    -------
    list[AnchorMatch]，与 anchor_pixels 一一对应，顺序保持一致

    Raises
    ------
    AnchorMatchError : 任意一个锚点找不到容差范围内的候选峰
    """
    if len(peak_centroids) == 0:
        raise AnchorMatchError(
            "候选峰质心列表为空，无法进行锚点匹配。"
            "请检查寻峰流程是否正常运行。"
        )

    if peak_passed_flags is not None and len(peak_passed_flags) != len(peak_centroids):
        raise ValueError(
            f"peak_passed_flags 长度 ({len(peak_passed_flags)}) 与 "
            f"peak_centroids 长度 ({len(peak_centroids)}) 不一致"
        )
    if peak_fail_reasons is not None and len(peak_fail_reasons) != len(peak_centroids):
        raise ValueError(
            f"peak_fail_reasons 长度 ({len(peak_fail_reasons)}) 与 "
            f"peak_centroids 长度 ({len(peak_centroids)}) 不一致"
        )

    centroids_arr = np.asarray(peak_centroids, dtype=float)
    matches: List[AnchorMatch] = []

    for i, anchor in enumerate(anchor_pixels):
        distances = np.abs(centroids_arr - anchor)
        nearest_idx = int(np.argmin(distances))
        nearest_dist = float(distances[nearest_idx])

        if nearest_dist > match_tolerance:
            raise AnchorMatchError(
                f"锚点 #{i+1} (理论像素位置 {anchor:.2f}) 在容差 "
                f"±{match_tolerance} px 范围内找不到任何候选峰。"
                f"最近的候选峰质心在 {centroids_arr[nearest_idx]:.2f} px，"
                f"距离 {nearest_dist:.2f} px，超出容差。"
                f"定标中止，请检查该锚点附近的实际峰是否存在。"
            )

        passed = peak_passed_flags[nearest_idx] if peak_passed_flags is not None else True
        reasons = peak_fail_reasons[nearest_idx] if peak_fail_reasons is not None else []

        matches.append(AnchorMatch(
            anchor_index     = i,
            anchor_pixel     = float(anchor),
            matched_centroid = float(centroids_arr[nearest_idx]),
            distance         = nearest_dist,
            passed_all       = bool(passed),
            fail_reasons     = list(reasons),
        ))

    return matches


def verify_anchor_ratios(
    matches  : List[AnchorMatch],
    tolerance: float = 0.10,
) -> None:
    """
    验证已匹配锚点之间的实际质心间距，与锚点理论间距的比例是否一致。

    设计背景：同一台仪器上，不同特征峰之间的相对像素间距应保持稳定的
    比例关系（由光栅色散特性决定）。若某个锚点其实误匹配到了邻近的
    错误峰（例如伴峰、混叠峰），它与相邻锚点的实际间距比例会明显偏离
    其他点对，由此可以发现匹配错误。

    做法：以相邻锚点对为单位，计算 实际间距 / 理论间距 的比值，
    所有比值应彼此接近（同一台仪器的整体缩放系数应一致）。
    若某一对的比值与中位数比值的相对偏差超过 tolerance，判定异常并抛出。

    Parameters
    ----------
    matches   : match_anchors_to_peaks() 的输出，顺序需与锚点原始顺序一致
                （函数内部按 anchor_pixel 排序后再计算相邻间距）
    tolerance : 允许的相对偏差，默认 0.10（即 ±10%）

    Raises
    ------
    AnchorRatioError : 任意相邻锚点对的间距比例偏离中位数超过 tolerance

    Notes
    -----
    至少需要 3 个锚点才能做有意义的比例交叉验证（2个点只能算1个比值，
    没有其他比值可比较）。少于 3 个锚点时直接跳过验证，不报错。
    """
    if len(matches) < 3:
        return  # 点数不足，无法做有意义的交叉验证，静默跳过

    # 按理论锚点位置排序，确保相邻关系正确
    sorted_matches = sorted(matches, key=lambda m: m.anchor_pixel)

    ratios = []
    pair_info = []
    for i in range(len(sorted_matches) - 1):
        a, b = sorted_matches[i], sorted_matches[i + 1]
        theoretical_gap = b.anchor_pixel - a.anchor_pixel
        actual_gap      = b.matched_centroid - a.matched_centroid

        if theoretical_gap <= 0:
            raise AnchorRatioError(
                f"锚点理论位置异常：锚点#{a.anchor_index+1} 和 #{b.anchor_index+1} "
                f"理论间距 <= 0，请检查锚点注册表是否按顺序排列。"
            )

        ratio = actual_gap / theoretical_gap
        ratios.append(ratio)
        pair_info.append((a.anchor_index, b.anchor_index, ratio))

    median_ratio = float(np.median(ratios))

    for (idx_a, idx_b, ratio) in pair_info:
        relative_dev = abs(ratio - median_ratio) / median_ratio
        if relative_dev > tolerance:
            raise AnchorRatioError(
                f"锚点 #{idx_a+1} 与 #{idx_b+1} 之间的实际/理论间距比例 "
                f"({ratio:.4f}) 与整体中位数比例 ({median_ratio:.4f}) "
                f"相对偏差达 {relative_dev:.1%}，超过容差 {tolerance:.0%}。"
                f"疑似其中一个锚点匹配到了错误的邻近峰，定标中止。"
            )


# ── 主入口 ────────────────────────────────────────────────────────────────────

def calibrate(
    centroids   : List[float],
    wavelengths : List[float],
    degree      : int = 3,
) -> CalibrationResult:
    """
    多项式波长定标。

    Parameters
    ----------
    centroids   : 质心像素位置列表，与 wavelengths 一一对应
    wavelengths : 对应的 NIST 参考波长（nm）
    degree      : 多项式阶数，默认 3（三次）

    Returns
    -------
    CalibrationResult

    Notes
    -----
    - 至少需要 degree+2 个点（留一个自由度用于残差评估）
    - 输入会按像素排序，顺序不要求一致
    - 推荐先用 explorer.py 确认特征峰，再调用此函数
    """
    centroids   = np.asarray(centroids,   dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)

    if len(centroids) != len(wavelengths):
        raise ValueError(
            f"centroids 和 wavelengths 长度不一致: "
            f"{len(centroids)} vs {len(wavelengths)}"
        )
    if len(centroids) < degree + 2:
        raise ValueError(
            f"至少需要 {degree + 2} 个点做 {degree} 阶拟合，"
            f"当前只有 {len(centroids)} 个"
        )

    # 按像素升序排列
    order = np.argsort(centroids)
    centroids   = centroids[order]
    wavelengths = wavelengths[order]

    # 多项式拟合（像素 → 波长）
    coeffs = np.polyfit(centroids, wavelengths, degree)
    fitted = np.polyval(coeffs, centroids)
    resid  = fitted - wavelengths

    rms_nm       = float(np.sqrt(np.mean(resid ** 2)))
    max_resid_nm = float(np.max(np.abs(resid)))

    # 色散率：在探测器中心像素处取一阶导数
    center_px = float(np.mean(centroids))
    deriv_coeffs = np.polyder(coeffs)
    dispersion = float(abs(np.polyval(deriv_coeffs, center_px)))

    rms_px = rms_nm / dispersion if dispersion > 0 else float("nan")

    return CalibrationResult(
        coeffs              = coeffs,
        degree              = degree,
        rms_nm              = rms_nm,
        rms_px              = rms_px,
        max_resid_nm        = max_resid_nm,
        centroids           = centroids,
        ref_wavelengths     = wavelengths,
        fitted_wavelengths  = fitted,
        residuals_nm        = resid,
        dispersion_nm_per_px= dispersion,
    )


# ── 正向/逆向转换 ──────────────────────────────────────────────────────────────

def pixel_to_wavelength(
    pixels: np.ndarray,
    result: CalibrationResult,
) -> np.ndarray:
    """像素 → 波长（nm），支持标量或数组输入。"""
    return np.polyval(result.coeffs, np.asarray(pixels, dtype=float))


def wavelength_to_pixel(
    wavelengths: np.ndarray,
    result     : CalibrationResult,
    n_pixels   : int = 2048,
) -> np.ndarray:
    """
    波长 → 像素，用数值方法求逆。
    在全像素范围内评估多项式，取最近点（对单调色散足够精确）。
    """
    wl     = np.asarray(wavelengths, dtype=float)
    px_all = np.arange(n_pixels, dtype=float)
    wl_all = np.polyval(result.coeffs, px_all)
    return np.array([
        float(px_all[np.argmin(np.abs(wl_all - w))])
        for w in wl
    ])


# ── 质量评估与报告 ─────────────────────────────────────────────────────────────

def evaluate_fit_quality(result: CalibrationResult) -> dict:
    """
    返回定标质量指标字典，供 explorer / pipeline 使用。
    包含：rms_nm, rms_px, max_resid_nm, dispersion, coverage_nm
    """
    coverage_nm = float(
        result.ref_wavelengths.max() - result.ref_wavelengths.min()
    )
    return {
        "n_points"          : len(result.centroids),
        "degree"            : result.degree,
        "rms_nm"            : round(result.rms_nm,        4),
        "rms_px"            : round(result.rms_px,        4),
        "max_resid_nm"      : round(result.max_resid_nm,  4),
        "dispersion_nm_per_px": round(result.dispersion_nm_per_px, 4),
        "coverage_nm"       : round(coverage_nm,          2),
    }


def print_calibration_report(
    result : CalibrationResult,
    matches: Optional[List["AnchorMatch"]] = None,
) -> None:
    """
    打印定标质量报告，适合在 pipeline 末尾调用。

    Parameters
    ----------
    result  : calibrate() 的输出
    matches : 可选，match_anchors_to_peaks() 的输出，顺序需与 result 中
              centroids 一致（按像素升序）。提供时会在表格中额外显示
              "质量检测"列，标注该匹配峰是否通过 peak_finder 的全部
              质量检测（不通过仅作提示，不影响已完成的拟合结果）。
    """
    q = evaluate_fit_quality(result)
    sep = "-" * 56

    # 若提供 matches，按 anchor_pixel 排序以与 result.centroids（已排序）对齐
    status_by_centroid = {}
    if matches:
        sorted_matches = sorted(matches, key=lambda m: m.matched_centroid)
        for m in sorted_matches:
            status_by_centroid[round(m.matched_centroid, 3)] = (
                "通过" if m.passed_all else f"标记({'; '.join(m.fail_reasons)})"
            )
        sep = "-" * 80

    print(sep)
    print("  波长定标报告")
    print(sep)
    print(f"  拟合阶数        : {q['degree']} 次多项式")
    print(f"  参与点数        : {q['n_points']}")
    print(f"  波长覆盖范围    : {q['coverage_nm']:.2f} nm")
    print(f"  色散率（中心）  : {q['dispersion_nm_per_px']:.4f} nm/px")
    print(f"  RMS 残差        : {q['rms_nm']:.4f} nm  ({q['rms_px']:.4f} px)")
    print(f"  最大绝对残差    : {q['max_resid_nm']:.4f} nm")
    print()
    if matches:
        print(f"  {'像素':>10}  {'参考波长(nm)':>14}  {'拟合波长(nm)':>14}  "
              f"{'残差(nm)':>10}  质量检测")
        print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*30}")
    else:
        print(f"  {'像素':>10}  {'参考波长(nm)':>14}  {'拟合波长(nm)':>14}  {'残差(nm)':>10}")
        print(f"  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*10}")

    for px, ref, fit, res in zip(
        result.centroids,
        result.ref_wavelengths,
        result.fitted_wavelengths,
        result.residuals_nm,
    ):
        flag = " ◄" if abs(res) == result.max_resid_nm else ""
        if matches:
            status = status_by_centroid.get(round(float(px), 3), "未知")
            print(f"  {px:10.3f}  {ref:14.4f}  {fit:14.4f}  {res:+10.4f}{flag}  {status}")
        else:
            print(f"  {px:10.3f}  {ref:14.4f}  {fit:14.4f}  {res:+10.4f}{flag}")
    print(sep)

    # 多项式系数
    print("\n  多项式系数（高次在前，np.polyval 格式）：")
    for i, c in enumerate(result.coeffs):
        power = result.degree - i
        print(f"    c[{power}] = {c:.8e}")
    print(sep)