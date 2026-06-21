"""
run_calibration.py — 波长定标主流程脚本
职责：选择数据文件 -> 判定光源类型 -> 全谱寻峰 -> RANSAC自动识别谱线对应关系
      -> 多项式拟合 -> 导出结果与报告

支持两种定标模式：
    单灯模式：选择一个 HgAr 灯文件，11 条谱线自动识别
    三灯联合模式：选择 KR + AR + NM 三个灯文件，各自寻峰后联合 RANSAC

使用方式：
    直接运行: python run_calibration.py
    会弹出文件选择窗口（支持多选），自动判断定标模式并执行。

设计原则（务必遵守）：
    任何环节出现异常（RANSAC 找不到足够内点等），立即弹窗报错并中止整个
    流程，不做任何静默兜底或退化处理。
"""

import os
import re
import sys
import datetime
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

from wavecal import (
    run_explorer, Config,
    calibrate, pixel_to_wavelength, print_calibration_report,
    ransac_match_wavelengths, print_ransac_report,
    calibrate_multi_lamp, MultiLampResult,
    RansacMatchError, FlagReason,
    auto_tune_config, AutoTuneError,
)
from lamp_registry import get_lamp_config, detect_lamp_from_filename


# ==============================================================================
# 全局工程配置参数（可根据实际实验条件调整）
# ==============================================================================
RANSAC_ITERATIONS       = 10000   # RANSAC 总迭代次数
RANSAC_MIN_INLIERS_RATIO = 0.5    # 最少需识别出已知波长总数的此比例
CALIBRATION_DEGREE      = 3       # 定标多项式阶数

# （原 IS3_DETECTOR_SIZE / IS3_MAX_HALF_WIDTH 两个写死常量已删除——
#  之前隐含"三灯联合=512通道"的假设不成立，三灯联合也可能用在2048通道
#  仪器上。现在 detector_size / max_half_width 等窗口参数统一由
#  auto_tune_config() 从实际数据里测，跟具体走单灯还是多灯路径无关）


# ==============================================================================
# 文件读取（兼容 SpecWaveCal 无 header CSV 和 IS3 带 header CSV）
# ==============================================================================

def _detect_skiprows(path: str) -> int:
    """检测数据起始行：跳过 Date/Temperature/Weavelenth 等元数据行。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i > 20:
                return 0
            if re.search(r"(date|temperature|weavelenth|weavelength|index)", line, re.I):
                continue
            parts = [p for p in re.split(r"[,\t ]+", line.strip()) if p]
            if len(parts) >= 2:
                num_pat = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
                if sum(1 for p in parts if num_pat.match(p)) / len(parts) >= 0.8:
                    return i
    return 0


def read_spectrum(path: str):
    """
    读取光谱文件，返回 (intensity, slope_sign)。

    intensity  — 1D 强度数组，**不做任何翻转**，保留仪器原始像素顺序。
    slope_sign — +1: 波长随像素递增（正常仪器）
                 -1: 波长随像素递减（传感器倒置，短波长在高像素端）
                  0: 无法从第一列判断（极少见，兜底用双向 RANSAC）

    判断依据：CSV 第一列若含波长值（>300 nm），直接看首尾方向；
             若为像素序号（如 0,1,2...），按正向递增处理。
    """
    skiprows = _detect_skiprows(path)
    df = pd.read_csv(path, header=None, skiprows=skiprows)

    if df.shape[1] >= 2:
        col0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        intensity = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    else:
        col0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        intensity = col0.copy()  # fallback

    valid = ~(col0.isna() | intensity.isna())
    col0_vals = np.asarray(col0[valid], dtype=float)
    intensity = np.asarray(intensity[valid], dtype=float)

    slope_sign = _detect_slope(col0_vals)

    return intensity, slope_sign


def _detect_slope(col0: np.ndarray) -> int:
    """
    从 CSV 第一列推断色散方向。
    返回 +1（波长随像素递增）或 -1（波长随像素递减，传感器倒置）。

    判断逻辑：
    - 若第一列最大值 > 300 → 是波长值 → 比较首尾判断递增/递减
    - 若第一列最大值 ≤ 300 → 是像素序号（0,1,2...），按正向递增处理
    """
    if len(col0) < 2:
        return +1

    col0_max = float(col0.max())

    if col0_max > 300:
        # 第一列是波长值（HgAr 范围 435~1014 nm，远大于像素序号）
        if float(col0[0]) > float(col0[-1]):
            return -1   # 递减 → 传感器倒置
        else:
            return +1   # 递增 → 正常
    else:
        # 第一列是像素序号（0, 1, 2, ...），正向递增
        return +1


# ==============================================================================
# 步骤函数
# ==============================================================================

def select_csv_files() -> list:
    """弹出多选文件窗口，返回用户选择的文件路径列表。"""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_paths = filedialog.askopenfilenames(
        title="选择定标灯文件（单灯选1个，三灯选KR+AR+NM各1个）",
        filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")],
    )
    root.destroy()
    if not file_paths:
        raise SystemExit("用户取消了文件选择。")
    return list(file_paths)


def detect_calibration_mode(file_paths: list) -> tuple:
    """
    根据文件名和数量判断定标模式，规则严格、不做模糊兜底：
        - 恰好 1 个文件，且能从文件名识别为 HgAr → ("single", "HgAr")
        - 恰好 3 个文件，且能一一对应到 KR / AR / NM（每个文件唯一匹配一个
          灯名，三个灯各出现一次，不能重复、不能缺失、不能有文件匹配不上）
          → ("multi", ["KR", "AR", "NM"])
        - 其它任何情况（文件数不对、单文件不是HgAr、三文件对应不上）一律
          直接报错并说明具体原因，不弹窗让用户手动选——避免在文件命名有
          歧义时静默猜测出错误的定标方式
    """
    n = len(file_paths)

    if n == 1:
        fname = os.path.basename(file_paths[0])
        detected = detect_lamp_from_filename(fname)
        if detected == "HgAr":
            return ("single", "HgAr")
        reason = (f"识别为 [{detected}]，但单灯模式仅支持 HgAr"
                   if detected else "文件名中未识别出任何已注册的光源标识")
        _fail_mode_detection(
            f"选择了 1 个文件，{reason}。\n"
            f"文件: {fname}\n\n"
            f"单灯模式仅支持 HgAr；如需做 KR/AR/NM 三灯联合定标，"
            f"请同时选择 3 个、分别在文件名中含 KR / AR / NM 标识的文件。"
        )

    if n == 3:
        file_lamp = {}   # 文件名 -> 识别到的灯名 / None / "AMBIGUOUS:.."
        for p in file_paths:
            fname = os.path.basename(p).upper()
            matches = [lamp for lamp in ("KR", "AR", "NM") if lamp in fname]
            if len(matches) == 1:
                file_lamp[p] = matches[0]
            elif len(matches) == 0:
                file_lamp[p] = None
            else:
                file_lamp[p] = "AMBIGUOUS:" + "+".join(matches)

        matched = [v for v in file_lamp.values() if v in ("KR", "AR", "NM")]
        if sorted(matched) == ["AR", "KR", "NM"]:
            return ("multi", ["KR", "AR", "NM"])

        detail = "\n".join(
            f"  {os.path.basename(p)} → {v if v else '无法识别'}"
            for p, v in file_lamp.items()
        )
        _fail_mode_detection(
            f"选择了 3 个文件，但不能唯一对应到 KR/AR/NM 三灯"
            f"（要求每个文件恰好匹配一个灯名，三个灯各出现一次，"
            f"不能重复也不能缺失）。各文件识别结果：\n{detail}"
        )

    _fail_mode_detection(
        f"选择了 {n} 个文件，无法判断定标模式。\n"
        f"单灯模式请选 1 个 HgAr 文件；三灯联合请选 3 个分别含 "
        f"KR / AR / NM 标识的文件。"
    )


def _fail_mode_detection(reason: str):
    """统一的模式判断失败处理：弹窗明确告知原因，然后中止。"""
    messagebox.showerror("无法判断定标模式", reason)
    raise SystemExit(reason)


def run_full_pipeline(intensity: np.ndarray, log_path: str):
    """跑完整寻峰流程，返回 PipelineResult。"""
    cfg = Config(log_path=log_path, quality_csv_path="")
    result = run_explorer(intensity, cfg, save_csv=False)
    return result


def generate_report(
    report_path        : str,
    file_path           : str,
    lamp_name           : str,
    lamp_description     : str,
    ransac_result        ,
    calibration_result   ,
) -> None:
    """生成定标质量报告文本文件。"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inliers = sorted(ransac_result.inliers, key=lambda x: x.true_wavelength)
    centroid_to_quality = getattr(ransac_result, '_centroid_to_quality', {})

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("光谱仪波长定标报告 (Wavelength Calibration Report)\n")
        f.write("=" * 60 + "\n")
        f.write(f"定标处理时间: {current_time}\n")
        f.write(f"数据源输入文件: {file_path}\n")
        f.write(f"选定光源类型: {lamp_name} ({lamp_description})\n")
        f.write(f"识别方式: RANSAC 自动谱线识别（二次模型，无需仪器专属锚点先验）\n")
        f.write(f"RANSAC 二次模型: WL = {ransac_result.a2:.8e}*px^2 "
                f"+ {ransac_result.a1:.6f}*px + {ransac_result.a0:.4f}\n")
        f.write(f"RANSAC 识别内点数: {len(inliers)}\n\n")

        if ransac_result.unmatched_wavelengths:
            f.write(f"*** 未被识别的已知波长: {ransac_result.unmatched_wavelengths} ***\n\n")

        f.write("------------------ 寻峰与拟合明细 ------------------\n")
        f.write(f"{'序号':<4}{'标准波长(nm)':<14}"
                f"{'质心像素':<14}{'标定波长(nm)':<16}"
                f"{'残差(nm)':<12}质量检测\n")

        for i, inl in enumerate(inliers):
            fitted_wl = calibration_result.fitted_wavelengths[i]
            resid     = calibration_result.residuals_nm[i]
            px = inl.centroid_pixel
            passed, reasons = centroid_to_quality.get(round(px, 6), (True, []))
            status = "通过" if passed else f"标记: {'; '.join(reasons)}"
            f.write(f"{i+1:<6}"
                    f"{inl.true_wavelength:<16.3f}"
                    f"{px:<16.3f}"
                    f"{fitted_wl:<18.4f}"
                    f"{resid:+.4f}  {status}\n")

        n_flagged = sum(
            1 for inl in inliers
            if not centroid_to_quality.get(round(inl.centroid_pixel, 6), (True, []))[0]
        )
        if n_flagged > 0:
            f.write(f"\n*** 注意: 以上 {n_flagged} 个匹配点存在质量标记，"
                    f"建议人工核查对应谱线形态后再决定是否采用该定标结果 ***\n")

        f.write("\n------------------ 拟合质量评估 ------------------\n")
        f.write(f"多项式拟合阶数 (Order): {calibration_result.degree} 阶多项式最小二乘拟合\n")
        f.write(f"最大绝对残差 (Max Error): {calibration_result.max_resid_nm:.4f} nm\n")
        f.write(f"定标均方根残差 (RMSE): {calibration_result.rms_nm:.4f} nm "
                f"({calibration_result.rms_px:.4f} px)\n")
        f.write(f"中心像素色散率: {calibration_result.dispersion_nm_per_px:.4f} nm/px\n\n")

        f.write(f"拟合多项式系数 (degree={calibration_result.degree}，"
                f"λ = sum(c[i] * x^i)，高次在前):\n")
        for i, c in enumerate(calibration_result.coeffs):
            power = calibration_result.degree - i
            f.write(f"c[{power}] = {c:.8e}\n")
        f.write("=" * 60 + "\n")


# ==============================================================================
# 单灯定标流程
# ==============================================================================

def run_single_lamp_calibration(
    file_path: str,
    lamp_name: str,
    lamp,
    intensity: np.ndarray,
    slope_sign: int,
    dir_name: str,
    base_name: str,
):
    """单灯模式：HgAr 等，一个文件 11 条线。"""
    N = len(intensity)
    log_path = os.path.join(dir_name, f"{base_name}_peakfind_log.txt")

    # ── 传感器倒置处理：flip 数据使波长随像素递增，方便 RANSAC ────────
    if slope_sign == -1:
        intensity_proc = intensity[::-1]
        print(f"[INFO] 第一列波长递减 → 传感器倒置。"
              f"处理时 flip 数据，输出时还原像素坐标 (N={N})。")
    else:
        intensity_proc = intensity

    # ── 寻峰 + RANSAC（始终正斜率，数据已归一化） ──────────────────────
    pipeline_result = run_full_pipeline(intensity_proc, log_path)
    candidate_centroids_proc = pipeline_result.centroids
    candidate_passed    = [p.passed_all   for p in pipeline_result.peaks]
    candidate_reasons   = [p.fail_reasons for p in pipeline_result.peaks]

    if len(candidate_centroids_proc) == 0:
        raise RuntimeError(
            f"全谱寻峰未找到任何候选峰。详细检测记录请查看: {log_path}"
        )

    # 提取标记为不可信（如 SUSPECTED_ARTIFACT）的候选峰，送入RANSAC前预先
    # 排除——与三灯联合路径（multi_lamp.py）保持一致的行为
    candidate_excluded_proc = [
        c for c, reasons in zip(candidate_centroids_proc, candidate_reasons)
        if any(FlagReason.SUSPECTED_ARTIFACT in r for r in reasons)
    ]

    min_inliers = max(2, int(np.ceil(len(lamp.true_wavelengths) * RANSAC_MIN_INLIERS_RATIO)))

    ransac_result = ransac_match_wavelengths(
        candidate_centroids      = candidate_centroids_proc,
        true_wavelengths         = lamp.true_wavelengths,
        tolerance_nm             = lamp.ransac_tolerance_nm,
        n_iterations             = RANSAC_ITERATIONS,
        min_inliers              = min_inliers,
        require_positive_slope   = True,
        excluded_centroids       = candidate_excluded_proc,
    )
    print_ransac_report(ransac_result)

    # ── 还原像素坐标到仪器原始空间 ──────────────────────────────────────
    pairs = ransac_result.calibration_pairs  # [(px_flipped, wl), ...]
    if slope_sign == -1:
        pairs = [(N - px, wl) for px, wl in pairs]
        candidate_centroids = [N - c for c in candidate_centroids_proc]
        # 同步还原 inliers 的像素值（report/generate_report 依赖此数据）
        for inl in ransac_result.inliers:
            inl.centroid_pixel = N - inl.centroid_pixel
    else:
        candidate_centroids = list(candidate_centroids_proc)

    # ── 多项式拟合 + 导出 ───────────────────────────────────────────────
    calibration_result = calibrate(
        centroids   = [p[0] for p in pairs],
        wavelengths = [p[1] for p in pairs],
        degree      = CALIBRATION_DEGREE,
    )
    print_calibration_report(calibration_result, matches=None)

    # 导出全谱定标 CSV（intensity 用原始顺序）
    all_pixels = np.arange(N)
    all_wavelengths = pixel_to_wavelength(all_pixels, calibration_result)
    calibrated_csv_path = os.path.join(dir_name, f"{base_name}_calibrated.csv")
    out_df = pd.DataFrame({"Wavelength_nm": all_wavelengths, "Intensity": intensity})
    out_df.to_csv(calibrated_csv_path, index=False, header=False)

    # 质量标记表（用还原后的像素坐标）
    centroid_to_quality = {}
    for c, passed, reasons in zip(candidate_centroids, candidate_passed, candidate_reasons):
        centroid_to_quality[round(c, 6)] = (passed, reasons)
    ransac_result._centroid_to_quality = centroid_to_quality

    # 报告
    report_path = os.path.join(dir_name, f"{base_name}_wavecal_report.txt")
    generate_report(report_path, file_path, lamp_name, lamp.description,
                    ransac_result, calibration_result)

    # 弹窗
    n_flagged = sum(1 for inl in ransac_result.inliers
                    if not centroid_to_quality.get(round(inl.centroid_pixel, 6), (True, []))[0])
    flag_note = f"\n⚠ {n_flagged} 个匹配点存在质量标记，详见报告" if n_flagged > 0 else ""
    unmatched_note = (f"\n⚠ {len(ransac_result.unmatched_wavelengths)} 条已知波长未被识别"
                      if ransac_result.unmatched_wavelengths else "")
    messagebox.showinfo(
        "波长定标成功",
        f"定标任务顺利完成！\n\n"
        f"光源类型: {lamp_name}\n"
        f"RANSAC 识别内点数: {len(ransac_result.inliers)} / {len(lamp.true_wavelengths)}\n"
        f"RMS残差: {calibration_result.rms_nm:.4f} nm\n"
        f"最大残差: {calibration_result.max_resid_nm:.4f} nm"
        f"{flag_note}{unmatched_note}\n\n"
        f"已导出:\n"
        f"1. {base_name}_calibrated.csv\n"
        f"2. {base_name}_wavecal_report.txt\n"
        f"3. {base_name}_peakfind_log.txt",
    )
    print(f"定标成功完成，结果已保存至: {dir_name}")


# ==============================================================================
# 三灯联合定标流程
# ==============================================================================

# 三灯联合的全部参考波长（KR + AR + NM）
MULTI_LAMP_WAVELENGTHS = [
    435.833, 546.074, 587.092, 696.543, 727.294,
    785.482, 850.887, 866.794, 892.869, 965.779, 1013.976,
]

def run_multi_lamp_calibration(
    file_paths: list,
    lamp_names: list,
    dir_name: str,
):
    """三灯联合模式：KR + AR + NM，各自寻峰后联合 RANSAC。"""
    # 读取（所有文件同属一台仪器，取第一个文件的方向）
    lamp_intensities_orig = {}  # 原始顺序
    lamp_intensities_proc = {}  # 处理用（可能已 flip）
    lamp_files = {}
    slope_sign = 0
    N = 0
    for p in file_paths:
        fname = os.path.basename(p).upper()
        for lamp in ["KR", "AR", "NM"]:
            if lamp in fname and lamp not in lamp_intensities_orig:
                intensity, ss = read_spectrum(p)
                lamp_intensities_orig[lamp] = intensity
                lamp_files[lamp] = p
                if slope_sign == 0:
                    slope_sign = ss
                    N = len(intensity)
                break

    if len(lamp_intensities_orig) != 3:
        raise RuntimeError(f"三灯联合需要 KR/AR/NM 各一个文件，当前仅识别到: {list(lamp_intensities_orig.keys())}")

    # ── 传感器倒置：flip 数据，方便 RANSAC ────────────────────────────
    if slope_sign == -1:
        for lamp in lamp_intensities_orig:
            lamp_intensities_proc[lamp] = lamp_intensities_orig[lamp][::-1]
        print(f"[INFO] 第一列波长递减 → 传感器倒置。"
              f"处理时 flip 数据，输出时还原像素坐标 (N={N})。")
    else:
        lamp_intensities_proc = dict(lamp_intensities_orig)

    # ── 自动推导这台仪器的寻峰窗口参数（不依赖任何写死的仪器型号常量）──
    #    从三灯各自的实际数据里测出"明显可信"的峰有多宽，据此反推
    #    min_half_width/max_half_width/min_peak_sep/baseline_window_size，
    #    detector_size 直接取实际谱长。min_snr/skewness_threshold 等
    #    不需要按仪器调的参数沿用 Config 默认值。
    base_cfg = Config(
        log_path=os.path.join(dir_name, "multi_lamp_peakfind_log.txt"),
        quality_csv_path="",
    )
    try:
        cfg = auto_tune_config(lamp_intensities_proc, base_config=base_cfg)
    except AutoTuneError as e:
        raise RuntimeError(
            f"三灯联合定标的自动调参失败，无法继续: {e}"
        ) from e

    result = calibrate_multi_lamp(
        lamp_intensities_proc,
        MULTI_LAMP_WAVELENGTHS,
        config=cfg,
        ransac_kwargs=dict(
            tolerance_nm=2.0,
            n_iterations=RANSAC_ITERATIONS,
            random_seed=42,
            require_positive_slope=True,
        ),
    )
    print_ransac_report(result.ransac_result)

    # ── 还原像素坐标 ──────────────────────────────────────────────────
    pairs_flipped = result.ransac_result.calibration_pairs
    if slope_sign == -1:
        pairs = [(N - px, wl) for px, wl in pairs_flipped]
        for inl in result.ransac_result.inliers:
            inl.centroid_pixel = N - inl.centroid_pixel
    else:
        pairs = pairs_flipped

    cal = calibrate(
        centroids   = [p[0] for p in pairs],
        wavelengths = [p[1] for p in pairs],
        degree      = CALIBRATION_DEGREE,
    )
    print_calibration_report(cal, matches=None)

    # 导出每个灯的定标 CSV（intensity 用原始顺序）
    for lamp_name, intensity in lamp_intensities_orig.items():
        all_pixels = np.arange(len(intensity))
        all_wls = pixel_to_wavelength(all_pixels, cal)
        out_path = os.path.join(dir_name, f"multi_lamp_{lamp_name}_calibrated.csv")
        pd.DataFrame({"Wavelength_nm": all_wls, "Intensity": intensity}).to_csv(
            out_path, index=False, header=False
        )

    # 报告
    report_path = os.path.join(dir_name, "multi_lamp_wavecal_report.txt")
    file_list_str = "; ".join(lamp_files.values())
    generate_report(report_path, file_list_str, "KR+AR+NM",
                    "三灯联合定标 (Kr + Ar + NeHg)", result.ransac_result, cal)

    n_matched = len(pairs)
    n_total = len(MULTI_LAMP_WAVELENGTHS)
    unmatched = result.ransac_result.unmatched_wavelengths
    unmatched_note = f"\n⚠ {len(unmatched)} 条已知波长未被识别: {unmatched}" if unmatched else ""
    messagebox.showinfo(
        "三灯联合定标成功",
        f"定标任务顺利完成！\n\n"
        f"定标模式: KR + AR + NM 三灯联合\n"
        f"RANSAC 识别内点数: {n_matched} / {n_total}\n"
        f"RMS残差: {cal.rms_nm:.4f} nm\n"
        f"最大残差: {cal.max_resid_nm:.4f} nm"
        f"{unmatched_note}\n\n"
        f"已导出 multi_lamp_KR/AR/NM_calibrated.csv 和定标报告",
    )
    print(f"三灯联合定标完成，结果已保存至: {dir_name}")


# ==============================================================================
# 主流程
# ==============================================================================

def run_wavelength_calibration():
    try:
        # ── 1. 选择文件 ──────────────────────────────────────────────────────
        file_paths = select_csv_files()
        first_path = file_paths[0]
        dir_name  = os.path.dirname(first_path)

        # ── 2. 判断定标模式 ──────────────────────────────────────────────────
        mode, info = detect_calibration_mode(file_paths)

        if mode == "single":
            lamp_name = info
            lamp = get_lamp_config(lamp_name)
            intensity, slope_sign = read_spectrum(first_path)
            base_name = os.path.splitext(os.path.basename(first_path))[0]
            run_single_lamp_calibration(
                first_path, lamp_name, lamp, intensity, slope_sign, dir_name, base_name
            )

        elif mode == "multi":
            run_multi_lamp_calibration(file_paths, info, dir_name)

    except RansacMatchError as e:
        messagebox.showerror("定标中止：RANSAC 识别失败", str(e))
        print(f"定标中止: {e}", file=sys.stderr)

    except SystemExit as e:
        print(f"已取消: {e}")

    except Exception as e:
        messagebox.showerror("定标中止：未预期的错误", str(e))
        print(f"定标中止（未预期错误）: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    run_wavelength_calibration()