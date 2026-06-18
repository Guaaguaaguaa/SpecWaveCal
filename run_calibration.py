"""
run_calibration.py — 波长定标主流程脚本
职责：选择数据文件 -> 自动/手动判定光源类型 -> 全谱寻峰 -> 锚点匹配 ->
      比例交叉验证 -> 多项式拟合 -> 导出结果与报告

与具体光源相关的配置（参考波长、锚点像素位置）都放在 lamp_registry.py 中，
本脚本只负责流程编排，新增光源类型不需要改动此文件。

使用方式：
    直接运行: python run_calibration.py
    会弹出文件选择窗口，选择 CSV 后自动执行完整定标流程。

设计原则（务必遵守）：
    任何环节出现异常（找不到合格候选峰、比例交叉验证失败等），
    立即弹窗报错并中止整个流程，不做任何静默兜底或退化处理。
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from wavecal import (
    run_explorer, Config,
    match_anchors_to_peaks, verify_anchor_ratios,
    calibrate, pixel_to_wavelength,
    AnchorMatchError, AnchorRatioError,
)
from lamp_registry import get_lamp_config, detect_lamp_from_filename, LAMP_REGISTRY


# ==============================================================================
# 全局工程配置参数（可根据实际实验条件调整）
# ==============================================================================
MATCH_TOLERANCE       = 5.0    # 锚点匹配容差（像素），超过此距离视为找不到匹配
RATIO_TOLERANCE        = 0.10   # 比例交叉验证容差（相对偏差）
CALIBRATION_DEGREE     = 3      # 定标多项式阶数
MANUAL_SHIFT_OVERRIDE  = None   # 手动Shift覆盖；None 则启动自动 Shift 评估


# ==============================================================================
# 步骤函数
# ==============================================================================

def select_csv_file() -> str:
    """弹出文件选择窗口，返回用户选择的 CSV 路径。"""
    file_path = filedialog.askopenfilename(
        title="请选择需要进行波长定标的光谱数据文件",
        filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")],
    )
    if not file_path:
        raise SystemExit("用户取消了文件选择。")
    return file_path


def resolve_lamp_name(file_path: str) -> str:
    """
    根据文件名自动判断光源类型；判断不到则弹窗要求用户手动选择。
    不做任何静默猜测或默认回退。
    """
    detected = detect_lamp_from_filename(os.path.basename(file_path))
    if detected is not None:
        return detected

    # 自动判断失败，弹窗要求手动选择
    available = list(LAMP_REGISTRY.keys())
    choice = simpledialog.askstring(
        "无法自动识别光源类型",
        f"文件名中未找到已注册的光源标识。\n"
        f"当前已注册的光源: {', '.join(available)}\n"
        f"请手动输入光源名称（区分大小写）：",
    )
    if not choice or choice not in LAMP_REGISTRY:
        raise SystemExit(
            f"未提供有效的光源名称（输入: {choice!r}），定标中止。"
        )
    return choice


def auto_estimate_global_shift(
    intensity     : np.ndarray,
    anchor_pixel  : float,
    search_radius : int = 50,
) -> int:
    """
    通过在较宽范围内扫描强特征峰的极大值，自动计算系统硬件的整体
    像素漂移偏置量（Shift）。仅用于后续锚点匹配前的粗略对齐，
    精确质心仍由 find_peaks 的完整流程给出。
    """
    anchor_idx = int(round(anchor_pixel))
    start = max(0, anchor_idx - search_radius)
    end   = min(len(intensity), anchor_idx + search_radius + 1)
    sub_array = intensity[start:end]

    if len(sub_array) == 0:
        return 0

    local_max_idx = int(np.argmax(sub_array))
    actual_peak_pixel = start + local_max_idx
    return int(actual_peak_pixel - anchor_idx)


def run_full_pipeline(intensity: np.ndarray, log_path: str):
    """跑完整寻峰流程，返回 PipelineResult（含全部合格候选峰）。"""
    cfg = Config(log_path=log_path, quality_csv_path="")
    result = run_explorer(intensity, cfg, save_csv=False)
    return result


def generate_report(
    report_path        : str,
    file_path           : str,
    lamp_name           : str,
    lamp_description     : str,
    shift_mode           : str,
    matches              : list,
    true_wavelengths      : list,
    calibration_result    ,
) -> None:
    """生成定标质量报告文本文件，沿用原有报告风格。"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("光谱仪波长定标报告 (Wavelength Calibration Report)\n")
        f.write("=" * 60 + "\n")
        f.write(f"定标处理时间: {current_time}\n")
        f.write(f"数据源输入文件: {file_path}\n")
        f.write(f"选定光源类型: {lamp_name} ({lamp_description})\n")
        f.write(f"像素偏移偏置模式 (Shift Mode): {shift_mode}\n")
        f.write(f"锚点匹配容差: ±{MATCH_TOLERANCE} px\n")
        f.write(f"比例交叉验证容差: ±{RATIO_TOLERANCE:.0%}\n\n")

        f.write("------------------ 寻峰与拟合明细 ------------------\n")
        f.write(f"{'序号':<4}{'标准波长(nm)':<14}{'理论锚点':<10}"
                f"{'匹配质心(px)':<14}{'匹配距离':<10}{'标定波长(nm)':<16}"
                f"{'残差(nm)':<12}质量检测\n")

        for i, m in enumerate(matches):
            fitted_wl = calibration_result.fitted_wavelengths[i]
            resid     = calibration_result.residuals_nm[i]
            status    = "通过" if m.passed_all else f"标记: {'; '.join(m.fail_reasons)}"
            f.write(f"{i+1:<6}"
                    f"{true_wavelengths[i]:<16.3f}"
                    f"{m.anchor_pixel:<12.1f}"
                    f"{m.matched_centroid:<16.3f}"
                    f"{m.distance:<12.3f}"
                    f"{fitted_wl:<18.4f}"
                    f"{resid:+.4f}  {status}\n")

        n_flagged = sum(1 for m in matches if not m.passed_all)
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
# 主流程
# ==============================================================================

def run_wavelength_calibration():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    try:
        # ── 1. 选择文件 ──────────────────────────────────────────────────────
        file_path = select_csv_file()
        dir_name  = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # ── 2. 读取数据（忽略第一列内容，按行号生成像素索引）──────────────────
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] >= 2:
            intensity = np.asarray(df.iloc[:, 1].values, dtype=float)
        else:
            intensity = np.asarray(df.iloc[:, 0].values, dtype=float)
        # 像素索引始终由行号生成，不依赖文件中的任何列
        # （find_peaks 内部本身就用数组下标，此处仅为显式说明，不需要单独变量）

        # ── 3. 判定光源类型 ──────────────────────────────────────────────────
        lamp_name = resolve_lamp_name(file_path)
        lamp      = get_lamp_config(lamp_name)

        # ── 4. 全局 Shift 预估（粗筛，提高鲁棒性）──────────────────────────────
        if MANUAL_SHIFT_OVERRIDE is not None:
            global_shift = int(MANUAL_SHIFT_OVERRIDE)
            shift_mode   = f"手动指定值 ({global_shift:+d} px)"
        else:
            global_shift = auto_estimate_global_shift(
                intensity, lamp.auto_shift_anchor, lamp.shift_search_radius,
            )
            shift_mode = f"系统自动评估 ({global_shift:+d} px)"

        shifted_anchors = [a + global_shift for a in lamp.anchor_pixels]

        # ── 5. 全谱完整寻峰，拿到全部候选峰质心（不论是否通过质量检测）──────────
        log_path = os.path.join(dir_name, f"{base_name}_peakfind_log.txt")
        pipeline_result = run_full_pipeline(intensity, log_path)
        candidate_centroids = pipeline_result.centroids
        candidate_passed    = [p.passed_all   for p in pipeline_result.peaks]
        candidate_reasons   = [p.fail_reasons for p in pipeline_result.peaks]

        if len(candidate_centroids) == 0:
            raise RuntimeError(
                "全谱寻峰未找到任何候选峰，定标无法进行。"
                f"详细检测记录请查看: {log_path}"
            )

        # ── 6. 锚点匹配（找不到立即报错，不做任何静默兜底）──────────────────────
        # 注：匹配池包含全部候选峰，不论是否通过质量检测；匹配到的峰若存在
        # 质量标记会在报告中标注，由人工判断是否采用该定标结果
        matches = match_anchors_to_peaks(
            anchor_pixels      = shifted_anchors,
            peak_centroids     = candidate_centroids,
            match_tolerance    = MATCH_TOLERANCE,
            peak_passed_flags  = candidate_passed,
            peak_fail_reasons  = candidate_reasons,
        )

        # ── 7. 比例交叉验证（排除误匹配到邻近峰的情况）──────────────────────────
        verify_anchor_ratios(matches, tolerance=RATIO_TOLERANCE)

        # ── 8. 多项式拟合 ────────────────────────────────────────────────────
        matched_centroids = [m.matched_centroid for m in matches]
        calibration_result = calibrate(
            centroids   = matched_centroids,
            wavelengths = lamp.true_wavelengths,
            degree      = CALIBRATION_DEGREE,
        )

        # ── 9. 导出全谱定标波长 CSV ───────────────────────────────────────────
        all_pixels = np.arange(len(intensity))
        all_wavelengths = pixel_to_wavelength(all_pixels, calibration_result)

        calibrated_csv_path = os.path.join(dir_name, f"{base_name}_calibrated.csv")
        out_df = pd.DataFrame({
            "Wavelength_nm": all_wavelengths,
            "Intensity"    : intensity,
        })
        out_df.to_csv(calibrated_csv_path, index=False, header=False)

        # ── 10. 生成报告 ─────────────────────────────────────────────────────
        report_path = os.path.join(dir_name, f"{base_name}_wavecal_report.txt")
        generate_report(
            report_path        = report_path,
            file_path           = file_path,
            lamp_name           = lamp_name,
            lamp_description     = lamp.description,
            shift_mode           = shift_mode,
            matches              = matches,
            true_wavelengths      = lamp.true_wavelengths,
            calibration_result    = calibration_result,
        )

        # ── 11. 成功提示 ─────────────────────────────────────────────────────
        n_flagged = sum(1 for m in matches if not m.passed_all)
        flag_note = f"\n⚠ {n_flagged} 个匹配点存在质量标记，详见报告" if n_flagged > 0 else ""
        messagebox.showinfo(
            "波长定标成功",
            f"定标任务顺利完成！\n\n"
            f"光源类型: {lamp_name}\n"
            f"使用锚点数: {len(matches)}\n"
            f"RMS残差: {calibration_result.rms_nm:.4f} nm\n"
            f"最大残差: {calibration_result.max_resid_nm:.4f} nm"
            f"{flag_note}\n\n"
            f"已导出:\n"
            f"1. {base_name}_calibrated.csv\n"
            f"2. {base_name}_wavecal_report.txt\n"
            f"3. {base_name}_peakfind_log.txt",
        )
        print(f"定标成功完成，结果已保存至: {dir_name}")

    except (AnchorMatchError, AnchorRatioError) as e:
        messagebox.showerror("定标中止：锚点匹配/验证失败", str(e))
        print(f"定标中止: {e}", file=sys.stderr)

    except SystemExit as e:
        print(f"已取消: {e}")

    except Exception as e:
        messagebox.showerror("定标中止：未预期的错误", str(e))
        print(f"定标中止（未预期错误）: {e}", file=sys.stderr)
        raise

    finally:
        root.destroy()


if __name__ == "__main__":
    run_wavelength_calibration()