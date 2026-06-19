"""
run_calibration.py — 波长定标主流程脚本
职责：选择数据文件 -> 判定光源类型 -> 全谱寻峰 -> RANSAC自动识别谱线对应关系
      -> 多项式拟合 -> 导出结果与报告

与具体光源相关的配置（参考波长）放在 lamp_registry.py 中，本脚本只负责
流程编排。新增光源类型只需在 lamp_registry.py 添加 true_wavelengths，
不需要为每台具体仪器录入锚点像素位置——RANSAC 自动识别取代了这一步。

使用方式：
    直接运行: python run_calibration.py
    会弹出文件选择窗口，选择 CSV 后自动执行完整定标流程。

设计原则（务必遵守）：
    任何环节出现异常（RANSAC 找不到足够内点等），立即弹窗报错并中止整个
    流程，不做任何静默兜底或退化处理。

设计说明（RANSAC 取代手动锚点匹配后的流程变化）：
    旧流程：读取注册表 anchor_pixels → 估算全局 Shift → 在锚点附近
            ±5px 窗口内查找 → match_anchors_to_peaks 做容差匹配 →
            verify_anchor_ratios 比例验证 → 多项式拟合
    新流程：全谱寻峰得到全部候选峰 → RANSAC 在候选峰与已知波长之间
            自动识别对应关系（不需要任何仪器专属的锚点先验）→
            多项式拟合（直接用 RANSAC 输出的质心-波长对应关系）
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
    calibrate, pixel_to_wavelength, print_calibration_report,
    ransac_match_wavelengths, print_ransac_report,
    RansacMatchError,
)
from lamp_registry import get_lamp_config, detect_lamp_from_filename, LAMP_REGISTRY


# ==============================================================================
# 全局工程配置参数（可根据实际实验条件调整）
# ==============================================================================
RANSAC_ITERATIONS       = 10000   # RANSAC 总迭代次数（v4: 5 轮 × 2000 次/轮，
                                   # 配合空间约束+覆盖约束+多轮投票保证收敛）
RANSAC_MIN_INLIERS_RATIO = 0.5   # 最少需识别出已知波长总数的此比例，否则报错
                                   # 二次模型残差更低但物理上允许适度放宽
CALIBRATION_DEGREE      = 3      # 定标多项式阶数


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


def run_full_pipeline(intensity: np.ndarray, log_path: str):
    """跑完整寻峰流程，返回 PipelineResult（含全部候选峰，不论是否通过质量检测）。"""
    cfg = Config(log_path=log_path, quality_csv_path="")
    result = run_explorer(intensity, cfg, save_csv=False)
    return result


def generate_report(
    report_path        : str,
    file_path           : str,
    lamp_name           : str,
    lamp_description     : str,
    ransac_result        ,
    calibration_result    ,
) -> None:
    """生成定标质量报告文本文件。"""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inliers = sorted(ransac_result.inliers, key=lambda x: x.true_wavelength)
    # 构建质心→质量标记的查找表
    centroid_to_quality = {}
    if hasattr(ransac_result, '_centroid_to_quality'):
        centroid_to_quality = ransac_result._centroid_to_quality

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
            passed, reasons = centroid_to_quality.get(
                round(px, 6), (True, [])
            )
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

        # ── 3. 判定光源类型 ──────────────────────────────────────────────────
        lamp_name = resolve_lamp_name(file_path)
        lamp      = get_lamp_config(lamp_name)

        # ── 4. 全谱完整寻峰，拿到全部候选峰（不论是否通过质量检测）─────────────
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

        # ── 5. RANSAC 自动识别候选峰与已知波长的对应关系 ───────────────────────
        # 不需要任何仪器专属的锚点像素先验，对通道数、整体平移、分辨率
        # 差异均有鲁棒性
        min_inliers = max(2, int(np.ceil(len(lamp.true_wavelengths)
                                          * RANSAC_MIN_INLIERS_RATIO)))
        ransac_result = ransac_match_wavelengths(
            candidate_centroids = candidate_centroids,
            true_wavelengths    = lamp.true_wavelengths,
            tolerance_nm        = lamp.ransac_tolerance_nm,
            n_iterations        = RANSAC_ITERATIONS,
            min_inliers         = min_inliers,
        )
        print_ransac_report(ransac_result)

        # ── 6. 多项式拟合（直接用 RANSAC 输出的质心-波长对应关系）─────────────
        pairs = ransac_result.calibration_pairs
        calibration_result = calibrate(
            centroids   = [p[0] for p in pairs],
            wavelengths = [p[1] for p in pairs],
            degree      = CALIBRATION_DEGREE,
        )
        print_calibration_report(calibration_result, matches=None)

        # ── 7. 导出全谱定标波长 CSV ───────────────────────────────────────────
        all_pixels = np.arange(len(intensity))
        all_wavelengths = pixel_to_wavelength(all_pixels, calibration_result)

        calibrated_csv_path = os.path.join(dir_name, f"{base_name}_calibrated.csv")
        out_df = pd.DataFrame({
            "Wavelength_nm": all_wavelengths,
            "Intensity"    : intensity,
        })
        out_df.to_csv(calibrated_csv_path, index=False, header=False)

        # ── 8. 构建质心→质量标记查找表，附到 ransac_result 供报告使用 ─────
        centroid_to_quality = {}
        for c, passed, reasons in zip(
            candidate_centroids, candidate_passed, candidate_reasons
        ):
            centroid_to_quality[round(c, 6)] = (passed, reasons)
        ransac_result._centroid_to_quality = centroid_to_quality

        # ── 9. 生成报告 ─────────────────────────────────────────────────────
        report_path = os.path.join(dir_name, f"{base_name}_wavecal_report.txt")
        generate_report(
            report_path        = report_path,
            file_path           = file_path,
            lamp_name           = lamp_name,
            lamp_description     = lamp.description,
            ransac_result        = ransac_result,
            calibration_result   = calibration_result,
        )

        # ── 10. 成功提示 ─────────────────────────────────────────────────────
        n_flagged = sum(
            1 for inl in ransac_result.inliers
            if not centroid_to_quality.get(round(inl.centroid_pixel, 6), (True, []))[0]
        )
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

    except RansacMatchError as e:
        messagebox.showerror("定标中止：RANSAC 识别失败", str(e))
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
    