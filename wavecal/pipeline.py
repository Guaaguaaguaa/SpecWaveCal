"""
pipeline.py — 主流程组装
职责：按顺序调用各模块，完成从原始光谱到波长定标的完整流程
暴露接口：
    run_pipeline(intensity, config)
        -> PipelineResult
    run_explorer(intensity, config)
        -> 仅做峰分析和质量报告，不做定标（首次建立方案时使用）

PipelineResult 包含：
    peaks       : list[PeakResult]      所有有效峰
    qualities   : list[PeakQuality]     峰质量报告
    calibration : CalibrationResult     定标结果（若已提供参考波长）
    baseline    : np.ndarray            逐像素基线
    noise_std   : float                 噪声标准差
    I_corr      : np.ndarray            扣基线后强度

典型使用流程：
    第一步（探索）：
        result = run_explorer(intensity, cfg)
        # 查看 result.qualities，手动选取定标峰像素
        # 去 NIST 查对应波长，填入 calibration_pairs

    第二步（定标）：
        result = run_pipeline(intensity, cfg,
                     calibration_pairs=[(centroid, wavelength), ...])
        print_calibration_report(result.calibration)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .config      import Config
from .logger      import Logger, RejectReason
from .baseline    import estimate_baseline, estimate_local_noise, snr as compute_snr
from .peak_finder import find_peaks, PeakResult
from .explorer    import peak_quality_report, suggest_calibration_peaks, \
                         print_quality_table, save_quality_csv, PeakQuality
from .calibration import calibrate, print_calibration_report, \
                         CalibrationResult, pixel_to_wavelength


# ── 结果数据类 ────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    peaks       : List[PeakResult]
    qualities   : List[PeakQuality]
    baseline    : np.ndarray
    noise_std   : float
    local_noise : np.ndarray
    I_corr      : np.ndarray
    calibration : Optional[CalibrationResult] = None

    # 便捷属性
    @property
    def centroids(self) -> List[float]:
        return [p.centroid for p in self.peaks]

    def wavelengths_of_peaks(self) -> Optional[np.ndarray]:
        """若已定标，返回各有效峰对应波长；否则返回 None。"""
        if self.calibration is None:
            return None
        return pixel_to_wavelength(self.centroids, self.calibration)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run_pipeline(
    intensity           : np.ndarray,
    config              : Optional[Config] = None,
    calibration_pairs   : Optional[List[Tuple[float, float]]] = None,
    print_report        : bool = True,
    save_csv            : bool = True,
) -> PipelineResult:
    """
    完整流程：基线 → 寻峰 → 质量报告 → 波长定标（可选）

    Parameters
    ----------
    intensity           : 原始强度数组，shape (N,)
    config              : Config 实例，None 则使用默认值
    calibration_pairs   : [(centroid_px, wavelength_nm), ...]
                          手动指定的像素-波长对应关系
                          None 则跳过定标步骤
    print_report        : 是否在终端打印定标报告
    save_csv            : 是否保存峰质量 CSV

    Returns
    -------
    PipelineResult
    """
    cfg = config or Config()
    intensity = np.asarray(intensity, dtype=float)

    with Logger(cfg.log_path) as log:

        # ── Step 1：基线估计 ──────────────────────────────────────────────────
        log.info("Step 1: 基线估计")
        baseline, noise_std = estimate_baseline(
            intensity,
            window_size = cfg.baseline_window_size,
            percentile  = cfg.baseline_percentile,
        )
        I_corr = intensity - baseline
        log.info(f"  noise_std(全局)={noise_std:.3f}  "
                 f"baseline 范围 [{baseline.min():.1f}, {baseline.max():.1f}]")

        # ── Step 1b：局部噪声估计 ─────────────────────────────────────────────
        # 用于边界判定，自适应不同区域背景"干净/脏"程度的差异
        # （如强峰远翼、杂散光导致的局部背景结构性偏高）
        local_noise = estimate_local_noise(
            intensity,
            baseline,
            window_size = cfg.baseline_window_size,
        )
        log.info(f"  local_noise 范围 [{local_noise.min():.3f}, "
                 f"{local_noise.max():.3f}]（全局值 {noise_std:.3f}）")

        # ── Step 2：寻峰 + 边界 + 对称性 + 质心 ─────────────────────────────
        log.info("Step 2: 寻峰分析")
        peaks = find_peaks(
            intensity            = intensity,
            baseline             = baseline,
            noise_std            = noise_std,
            logger               = log,
            local_noise          = local_noise,
            min_snr              = cfg.min_snr,
            min_peak_sep         = cfg.min_peak_sep,
            boundary_n_sigma     = cfg.boundary_n_sigma,
            boundary_consec      = cfg.boundary_consec,
            saddle_n_sigma       = cfg.saddle_n_sigma,
            min_half_width       = cfg.min_half_width,
            max_half_width       = cfg.max_half_width,
            skewness_threshold   = cfg.skewness_threshold,
            core_height_ratio    = cfg.core_height_ratio,
        )

        # ── Step 3：质量报告 ──────────────────────────────────────────────────
        log.info("Step 3: 质量评估")
        qualities = peak_quality_report(
            peaks                        = peaks,
            I_corr                       = I_corr,
            snr_threshold_calibration    = cfg.explorer_snr_thr,
            skew_threshold_calibration   = cfg.explorer_skew_thr,
        )

        if save_csv and cfg.quality_csv_path:
            save_quality_csv(qualities, cfg.quality_csv_path)

        # 推荐定标峰（仅供参考，不强制使用）
        suggested = suggest_calibration_peaks(
            qualities        = qualities,
            n                = cfg.explorer_n_suggest,
            coverage_weight  = cfg.explorer_coverage_weight,
            detector_size    = cfg.detector_size,
        )

        if print_report:
            print_quality_table(qualities, highlight=suggested)

        # ── Step 4：波长定标（可选）──────────────────────────────────────────
        calibration = None
        if calibration_pairs:
            log.info(f"Step 4: 波长定标，使用 {len(calibration_pairs)} 个点")
            centroids_cal = [pair[0] for pair in calibration_pairs]
            wavelengths_cal = [pair[1] for pair in calibration_pairs]

            calibration = calibrate(
                centroids   = centroids_cal,
                wavelengths = wavelengths_cal,
                degree      = cfg.calibration_degree,
            )
            log.info(f"  RMS={calibration.rms_nm:.4f} nm  "
                     f"max_resid={calibration.max_resid_nm:.4f} nm")

            if print_report:
                print_calibration_report(calibration)
        else:
            log.info("Step 4: 跳过定标（未提供 calibration_pairs）")

        # ── 汇总 ─────────────────────────────────────────────────────────────
        # 注：peak_finder 不再剔除任何候选峰，total 即为最终返回的峰总数
        n_total  = len(peaks)
        n_passed = sum(1 for p in peaks if p.passed_all)
        log.summary(
            total    = n_total,
            used     = n_passed,
            rejected = n_total - n_passed,  # 此处含义为"存在质量标记"，非真正剔除
        )

    return PipelineResult(
        peaks       = peaks,
        qualities   = qualities,
        baseline    = baseline,
        noise_std   = noise_std,
        local_noise = local_noise,
        I_corr      = I_corr,
        calibration = calibration,
    )


def run_explorer(
    intensity : np.ndarray,
    config    : Optional[Config] = None,
    save_csv  : bool = True,
) -> PipelineResult:
    """
    仅做峰分析和质量报告，不做定标。
    首次建立定标方案时使用：看完质量表后，手动选取 calibration_pairs。

    使用示例：
        result = run_explorer(intensity, cfg)
        # 终端会打印质量表和推荐峰
        # 根据 result.qualities 中的 centroid 去查 NIST 波长
        # 然后调用 run_pipeline(intensity, cfg, calibration_pairs=[...])
    """
    return run_pipeline(
        intensity         = intensity,
        config            = config,
        calibration_pairs = None,
        print_report      = True,
        save_csv          = save_csv,
    )