"""
spectra_tools — CT 光谱仪光谱处理工具包
核心流程：
    from spectra_tools import run_pipeline, run_explorer, Config
    result = run_explorer(intensity, Config())          # 第一步：探索
    result = run_pipeline(intensity, Config(),          # 第二步：定标
                calibration_pairs=[(px, nm), ...])

各模块独立导入：
    from spectra_tools.baseline    import estimate_baseline
    from spectra_tools.peak_finder import find_peaks, extract_peak_window, \
                                          compute_centroid, compute_skewness
    from spectra_tools.calibration import calibrate, pixel_to_wavelength, \
                                          match_anchors_to_peaks, verify_anchor_ratios
    from spectra_tools.explorer    import peak_quality_report
    from spectra_tools.logger      import Logger, RejectReason
    from spectra_tools.config      import Config
"""

from .pipeline    import run_pipeline, run_explorer, PipelineResult
from .config      import Config
from .logger      import Logger, RejectReason
from .baseline    import estimate_baseline, estimate_local_noise, snr
from .peak_finder import (find_peaks, PeakResult,
                           extract_peak_window,
                           compute_centroid, compute_sigma, compute_skewness)
from .calibration import (calibrate, pixel_to_wavelength, wavelength_to_pixel,
                           print_calibration_report, evaluate_fit_quality,
                           CalibrationResult,
                           AnchorMatch, match_anchors_to_peaks, verify_anchor_ratios,
                           AnchorMatchError, AnchorRatioError)
from .explorer    import (peak_quality_report, suggest_calibration_peaks,
                           print_quality_table, save_quality_csv, PeakQuality)

__all__ = [
    # pipeline
    "run_pipeline", "run_explorer", "PipelineResult",
    # config
    "Config",
    # logger
    "Logger", "RejectReason",
    # baseline
    "estimate_baseline", "estimate_local_noise", "snr",
    # peak_finder
    "find_peaks", "PeakResult",
    "extract_peak_window", "compute_centroid", "compute_sigma", "compute_skewness",
    # calibration
    "calibrate", "pixel_to_wavelength", "wavelength_to_pixel",
    "print_calibration_report", "evaluate_fit_quality", "CalibrationResult",
    "AnchorMatch", "match_anchors_to_peaks", "verify_anchor_ratios",
    "AnchorMatchError", "AnchorRatioError",
    # explorer
    "peak_quality_report", "suggest_calibration_peaks",
    "print_quality_table", "save_quality_csv", "PeakQuality",
]