"""
config.py — 全局参数管理
职责：集中管理所有可调参数，避免魔法数字散落在各模块
使用方式：
    from spectra_tools.config import Config
    cfg = Config()                        # 使用默认值
    cfg = Config(min_snr=20)              # 覆盖单个参数
    cfg = Config.from_dict({...})         # 从字典加载
    cfg = Config.from_json("cfg.json")    # 从 JSON 文件加载
    cfg.save_json("cfg.json")             # 保存当前配置

参数分组：
    [baseline]    基线估计
    [peak]        寻峰与边界判定
    [symmetry]    对称性检验
    [calibration] 波长定标
    [explorer]    探索接口
    [io]          输入输出路径
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Config:

    # ── 基线估计 ─────────────────────────────────────────────────────────────
    # 滚动窗口宽度（px）
    # 依据：应远大于最宽峰（~40 px），远小于探测器总长（2048 px）
    # 典型值：100 px
    baseline_window_size : int   = 100

    # 取窗口内第几百分位数作为基线
    # 依据：基线平坦时 10 足够；有背景荧光时可提高到 20
    baseline_percentile  : float = 10.0

    # ── 寻峰 ─────────────────────────────────────────────────────────────────
    # 最小信噪比（相对噪声标准差）
    # 依据：NIST 光谱定标指南，SNR < 10 时质心误差 > 0.1 px
    min_snr              : float = 10.0

    # 最小峰间距（px）
    # 依据：小于两峰各自 FWHM 之和时发生混叠；保守取 20 px
    # 2048 px 探测器上 HgAr 约 20~30 条线，平均间距 ~70 px，20 px 是合理下限
    min_peak_sep         : int   = 20

    # ── 边界判定 ─────────────────────────────────────────────────────────────
    # 边界阈值：intensity <= baseline + N × noise_std 视为回落到基线
    # 依据：3σ 是统计学上区分信号与噪声的标准界限
    boundary_n_sigma     : float = 3.0

    # 连续满足边界条件的点数才确认边界
    # 依据：参考 ESO DRS pipeline，3 点连续判断可抵抗单点噪声尖刺
    #       对 FWHM 3~8 px 的峰，3 点约占半宽的 30~50%，物理上合理
    boundary_consec      : int   = 3

    # 谷底（鞍点）判定宽容倍数
    # 依据：实际光谱中常见主峰附近紧贴更弱的伴峰，两者之间会形成一个
    #       局部谷底而非真正回到基线。若谷底值 <= saddle_n_sigma × 局部阈值，
    #       视为"足够接近基线"，可作为边界；否则判定为双峰连叠仍未分开。
    #       1.5~2.0 是常见起点，需结合实际数据中谷底/阈值的真实比值调整
    #       （过小会让真正连叠的峰被误判为分开；过大会让该舍弃的弱峰被保留）
    saddle_n_sigma        : float = 1.5

    # ── 硬宽度限制（config 级保护，独立于自适应边界）────────────────────────
    # 单侧最小半宽（px）
    # 依据：< 3 px 时质心精度不足（加权点太少），可能是噪声尖刺
    min_half_width       : int   = 3

    # 单侧最大半宽（px）
    # 依据：参考 NIST / ESO 规程，FWHM 上限约 2~3 倍典型值
    #       典型 FWHM 3~8 px，保守上限取 40 px；
    #       如果仪器分辨率低或峰很宽，可适当放大到 60~80
    max_half_width       : int   = 40

    # ── 对称性检验 ───────────────────────────────────────────────────────────
    # 有效峰对称性阈值（归一化三阶矩 |skewness|）
    # 依据：ASTM E1655 / 工业近红外领域常用 0.5 作为可接受上限
    #       HgAr 谱线物理上极窄且对称，0.5 已足够宽松
    skewness_threshold   : float = 0.5

    # 核心窗口阈值比例（质心/对称性计算只在此区域内进行）
    # 依据：紧贴主峰的弱伴峰残留常滞留在裙边（峰高 < 5%的区域），
    #       这部分会让质心和skewness产生系统性偏移，但不代表主峰本身
    #       不对称。0.05（5%）是常见起点，伴峰干扰严重时可适当提高
    core_height_ratio    : float = 0.05

    # ── 定标 ─────────────────────────────────────────────────────────────────
    # 波长定标多项式阶数
    # 依据：CT 光谱仪色散非线性，三次多项式通常可将残差压到 < 0.05 nm
    #       点数不足时降到 2；高精度场合可升到 4（需更多定标点）
    calibration_degree   : int   = 3

    # ── 探索接口（explorer.py）──────────────────────────────────────────────
    # 推荐定标峰的更高 SNR 门槛（仅影响评分，不过滤）
    # 依据：定标峰要求比普通有效峰更高，50 对应约 5× min_snr
    explorer_snr_thr     : float = 50.0

    # 推荐定标峰的更严对称性门槛（仅影响评分，不过滤）
    explorer_skew_thr    : float = 0.3

    # 推荐定标峰数量
    explorer_n_suggest   : int   = 8

    # 覆盖均匀性权重（0=纯质量评分，1=纯空间覆盖）
    explorer_coverage_weight: float = 0.4

    # ── 输入输出 ─────────────────────────────────────────────────────────────
    # 日志文件路径
    log_path             : str   = "log.txt"

    # 峰质量报告 CSV 路径（空字符串表示不保存）
    quality_csv_path     : str   = "peak_quality.csv"

    # 探测器像素数（用于 wavelength_to_pixel 和覆盖评分）
    detector_size        : int   = 2048

    # ── 工厂方法 ─────────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """从字典加载，忽略未知键，方便部分覆盖。"""
        valid = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """从 JSON 文件加载配置。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def save_json(self, path: str) -> None:
        """将当前配置保存为 JSON 文件（含注释版本见下方 save_annotated）。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        print(f"[config] 配置已保存: {path}")

    def save_annotated(self, path: str) -> None:
        """
        保存带注释的配置文件（JSON 不支持注释，改用 .jsonc 风格纯文本）。
        方便人工阅读和修改。
        """
        lines = [
            "// spectra_tools 配置文件",
            "// 修改后重命名为 .json 即可用 Config.from_json() 加载",
            "{",
        ]
        annotations = {
            "baseline_window_size" : "基线滚动窗口宽度（px），建议 >> 峰宽",
            "baseline_percentile"  : "基线百分位数，基线平坦时用 10",
            "min_snr"              : "最小信噪比，NIST 建议 >= 10",
            "min_peak_sep"         : "最小峰间距（px），防止混叠",
            "boundary_n_sigma"     : "边界判定阈值（N × noise_std）",
            "boundary_consec"      : "连续满足边界条件的点数，推荐 3",
            "saddle_n_sigma"       : "谷底判定宽容倍数，推荐 1.5~2.0",
            "min_half_width"       : "单侧最小半宽（px），< 3 视为噪声尖刺",
            "max_half_width"       : "单侧最大半宽（px），宽峰仪器可放大到 60~80",
            "skewness_threshold"   : "对称性阈值，|skewness| > 此值则舍弃",
            "core_height_ratio"    : "核心窗口阈值比例，质心/对称性只在此区域内计算，推荐0.05",
            "calibration_degree"   : "定标多项式阶数，通常 3",
            "explorer_snr_thr"     : "推荐定标峰的 SNR 评分门槛",
            "explorer_skew_thr"    : "推荐定标峰的对称性评分门槛",
            "explorer_n_suggest"   : "推荐定标峰数量",
            "explorer_coverage_weight": "空间覆盖权重 (0=纯质量, 1=纯覆盖)",
            "log_path"             : "日志文件路径",
            "quality_csv_path"     : "峰质量 CSV 路径，空字符串则不保存",
            "detector_size"        : "探测器像素总数",
        }
        d = asdict(self)
        items = list(d.items())
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            note  = annotations.get(k, "")
            val   = json.dumps(v, ensure_ascii=False)
            lines.append(f'  "{k}": {val}{comma}  // {note}')
        lines.append("}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[config] 带注释配置已保存: {path}")

    # ── 调试 ─────────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        lines = ["Config("]
        for k, v in asdict(self).items():
            lines.append(f"  {k:<30s} = {v!r}")
        lines.append(")")
        return "\n".join(lines)