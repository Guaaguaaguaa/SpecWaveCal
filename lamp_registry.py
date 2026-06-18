"""
lamp_registry.py — 校准光源注册表
职责：集中管理不同光源（HgAr、HgNe等）的参考波长配置，与具体的定标
      流程逻辑解耦。

设计变更（RANSAC 自动识别后）：
    引入 RANSAC 自动谱线识别（ransac_matcher.py）后，不再需要为每台
    具体仪器单独录入 anchor_pixels（理论锚点像素位置）——RANSAC 只需要
    true_wavelengths（光源本身的物理属性，与仪器无关）就能自动识别出
    候选峰与已知波长的对应关系，对不同通道数、不同批次的整体平移/缩放、
    分辨率差异都有鲁棒性。

    anchor_pixels 和 auto_shift_anchor 字段保留为可选项，仅用于：
    - 旧版手动锚点匹配流程的向后兼容（match_anchors_to_peaks 仍可使用）
    - 在 RANSAC 识别失败时，作为人工排查的参考基准
    新增光源时，只需提供 true_wavelengths 即可启用 RANSAC 自动识别流程，
    不强制要求填写 anchor_pixels。

新增一种光源时，只需要在 LAMP_REGISTRY 中添加一个条目，不需要改动
run_calibration.py 中的任何流程逻辑。

暴露接口：
    LAMP_REGISTRY              字典，key为光源名称，value为 LampConfig
    get_lamp_config(name)      根据名称获取配置，找不到则报错
    detect_lamp_from_filename(filename) -> str
        根据文件名猜测光源类型（可选功能，供 run_calibration.py 自动判断使用）
    INSTRUMENT_TEMPLATES       已知仪器型号的锚点位置模板库（仅供人工参考，
                                不参与 RANSAC 自动识别流程本身）
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LampConfig:
    """单个光源的完整配置。"""
    name              : str          # 光源名称，如 "HgAr"
    description       : str          # 说明文字
    true_wavelengths  : List[float]  # NIST 参考波长（nm）— RANSAC 自动识别
                                      # 流程的唯一必需输入，与具体仪器无关

    # ── 以下字段可选，仅用于旧版手动匹配流程的向后兼容 ──────────────────────
    anchor_pixels      : Optional[List[float]] = None  # 某台参考仪器的
                                                         # 理论锚点像素位置
    auto_shift_anchor   : Optional[float] = None        # 全局 Shift 自动
                                                         # 评估的基准锚点
    shift_search_radius : int = 50                      # Shift 评估扫描半径

    # ── RANSAC 识别相关的光源专属推荐参数 ────────────────────────────────────
    ransac_tolerance_nm: float = 2.0   # 该光源谱线密度下推荐的 RANSAC 容差
                                        # 谱线越密集，应设得越小，避免误匹配

    def __post_init__(self):
        if self.anchor_pixels is not None and \
           len(self.true_wavelengths) != len(self.anchor_pixels):
            raise ValueError(
                f"光源 [{self.name}] 配置错误: true_wavelengths "
                f"({len(self.true_wavelengths)}个) 和 anchor_pixels "
                f"({len(self.anchor_pixels)}个) 数量不一致"
            )


# ── 光源注册表 ────────────────────────────────────────────────────────────────
# 新增光源时，在此字典中添加新条目即可，无需改动其他任何文件
# 注：anchor_pixels 在此仅保留 "004" 这台仪器的数据作为旧版兼容/调试参考，
#     RANSAC 自动识别流程不依赖此字段

LAMP_REGISTRY = {
    "HgAr": LampConfig(
        name              = "HgAr",
        description       = "汞氩灯 (HgAr) - 覆盖 350-1100 nm 宽谱段精选特征点",
        true_wavelengths  = [ 435.833, 546.074, 696.543, 727.294, 763.511, 794.818, 826.452, 852.144,  866.794, 922.45, 965.778],
        # 以下为 "004" 仪器的参考值，仅供旧版手动匹配/调试使用
        anchor_pixels      = [
            202, 477, 884, 975, 1085,
            1187, 1289, 1376, 1428, 1596, 1809,
        ],
        auto_shift_anchor   = 477,
        shift_search_radius = 50,
        ransac_tolerance_nm = 2.0,
    ),
    # 未来新增光源示例（取消注释并填入真实数据即可启用，anchor_pixels 可省略）：
    # "HgNe": LampConfig(
    #     name               = "HgNe",
    #     description        = "汞氖灯 (HgNe)",
    #     true_wavelengths   = [...],
    #     ransac_tolerance_nm= 2.0,
    # ),
}


# ── 已知仪器型号模板库 ──────────────────────────────────────────────────────────
# 仅供人工参考、调试比对使用，不参与 RANSAC 自动识别流程。
# 这些数据曾用于早期"手动 anchor_pixels"方案，现保留作为：
#   1. 验证 RANSAC 识别结果是否与已知良好仪器的模式吻合的参考基准
#   2. 未来若 RANSAC 识别失败，人工排查时的对照数据
#
# 结构: { 型号名: [11个锚点像素位置，与 LAMP_REGISTRY["HgAr"].true_wavelengths
#                  按升序一一对应] }

INSTRUMENT_TEMPLATES = {
    "004"      : [204, 477, 884, 975, 1085, 1187, 1289, 1376, 1428, 1596, 1809],
    "007"      : [169, 437, 840, 928, 1036, 1133, 1234, 1320, 1370, 1571, 1741],
    "TF-26001" : [148, 419, 828, 920, 1029, 1129, 1233, 1320, 1373, 1581, 1758],
    "TF-26002" : [200, 470, 874, 963, 1072, 1169, 1272, 1358, 1409, 1610, 1782],
    "TF-2506"  : [128, 397, 801, 890, 998, 1095, 1198, 1283, 1333, 1534, 1704],
}


# ── 查询接口 ──────────────────────────────────────────────────────────────────

def get_lamp_config(name: str) -> LampConfig:
    """
    根据光源名称获取配置。

    Raises
    ------
    KeyError : 注册表中找不到对应名称时抛出，附带可用光源列表提示
    """
    if name not in LAMP_REGISTRY:
        available = ", ".join(LAMP_REGISTRY.keys())
        raise KeyError(
            f"未在光源注册表中找到 [{name}]。"
            f"当前已注册的光源: {available}。"
            f"如需新增光源，请在 lamp_registry.py 的 LAMP_REGISTRY 中添加条目。"
        )
    return LAMP_REGISTRY[name]


def detect_lamp_from_filename(filename: str) -> Optional[str]:
    """
    尝试根据文件名猜测光源类型（简单子串匹配，大小写不敏感）。

    Parameters
    ----------
    filename : 文件名或完整路径

    Returns
    -------
    匹配到的光源名称；若文件名中找不到任何已注册光源名称的子串，返回 None
    （调用方应对 None 结果做处理，例如要求用户手动指定，而不是静默猜测）
    """
    lower_name = filename.lower()
    for lamp_name in LAMP_REGISTRY.keys():
        if lamp_name.lower() in lower_name:
            return lamp_name
    return None

 