"""
lamp_registry.py — 校准光源注册表
职责：集中管理不同光源（HgAr、HgNe等）的参考波长、理论锚点像素位置等
      光源专属配置，与具体的定标流程逻辑解耦。

新增一种光源时，只需要在 LAMP_REGISTRY 中添加一个条目，不需要改动
run_calibration.py 中的任何流程逻辑。

暴露接口：
    LAMP_REGISTRY              字典，key为光源名称，value为 LampConfig
    get_lamp_config(name)      根据名称获取配置，找不到则报错
    detect_lamp_from_filename(filename) -> str
        根据文件名猜测光源类型（可选功能，供 run_calibration.py 自动判断使用）
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LampConfig:
    """单个光源的完整配置。"""
    name              : str          # 光源名称，如 "HgAr"
    description       : str          # 说明文字
    true_wavelengths  : List[float]  # NIST 参考波长（nm），与 anchor_pixels 一一对应
    anchor_pixels     : List[float]  # 理论锚点像素位置（同一台仪器的经验值）
    auto_shift_anchor : float        # 用于全局 Shift 自动评估的基准锚点像素位置
    shift_search_radius: int = 50    # Shift 评估时的扫描半径（像素）

    def __post_init__(self):
        if len(self.true_wavelengths) != len(self.anchor_pixels):
            raise ValueError(
                f"光源 [{self.name}] 配置错误: true_wavelengths "
                f"({len(self.true_wavelengths)}个) 和 anchor_pixels "
                f"({len(self.anchor_pixels)}个) 数量不一致"
            )


# ── 光源注册表 ────────────────────────────────────────────────────────────────
# 新增光源时，在此字典中添加新条目即可，无需改动其他任何文件

LAMP_REGISTRY = {
    "HgAr": LampConfig(
        name              = "HgAr",
        description       = "汞氩灯 (HgAr) - 覆盖 350-1100 nm 宽谱段精选特征点",
        true_wavelengths  = [
            435.833, 546.074, 696.543, 727.294, 763.511,
            794.818, 826.452, 852.144, 866.794, 922.45, 965.778,
        ],
        anchor_pixels     = [
            169, 437, 840, 928, 1036,
            1133, 1234, 1320, 1370, 1571, 1741,
        ],

        # 004: anchor_pixels     = [202, 477, 884, 975, 1085,1187, 1289, 1376, 1428, 1631, 1809],

        auto_shift_anchor  = 437,   # 546.074 nm 强独立特征峰，用于全局 Shift 评估
        shift_search_radius= 50,
    ),
    # 未来新增光源示例（取消注释并填入真实数据即可启用）：
    # "HgNe": LampConfig(
    #     name               = "HgNe",
    #     description        = "汞氖灯 (HgNe)",
    #     true_wavelengths   = [...],
    #     anchor_pixels      = [...],
    #     auto_shift_anchor  = ...,
    #     shift_search_radius= 50,
    # ),
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