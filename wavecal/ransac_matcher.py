"""
ransac_matcher.py — 基于 RANSAC 的自动谱线识别
职责：在不知道任何仪器专属锚点像素位置的前提下，仅依据"已知参考波长列表"
      和"全谱寻峰得到的候选峰质心列表"，自动识别出哪个候选峰对应哪条已知
      谱线，对仪器型号、通道数、分辨率差异、整体平移、强度顺序变化均免疫。

设计背景：
    不同批次/型号的同一光源（如 HgAr 灯），在不同光谱仪上测得的特征峰
    像素位置可能天差地别（如 2048 通道仪器上某峰在 1085px，512 通道
    仪器上可能在 270px），但物理上谱线之间的"相对几何关系"（在一阶
    近似下，像素与波长接近线性关系）是稳定的。RANSAC 利用这一点：
    随机抽取 2 个候选峰，假设它们对应 2 条已知波长，解出临时线性映射
    pixel = a*wavelength + b（或等价的 wavelength = a'*pixel + b'），
    再用这个映射去检验其余候选峰能否与其余已知波长对应上（容差内）。
    重复多次随机抽样，选择"内点"（成功对应）数量最多的一组映射作为
    最终识别结果。

    RANSAC 对以下情况天然免疫，不需要任何额外的特殊处理：
    - 高分辨率仪器把某条线分裂成双峰（多出来的峰自动被归为外点）
    - 低分辨率仪器漏检了某条弱线（缺失峰不影响其余内点的识别）
    - 不同批次仪器存在整体像素平移或轻微缩放（这正是 a, b 两个
      自由参数所建模的对象）

    RANSAC 阶段使用的线性模型只是"识别用的脚手架"，目的是快速确定
    候选峰与已知波长的对应关系，不是最终的定标结果。识别完成后，
    真正的精确定标仍交由 calibration.py 的 calibrate()（三次多项式）
    完成，二者职责不重叠。

暴露接口：
    RansacMatchResult                  — 识别结果数据类
    ransac_match_wavelengths(
        candidate_centroids, true_wavelengths,
        tolerance_nm, n_iterations, min_inliers, random_seed,
    ) -> RansacMatchResult
        执行 RANSAC 识别，返回最佳线性模型及内点对应关系。
        若找不到足够内点的模型，抛出 RansacMatchError（不做静默兜底）。

异常：
    RansacMatchError — 所有随机抽样迭代后仍找不到满足 min_inliers 的
                        模型时抛出，提示数据质量问题或参数需要调整。
"""

import itertools
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── 异常类型 ──────────────────────────────────────────────────────────────────

class RansacMatchError(Exception):
    """RANSAC 迭代后仍找不到满足最小内点数要求的模型时抛出。"""
    pass


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class RansacInlier:
    """单个内点对应关系：某个候选峰质心被识别为对应某条已知波长。"""
    centroid_pixel  : float   # 候选峰质心像素位置
    true_wavelength : float   # 对应的已知参考波长（nm）
    predicted_wavelength: float  # 用最佳模型从像素预测出的波长（nm）
    residual_nm     : float   # |predicted_wavelength - true_wavelength|


@dataclass
class RansacMatchResult:
    """RANSAC 识别的完整结果。"""
    a               : float              # wavelength = a * pixel + b 的斜率
    b               : float              # wavelength = a * pixel + b 的截距
    inliers         : List[RansacInlier] # 被识别为内点的对应关系
    outlier_centroids   : List[float]    # 候选峰中未被任何已知波长匹配上的质心
    unmatched_wavelengths: List[float]   # 已知波长中未被任何候选峰匹配上的波长
    n_iterations_used: int               # 实际执行的迭代次数
    inlier_tolerance_nm: float           # 本次识别使用的容差（nm）

    @property
    def matched_centroids(self) -> List[float]:
        """按已知波长升序排列的、已匹配候选峰质心列表（便于喂给 calibrate）。"""
        sorted_inliers = sorted(self.inliers, key=lambda x: x.true_wavelength)
        return [inl.centroid_pixel for inl in sorted_inliers]

    @property
    def matched_wavelengths(self) -> List[float]:
        """按波长升序排列的已知波长列表（与 matched_centroids 一一对应）。"""
        sorted_inliers = sorted(self.inliers, key=lambda x: x.true_wavelength)
        return [inl.true_wavelength for inl in sorted_inliers]


# ── 主入口 ────────────────────────────────────────────────────────────────────

def ransac_match_wavelengths(
    candidate_centroids : List[float],
    true_wavelengths    : List[float],
    tolerance_nm        : float = 2.0,
    n_iterations         : int   = 500,
    min_inliers          : Optional[int] = None,
    random_seed           : Optional[int] = None,
    require_positive_slope: bool = True,
) -> RansacMatchResult:
    """
    用 RANSAC 自动识别候选峰质心与已知参考波长的对应关系。

    算法流程：
        重复 n_iterations 次：
            1. 从 candidate_centroids 中随机抽取 2 个质心
            2. 从 true_wavelengths 中随机抽取 2 个波长（与抽取的质心一一配对，
               两种配对方式都尝试，因为不知道哪个质心对应哪个波长）
            3. 用这 2 对 (pixel, wavelength) 解出线性模型 wavelength = a*pixel+b
            4. 用该模型预测所有候选峰质心的波长，与已知波长列表比对，
               差值在 tolerance_nm 内的计为"内点"
            5. 记录内点数最多的模型
        迭代结束后，用内点数最多的模型重新做一次最小二乘精修（用全部内点，
        而不仅是最初抽样的2个点），得到最终的 (a, b)

    Parameters
    ----------
    candidate_centroids : 全谱寻峰得到的候选峰质心列表（像素），
                           不要求提前筛选质量好坏，RANSAC 自身会剔除外点
    true_wavelengths    : 已知参考波长列表（nm），如 NIST HgAr 谱线表
    tolerance_nm         : 内点判定容差（nm），默认 2.0
                            过小会导致正确对应也被误判为外点（尤其是
                            初始2点抽样恰好选到噪声质心时，模型本身偏差大）；
                            过大会引入错误对应。建议根据光源谱线密度调整：
                            谱线稀疏（如本场景11条线）可以适当放宽到2~3nm，
                            谱线密集场景需要收紧到0.5nm以内
    n_iterations         : 随机抽样迭代次数，默认 500
                            候选峰和波长数都不多时（数十量级），500次足够
                            高概率覆盖所有可能的"正确2点组合"
    min_inliers          : 最少需要识别出的内点数，默认为
                            len(true_wavelengths) 的 60%（向上取整），
                            低于此数视为识别失败
    random_seed           : 随机种子，便于复现实验结果；None 则不固定
    require_positive_slope: 是否强制要求斜率 a > 0，默认 True。
                            这是基于光谱仪物理先验的约束——像素索引增大时
                            波长应单调增大（CT 光栅光谱仪的标准布局）。
                            没有这条约束，RANSAC 在已知波长数据本身有误，
                            或候选峰质量很差时，可能找到一个数学上自洽但
                            方向颠倒（a<0）的错误模型，且内点数未必更少，
                            因此必须由物理先验过滤掉，不能仅靠内点数判断。
                            若仪器存在反向色散布局，应设为 False。

    Returns
    -------
    RansacMatchResult

    Raises
    ------
    RansacMatchError : 所有迭代后最佳模型的内点数仍低于 min_inliers，
                        或唯一找到的高内点模型违反 require_positive_slope
    ValueError        : 输入数据点数不足（候选峰或已知波长少于2个）
    """
    candidates = list(candidate_centroids)
    wavelengths = list(true_wavelengths)

    if len(candidates) < 2:
        raise ValueError(
            f"候选峰数量 ({len(candidates)}) 少于 2 个，无法进行 RANSAC 识别。"
        )
    if len(wavelengths) < 2:
        raise ValueError(
            f"已知波长数量 ({len(wavelengths)}) 少于 2 个，无法进行 RANSAC 识别。"
        )

    if min_inliers is None:
        min_inliers = max(2, int(np.ceil(len(wavelengths) * 0.6)))

    rng = random.Random(random_seed)

    best_a, best_b = None, None
    best_inlier_indices: List[Tuple[int, int]] = []  # (candidate_idx, wavelength_idx)

    candidates_arr   = np.asarray(candidates, dtype=float)
    wavelengths_arr  = np.asarray(wavelengths, dtype=float)

    for _ in range(n_iterations):
        # ── 随机抽 2 个候选峰 + 2 个已知波长 ─────────────────────────────────
        if len(candidates) < 2 or len(wavelengths) < 2:
            break
        c1, c2 = rng.sample(candidates, 2)
        w1, w2 = rng.sample(wavelengths, 2)

        if c1 == c2:
            continue  # 极端退化情况，跳过

        # 两种配对方式都尝试：(c1->w1, c2->w2) 和 (c1->w2, c2->w1)
        for (pw1, pw2) in [((c1, w1), (c2, w2)), ((c1, w2), (c2, w1))]:
            (px1, wl1), (px2, wl2) = pw1, pw2
            if px1 == px2:
                continue

            # 解线性方程 wavelength = a*pixel + b
            a = (wl2 - wl1) / (px2 - px1)
            b = wl1 - a * px1

            # 物理先验过滤：色散方向应为正（像素增大波长增大）
            if require_positive_slope and a <= 0:
                continue

            # ── 用该模型对全部候选峰投票，找内点 ─────────────────────────────
            predicted = a * candidates_arr + b  # shape: (n_candidates,)
            # 对每个候选峰，找与其预测波长最接近的已知波长
            diff_matrix = np.abs(predicted[:, None] - wavelengths_arr[None, :])
            nearest_wl_idx = np.argmin(diff_matrix, axis=1)
            nearest_wl_dist = diff_matrix[np.arange(len(candidates)), nearest_wl_idx]

            inlier_mask = nearest_wl_dist <= tolerance_nm
            # 同一条已知波长不能被多个候选峰同时匹配，保留误差最小的一个
            inlier_candidate_idx = np.where(inlier_mask)[0]
            wl_to_best_candidate = {}
            for c_idx in inlier_candidate_idx:
                wl_idx = int(nearest_wl_idx[c_idx])
                dist   = nearest_wl_dist[c_idx]
                if wl_idx not in wl_to_best_candidate or dist < wl_to_best_candidate[wl_idx][1]:
                    wl_to_best_candidate[wl_idx] = (int(c_idx), dist)

            n_inliers = len(wl_to_best_candidate)

            if n_inliers > len(best_inlier_indices):
                best_a, best_b = a, b
                best_inlier_indices = [
                    (c_idx, wl_idx) for wl_idx, (c_idx, _dist) in wl_to_best_candidate.items()
                ]

    if best_a is None or len(best_inlier_indices) < min_inliers:
        found = len(best_inlier_indices)
        raise RansacMatchError(
            f"RANSAC 识别失败：经过 {n_iterations} 次迭代，最佳模型只找到 "
            f"{found} 个内点，低于要求的最小内点数 {min_inliers}。"
            f"可能原因：候选峰质量太差、容差 tolerance_nm={tolerance_nm} "
            f"过严、或该谱与已知波长表本身不匹配。"
            f"建议检查候选峰列表是否合理，或适当放宽 tolerance_nm。"
        )

    # ── 用全部内点做最小二乘精修，而非仅用最初抽样的2个点 ───────────────────────
    inlier_pixels = np.array([candidates_arr[c_idx] for c_idx, _ in best_inlier_indices])
    inlier_wls    = np.array([wavelengths_arr[wl_idx] for _, wl_idx in best_inlier_indices])

    # 一次线性最小二乘： wavelength = a*pixel + b
    A = np.vstack([inlier_pixels, np.ones_like(inlier_pixels)]).T
    refined_a, refined_b = np.linalg.lstsq(A, inlier_wls, rcond=None)[0]

    if require_positive_slope and refined_a <= 0:
        raise RansacMatchError(
            f"RANSAC 识别失败：最佳内点集合经最小二乘精修后斜率为负 "
            f"(a={refined_a:.6f})，违反色散方向应为正的物理先验。"
            f"这通常说明候选峰质量太差或参考波长数据本身有误，"
            f"建议检查 true_wavelengths 是否准确，或扩大候选峰来源数据质量。"
        )

    # ── 组装最终结果 ─────────────────────────────────────────────────────────
    inliers: List[RansacInlier] = []
    matched_candidate_idx = set()
    matched_wl_idx = set()

    for c_idx, wl_idx in best_inlier_indices:
        px = float(candidates_arr[c_idx])
        true_wl = float(wavelengths_arr[wl_idx])
        pred_wl = float(refined_a * px + refined_b)
        inliers.append(RansacInlier(
            centroid_pixel       = px,
            true_wavelength      = true_wl,
            predicted_wavelength = pred_wl,
            residual_nm          = abs(pred_wl - true_wl),
        ))
        matched_candidate_idx.add(c_idx)
        matched_wl_idx.add(wl_idx)

    outlier_centroids = [
        float(candidates_arr[i]) for i in range(len(candidates))
        if i not in matched_candidate_idx
    ]
    unmatched_wavelengths = [
        float(wavelengths_arr[i]) for i in range(len(wavelengths))
        if i not in matched_wl_idx
    ]

    return RansacMatchResult(
        a                     = float(refined_a),
        b                     = float(refined_b),
        inliers               = inliers,
        outlier_centroids     = outlier_centroids,
        unmatched_wavelengths = unmatched_wavelengths,
        n_iterations_used     = n_iterations,
        inlier_tolerance_nm   = tolerance_nm,
    )


def print_ransac_report(result: RansacMatchResult) -> None:
    """打印 RANSAC 识别报告，便于人工核查识别质量。"""
    sep = "-" * 70
    print(sep)
    print("  RANSAC 自动谱线识别报告")
    print(sep)
    print(f"  线性模型: wavelength = {result.a:.6f} * pixel + {result.b:.4f}")
    print(f"  识别到内点数: {len(result.inliers)}")
    print(f"  外点候选峰数: {len(result.outlier_centroids)}")
    print(f"  未匹配已知波长数: {len(result.unmatched_wavelengths)}")
    print()
    print(f"  {'像素':>10}  {'已知波长(nm)':>14}  {'预测波长(nm)':>14}  {'残差(nm)':>10}")
    for inl in sorted(result.inliers, key=lambda x: x.true_wavelength):
        print(f"  {inl.centroid_pixel:10.3f}  {inl.true_wavelength:14.3f}  "
              f"{inl.predicted_wavelength:14.3f}  {inl.residual_nm:10.4f}")
    if result.unmatched_wavelengths:
        print()
        print(f"  未匹配的已知波长: {result.unmatched_wavelengths}")
    print(sep)