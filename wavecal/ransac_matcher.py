"""
ransac_matcher.py — 基于 RANSAC 的自动谱线识别
职责：在不知道任何仪器专属锚点像素位置的前提下，仅依据"已知参考波长列表"
      和"全谱寻峰得到的候选峰质心列表"，自动识别出哪个候选峰对应哪条已知
      谱线，对仪器型号、通道数、分辨率差异、整体平移、强度顺序变化均免疫。

算法设计 (v4):
    采用 3 点二次 RANSAC（wavelength = a2·pixel² + a1·pixel + a0），
    配合多重保障机制确保收敛：

    1. 空间多样性约束 — 3 个抽样点必须跨越 > 30% 像素范围
    2. 波长覆盖约束 — 模型预测的波长跨度必须 >= 真值跨度的 50%
    3. 单调性约束 — 匹配的 (像素, 波长) 对必须严格单调
    4. 主动搜索 — 用精修模型反向定位遗漏的波长
    5. 多轮独立运行 — 5 轮独立 RANSAC，取内点数最多的结果

    为什么不用 sin 模型：
    二次多项式在 435-966 nm 范围内残差 < 0.5 nm，远小于 HgAr 最小线距
    14.6 nm。sin 模型需 4 参数，搜索空间暴增 50 倍，收敛不可靠。
    最终定标由 calibrate() 用三次多项式完成，追求物理精度。

设计变更记录（本版本）：
    新增 excluded_centroids 参数：调用方可以把 peak_finder.py 里打了
    FlagReason.SUSPECTED_ARTIFACT（疑似伪峰：窄峰+边际信噪比组合）等
    强标记的候选峰质心传进来，在RANSAC种子抽样和内点投票开始前就排除，
    避免这类低置信度候选污染二次模型拟合或意外抢占某条波长的"最佳匹配"
    名额。默认 None，不传时行为与旧版完全一致。
    排除过程全程可追溯：被排除的质心会原样保留在
    RansacMatchResult.excluded_centroids 字段中，不是静默丢弃。

暴露接口：
    RansacMatchResult, RansacInlier, RansacMatchError
    ransac_match_wavelengths(...)
    print_ransac_report(result)
"""

import itertools
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np


class RansacMatchError(Exception):
    """RANSAC 识别失败（所有轮次均找不到足够内点）。"""
    pass


@dataclass
class RansacInlier:
    centroid_pixel : float
    true_wavelength: float


@dataclass
class RansacMatchResult:
    a2              : float
    a1              : float
    a0              : float
    inliers         : List[RansacInlier]
    outlier_centroids   : List[float]
    unmatched_wavelengths: List[float]
    n_iterations_used: int
    inlier_tolerance_nm: float
    # 预过滤阶段被排除的候选峰（如调用方根据SUSPECTED_ARTIFACT等标记判定
    # 为不可信、不参与匹配的质心）。全程记录，不是静默丢弃。
    excluded_centroids : List[float] = field(default_factory=list)

    @property
    def calibration_pairs(self) -> List[tuple]:
        """按波长升序的 (centroid_pixel, wavelength_nm) 对，可直接喂给 calibrate()。"""
        return [(inl.centroid_pixel, inl.true_wavelength)
                for inl in sorted(self.inliers, key=lambda x: x.true_wavelength)]

    def predict(self, pixel: float) -> float:
        return float(self.a2 * pixel ** 2 + self.a1 * pixel + self.a0)


# ── 内部：单轮 3 点二次 RANSAC ─────────────────────────────────────────────────

def _ransac_one_round(
    candidates_arr: np.ndarray,
    wavelengths_arr: np.ndarray,
    tolerance_nm: float,
    n_iterations: int,
    require_positive_slope: bool,
    rng: random.Random,
) -> Tuple[Optional[float], Optional[float], Optional[float],
           List[Tuple[int, int]]]:
    """执行一轮 3 点二次 RANSAC，返回最佳模型系数和内点索引列表。"""

    best_a2, best_a1, best_a0 = None, None, None
    best_pairs: List[Tuple[int, int]] = []

    px_min = float(candidates_arr.min())
    px_max = float(candidates_arr.max())
    wl_min = float(wavelengths_arr.min())
    wl_max = float(wavelengths_arr.max())
    span_threshold = 0.3 * (px_max - px_min)

    wl_range = wl_max - wl_min

    for _ in range(n_iterations):
        if len(candidates_arr) < 3 or len(wavelengths_arr) < 3:
            break

        # 空间多样性约束
        c_vals = None
        for _retry in range(20):
            trial = rng.sample(list(candidates_arr), 3)
            if max(trial) - min(trial) >= span_threshold:
                c_vals = trial
                break
        if c_vals is None:
            c_vals = rng.sample(list(candidates_arr), 3)

        w_vals = rng.sample(list(wavelengths_arr), 3)
        if len(set(c_vals)) < 3 or len(set(w_vals)) < 3:
            continue

        for w_perm in itertools.permutations(w_vals):
            px_arr = np.array(c_vals, dtype=float)
            wl_arr = np.array(w_perm, dtype=float)
            if np.any(np.diff(np.sort(px_arr)) < 0.5):
                continue

            try:
                coeffs = np.polyfit(px_arr, wl_arr, 2)
            except np.linalg.LinAlgError:
                continue
            a2, a1, a0 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

            # 色散方向
            if require_positive_slope:
                if a2 >= 0:
                    min_deriv = 2 * a2 * px_min + a1
                else:
                    min_deriv = 2 * a2 * px_max + a1
                if min_deriv <= 1e-9:
                    continue

            # 波长覆盖约束：模型在整个像素范围内的预测波长跨度
            pred_at_ends = np.array([
                a2 * px_min ** 2 + a1 * px_min + a0,
                a2 * px_max ** 2 + a1 * px_max + a0,
            ])
            model_wl_range = abs(float(pred_at_ends[1] - pred_at_ends[0]))
            if abs(a2) > 1e-12:
                # 二次模型波长跨度取决于抛物线段
                vertex_px = -a1 / (2 * a2)
                if px_min < vertex_px < px_max:
                    vertex_wl = a2 * vertex_px ** 2 + a1 * vertex_px + a0
                    model_wl_range = max(
                        abs(float(pred_at_ends[1] - vertex_wl)),
                        abs(float(pred_at_ends[0] - vertex_wl)),
                    )
            if model_wl_range < 0.4 * wl_range:
                continue

            # 对全部候选峰投票
            predicted = a2 * candidates_arr ** 2 + a1 * candidates_arr + a0
            diff_matrix = np.abs(predicted[:, None] - wavelengths_arr[None, :])
            nearest_wl_idx = np.argmin(diff_matrix, axis=1)
            nearest_wl_dist = diff_matrix[
                np.arange(len(candidates_arr)), nearest_wl_idx
            ]

            inlier_mask = nearest_wl_dist <= tolerance_nm
            inlier_c_idx = np.where(inlier_mask)[0]

            # 每条波长只保留最佳匹配
            wl_to_best = {}
            for c_idx in inlier_c_idx:
                wl_idx = int(nearest_wl_idx[c_idx])
                dist = nearest_wl_dist[c_idx]
                if wl_idx not in wl_to_best or dist < wl_to_best[wl_idx][1]:
                    wl_to_best[wl_idx] = (int(c_idx), dist)

            n_inliers = len(wl_to_best)

            # 单调性约束
            if n_inliers > len(best_pairs) and n_inliers >= 3:
                sorted_pairs = sorted(
                    wl_to_best.items(),
                    key=lambda item: candidates_arr[item[1][0]],
                )
                sorted_wls = [wavelengths_arr[wl_idx]
                              for wl_idx, _ in sorted_pairs]
                if not all(sorted_wls[i] < sorted_wls[i + 1]
                           for i in range(len(sorted_wls) - 1)):
                    continue

            if n_inliers > len(best_pairs):
                best_a2, best_a1, best_a0 = a2, a1, a0
                best_pairs = [
                    (c_idx, wl_idx)
                    for wl_idx, (c_idx, _) in wl_to_best.items()
                ]

    return best_a2, best_a1, best_a0, best_pairs


# ── 主入口 ────────────────────────────────────────────────────────────────────

def ransac_match_wavelengths(
    candidate_centroids  : List[float],
    true_wavelengths     : List[float],
    tolerance_nm         : float = 2.0,
    n_iterations          : int   = 5000,
    min_inliers           : Optional[int] = None,
    random_seed            : Optional[int] = None,
    require_positive_slope : bool = True,
    excluded_centroids     : Optional[Iterable[float]] = None,
    exclude_match_tol_px   : float = 1e-3,
) -> RansacMatchResult:
    """
    用 RANSAC 自动识别候选峰质心与已知参考波长的对应关系。

    v4: 3 点二次模型 + 空间约束 + 覆盖约束 + 单调性 + 主动搜索 + 多轮运行

    Parameters
    ----------
    candidate_centroids : 候选峰质心列表（像素）
    true_wavelengths    : 已知参考波长列表（nm）
    tolerance_nm         : 内点判定容差（nm），默认 2.0
    n_iterations         : 每轮 RANSAC 迭代次数，默认 5000
    min_inliers          : 最少内点数，默认 ceil(len(wavelengths) * 0.5)
    random_seed           : 随机种子，None 则不固定
    require_positive_slope: 是否强制色散方向为正
    excluded_centroids    : 调用方根据峰质量标记（如 peak_finder.FlagReason.
                             SUSPECTED_ARTIFACT）预先判定为不可信、不参与
                             匹配的候选峰质心列表。这些点在种子抽样和内点
                             投票开始前就被剔除，避免疑似伪峰被当作种子点
                             拉偏二次模型，或者意外成为某条波长的"最佳匹配"。
                             不是静默丢弃——会完整保留在返回结果的
                             RansacMatchResult.excluded_centroids 字段中。
                             默认 None，即不做任何预过滤，行为与旧版一致。
    exclude_match_tol_px  : 把 candidate_centroids 中的值与 excluded_centroids
                             做匹配的容差（像素），默认 1e-3，足够覆盖浮点
                             精度误差

    Returns
    -------
    RansacMatchResult
    """
    candidates_input = list(candidate_centroids)
    wavelengths = list(true_wavelengths)

    # ── 预过滤：剔除调用方标记为不可信的候选峰 ──────────────────────────────
    #    必须在RANSAC种子抽样之前执行，否则疑似伪峰仍可能被抽中作为3点种子
    #    之一，拉偏整个二次模型；或者在投票阶段意外成为某条波长距离最近的
    #    "最佳匹配"，把真正的候选峰挤掉。
    excluded_list: List[float] = []
    if excluded_centroids:
        excl_arr = np.asarray(list(excluded_centroids), dtype=float)
        candidates: List[float] = []
        for c in candidates_input:
            if excl_arr.size and np.any(np.abs(excl_arr - c) <= exclude_match_tol_px):
                excluded_list.append(c)
            else:
                candidates.append(c)
    else:
        candidates = candidates_input

    if len(candidates) < 3:
        raise ValueError(
            f"候选峰数量 ({len(candidates)}) 少于 3 个"
            + (f"（已预先排除 {len(excluded_list)} 个标记为不可信的候选峰，"
               f"若数量不足，请检查是否过滤过严）" if excluded_list else "")
            + "。"
        )
    if len(wavelengths) < 3:
        raise ValueError(f"已知波长数量 ({len(wavelengths)}) 少于 3 个。")

    if min_inliers is None:
        min_inliers = max(2, int(np.ceil(len(wavelengths) * 0.5)))

    candidates_arr = np.asarray(candidates, dtype=float)
    wavelengths_arr = np.asarray(wavelengths, dtype=float)
    px_min = float(candidates_arr.min())
    px_max = float(candidates_arr.max())

    # ── 多轮独立 RANSAC，取内点数最多的结果 ───────────────────────────────
    n_rounds = 5
    per_round_iterations = max(1000, n_iterations // n_rounds)

    best_a2, best_a1, best_a0 = None, None, None
    best_pairs: List[Tuple[int, int]] = []

    for round_idx in range(n_rounds):
        seed = (random_seed or 0) + round_idx * 7919  # 大质数间隔，避免轮次间相关性
        rng = random.Random(seed)
        a2, a1, a0, pairs = _ransac_one_round(
            candidates_arr, wavelengths_arr,
            tolerance_nm, per_round_iterations,
            require_positive_slope, rng,
        )
        if a2 is not None and len(pairs) > len(best_pairs):
            best_a2, best_a1, best_a0 = a2, a1, a0
            best_pairs = pairs

    if best_a2 is None or len(best_pairs) < min_inliers:
        found = len(best_pairs)
        raise RansacMatchError(
            f"RANSAC 识别失败：{n_rounds} 轮 × {per_round_iterations} 次迭代后，"
            f"最佳模型仅 {found} 个内点，低于最小值 {min_inliers}。"
        )

    # ── 二次最小二乘精修 ─────────────────────────────────────────────────
    inlier_px = np.array([candidates_arr[c] for c, _ in best_pairs])
    inlier_wl = np.array([wavelengths_arr[w] for _, w in best_pairs])
    A = np.vstack([inlier_px ** 2, inlier_px, np.ones_like(inlier_px)]).T
    coeffs = np.linalg.lstsq(A, inlier_wl, rcond=None)[0]
    refined_a2, refined_a1, refined_a0 = (
        float(coeffs[0]), float(coeffs[1]), float(coeffs[2]),
    )

    # 色散验证
    if require_positive_slope:
        if refined_a2 >= 0:
            min_deriv = 2 * refined_a2 * px_min + refined_a1
        else:
            min_deriv = 2 * refined_a2 * px_max + refined_a1
        if min_deriv <= 1e-9:
            raise RansacMatchError(
                f"精修后色散方向不满足单调递增 (最小导数={min_deriv:.6f})。"
            )

    # ── 覆盖范围安全检查 + 救援 RANSAC ─────────────────────────────────
    # 正确模型必须覆盖大部分真值波长范围。若不足 85%，
    # 追加额外轮次尝试找到更优模型。必须在主动搜索之前执行，
    # 否则救援结果会覆盖主动搜索已补入的点。
    matched_wls_for_check = [wavelengths_arr[w] for _, w in best_pairs]
    wl_coverage = ((max(matched_wls_for_check) - min(matched_wls_for_check))
                   / (float(wavelengths_arr.max()) - float(wavelengths_arr.min()))
                   ) if matched_wls_for_check else 0

    if wl_coverage < 0.85 and len(best_pairs) < len(wavelengths):
        for rescue_idx in range(3):
            rescue_seed = (random_seed or 0) + 12345 + rescue_idx * 7919
            rescue_rng = random.Random(rescue_seed)
            a2r, a1r, a0r, rescue_pairs = _ransac_one_round(
                candidates_arr, wavelengths_arr,
                tolerance_nm, per_round_iterations * 2,
                require_positive_slope, rescue_rng,
            )
            if a2r is not None and len(rescue_pairs) > len(best_pairs):
                best_a2, best_a1, best_a0 = a2r, a1r, a0r
                best_pairs = rescue_pairs

    # 救援后重新精修
    inlier_px = np.array([candidates_arr[c] for c, _ in best_pairs])
    inlier_wl = np.array([wavelengths_arr[w] for _, w in best_pairs])
    A = np.vstack([inlier_px ** 2, inlier_px, np.ones_like(inlier_px)]).T
    coeffs = np.linalg.lstsq(A, inlier_wl, rcond=None)[0]
    refined_a2, refined_a1, refined_a0 = (
        float(coeffs[0]), float(coeffs[1]), float(coeffs[2]),
    )

    if require_positive_slope:
        if refined_a2 >= 0:
            min_deriv = 2 * refined_a2 * px_min + refined_a1
        else:
            min_deriv = 2 * refined_a2 * px_max + refined_a1
        if min_deriv <= 1e-9:
            raise RansacMatchError(
                f"精修后色散方向不满足单调递增 (最小导数={min_deriv:.6f})。"
            )

    # ── 主动搜索遗漏波长 ─────────────────────────────────────────────────
    matched_wl_set = set(w for _, w in best_pairs)
    matched_c_set = set(c for c, _ in best_pairs)
    # 允许预测像素附近寻找遗漏谱线。
    # 30 px 约等于典型峰宽的 2~3 倍，
    # 用于容忍粗定位模型残余非线性误差。
    search_tol_px = 30.0

    for wl_idx in range(len(wavelengths)):
        if wl_idx in matched_wl_set:
            continue
        wl = wavelengths_arr[wl_idx]

        # 数值稳定：接近线性时用线性公式，避免二次求根病态
        if abs(refined_a2) < 1e-8:
            pred_px = (wl - refined_a0) / refined_a1 if abs(refined_a1) > 1e-9 else None
        else:
            disc = refined_a1 ** 2 - 4 * refined_a2 * (refined_a0 - wl)
            if disc < 0:
                pred_px = None
            else:
                px1 = (-refined_a1 + np.sqrt(disc)) / (2 * refined_a2)
                px2 = (-refined_a1 - np.sqrt(disc)) / (2 * refined_a2)
                pred_px = px1 if px_min <= px1 <= px_max else (
                    px2 if px_min <= px2 <= px_max else None
                )
        if pred_px is None:
            continue
        dists = np.abs(candidates_arr - pred_px)
        nearest = int(np.argmin(dists))
        if dists[nearest] > search_tol_px or nearest in matched_c_set:
            continue

        # 拓扑验证：补入该点后像素↑→波长↑必须保持严格单调
        trial_pairs = best_pairs + [(nearest, wl_idx)]
        trial_sorted = sorted(trial_pairs, key=lambda x: candidates_arr[x[0]])
        trial_wls = [wavelengths_arr[w] for _, w in trial_sorted]
        if not all(trial_wls[i] < trial_wls[i + 1]
                   for i in range(len(trial_wls) - 1)):
            continue  # 拓扑破坏，拒绝此补点

        best_pairs.append((nearest, wl_idx))
        matched_c_set.add(nearest)
        matched_wl_set.add(wl_idx)

    # 最终拟合
    final_px = np.array([candidates_arr[c] for c, _ in best_pairs])
    final_wl = np.array([wavelengths_arr[w] for _, w in best_pairs])
    A_f = np.vstack([final_px ** 2, final_px, np.ones_like(final_px)]).T
    final_coeffs = np.linalg.lstsq(A_f, final_wl, rcond=None)[0]
    final_a2, final_a1, final_a0 = (
        float(final_coeffs[0]), float(final_coeffs[1]), float(final_coeffs[2]),
    )

    # 主动搜索补点后最终模型再次验证色散方向
    if require_positive_slope:
        if final_a2 >= 0:
            min_deriv_f = 2 * final_a2 * px_min + final_a1
        else:
            min_deriv_f = 2 * final_a2 * px_max + final_a1
        if min_deriv_f <= 1e-9:
            raise RansacMatchError(
                f"主动搜索后最终模型色散方向不满足单调递增 "
                f"(最小导数={min_deriv_f:.6f})。"
            )

    # ── 组装结果 ─────────────────────────────────────────────────────────
    inliers: List[RansacInlier] = []
    matched_c_set = set()
    matched_w_set = set()

    for c_idx, wl_idx in best_pairs:
        px = float(candidates_arr[c_idx])
        true_wl = float(wavelengths_arr[wl_idx])
        inliers.append(RansacInlier(
            centroid_pixel  = px,
            true_wavelength = true_wl,
        ))
        matched_c_set.add(c_idx)
        matched_w_set.add(wl_idx)

    outlier_centroids = [
        float(candidates_arr[i])
        for i in range(len(candidates))
        if i not in matched_c_set
    ]
    unmatched_wavelengths = [
        float(wavelengths_arr[i])
        for i in range(len(wavelengths))
        if i not in matched_w_set
    ]

    return RansacMatchResult(
        a2                  = final_a2,
        a1                  = final_a1,
        a0                  = final_a0,
        inliers             = inliers,
        outlier_centroids   = outlier_centroids,
        unmatched_wavelengths = unmatched_wavelengths,
        n_iterations_used   = n_iterations,
        inlier_tolerance_nm = tolerance_nm,
        excluded_centroids  = excluded_list,
    )


def print_ransac_report(result: RansacMatchResult) -> None:
    sep = "-" * 70
    print(sep)
    print("  RANSAC 自动谱线识别报告")
    print(sep)
    print(f"  二次模型: WL = {result.a2:.8e}*px^2 "
          f"+ {result.a1:.6f}*px + {result.a0:.4f}")
    if result.inliers:
        inl_px = [x.centroid_pixel for x in result.inliers]
        d_min = 2 * result.a2 * min(inl_px) + result.a1
        d_max = 2 * result.a2 * max(inl_px) + result.a1
        print(f"  色散率范围 [{min(inl_px):.0f}, {max(inl_px):.0f}] px: "
              f"[{d_min:.4f}, {d_max:.4f}] nm/px")
    print(f"  内点数: {len(result.inliers)}  "
          f"外点数: {len(result.outlier_centroids)}  "
          f"未匹配波长数: {len(result.unmatched_wavelengths)}  "
          f"预过滤排除数: {len(result.excluded_centroids)}")
    if result.excluded_centroids:
        print(f"  预过滤排除的候选峰（如疑似伪峰）: {result.excluded_centroids}")
    if result.unmatched_wavelengths:
        print(f"  未匹配波长: {result.unmatched_wavelengths}")
    print()
    print(f"  {'像素(px)':>10}  {'标准波长(nm)':>14}  "
          f"{'预测波长(nm)':>14}  {'残差(nm)':>10}")
    for inl in sorted(result.inliers, key=lambda x: x.true_wavelength):
        pred = result.predict(inl.centroid_pixel)
        resid = pred - inl.true_wavelength
        print(f"  {inl.centroid_pixel:10.3f}  {inl.true_wavelength:14.3f}  "
              f"{pred:14.3f}  {resid:10.4f}")
    print(sep)