import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from wavecal import estimate_baseline, estimate_local_noise, Config


def select_csv_file() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(
        title="选择要检查的光谱 CSV 文件",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not file_path:
        raise SystemExit("未选择文件，程序退出。")
    return file_path


# ── 选择文件 ──────────────────────────────────────────────────────────────────
csv_path = select_csv_file()
print(f"已选择文件: {csv_path}\n")

# ── 读取数据 ──────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path, header=None, names=["pixel", "intensity"])
intensity = np.asarray(df["intensity"].values, dtype=float)

# ── 计算基线 + 局部噪声 ─────────────────────────────────────────────────────────
cfg = Config(boundary_n_sigma=3.0)   # 务必和实际跑pipeline时使用的参数保持一致
baseline, noise_std = estimate_baseline(
    intensity,
    window_size=cfg.baseline_window_size,
    percentile=cfg.baseline_percentile,
)
local_noise = estimate_local_noise(
    intensity,
    baseline,
    window_size=cfg.baseline_window_size,
)

# ── 检查区间（按需修改 START / END）─────────────────────────────────────────
START, END = 241, 248   # 改成你想看的峰左侧范围

print(f"noise_std(全局) = {noise_std:.4f}\n")
print(f"{'px':>4}  {'intensity':>10}  {'baseline':>9}  {'I_corr':>9}  "
      f"{'local_noise':>11}  {'local_thresh':>12}  回落")
print("-" * 70)

for p in range(START, END):
    I_corr = intensity[p] - baseline[p]
    local_thresh = cfg.boundary_n_sigma * local_noise[p]
    fallen = "是" if I_corr <= local_thresh else "否"
    print(f"{p:4d}  {intensity[p]:10.1f}  {baseline[p]:9.2f}  "
          f"{I_corr:9.2f}  {local_noise[p]:11.3f}  {local_thresh:12.3f}  {fallen}")