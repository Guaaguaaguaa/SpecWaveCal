import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

from wavecal import run_explorer, Config


def select_csv_file() -> str:
    """弹出文件选择窗口，返回用户选择的 CSV 路径。"""
    root = tk.Tk()
    root.withdraw()          # 不显示主窗口，只弹出文件选择框
    root.attributes("-topmost", True)  # 确保弹窗在最前面，不被IDE遮挡

    file_path = filedialog.askopenfilename(
        title="选择要处理的光谱 CSV 文件",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()

    if not file_path:
        raise SystemExit("未选择文件，程序退出。")
    return file_path


# ── 选择文件 ──────────────────────────────────────────────────────────────────
csv_path = select_csv_file()
print(f"已选择文件: {csv_path}")

# ── 读取数据 ──────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path, header=None, names=["pixel", "intensity"])
intensity = np.asarray(df["intensity"].values, dtype=float)  # 显式转为 ndarray[float]，消除类型警告

# ── 运行探索流程 ──────────────────────────────────────────────────────────────
cfg = Config(boundary_n_sigma=4.0, log_path="log.txt", quality_csv_path="peak_quality.csv")
result = run_explorer(intensity, cfg)