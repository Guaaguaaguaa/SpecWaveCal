"""
logger.py — 统一日志管理
职责：记录被舍弃的峰及原因，输出到 log.txt
暴露接口：
    Logger(path)          初始化，绑定输出文件
    logger.rejected(...)  记录一条舍弃记录
    logger.info(...)      记录一条普通信息
    logger.summary(...)   在文件末尾写入汇总
"""

import datetime
from pathlib import Path
from typing import Optional


# ── 舍弃原因枚举（字符串常量，避免拼写错误）──────────────────────────────────
class RejectReason:
    LOW_SNR           = "SNR不足"
    BOUNDARY_BLENDED  = "边界未收敛（双峰连叠或探测器边缘）"
    WIDTH_TOO_NARROW  = "半宽过窄（可能为噪声尖刺）"
    WIDTH_TOO_WIDE    = "半宽过宽（边界异常）"
    ASYMMETRIC        = "峰形不对称（|skewness| 超过阈值）"
    NEGATIVE_WING     = "扣基线后出现大范围负值（基线估计异常）"


class Logger:
    """
    用法示例：
        log = Logger("log.txt")
        log.info("开始处理 down_HgAr_mean.csv")
        log.rejected(peak_pixel=475, reason=RejectReason.ASYMMETRIC,
                     details={"skewness": 0.82, "threshold": 0.5})
        log.summary(total=30, used=22, rejected=8)
        log.close()
    """

    # 固定列宽，方便对齐阅读
    _SEP = "-" * 72

    def __init__(self, path: str = "log.txt"):
        self._path = Path(path)
        self._file = open(self._path, "w", encoding="utf-8")
        self._rejected_count = 0
        self._write_header()

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    def info(self, message: str) -> None:
        """普通信息行，带时间戳。"""
        self._writeln(f"[INFO]  {self._ts()}  {message}")

    def rejected(
        self,
        peak_pixel: float,
        reason: str,
        details: Optional[dict] = None,
    ) -> None:
        """
        记录一个被舍弃的峰。
        peak_pixel : 粗定位的峰顶像素（可为小数）
        reason     : RejectReason 中的常量字符串
        details    : 附加数值信息，如 {"snr": 3.2, "threshold": 10}
        """
        self._rejected_count += 1
        self._writeln(f"[REJECT #{self._rejected_count:03d}]")
        self._writeln(f"  峰顶像素   : {peak_pixel:.2f} px")
        self._writeln(f"  舍弃原因   : {reason}")
        if details:
            for k, v in details.items():
                # 数值保留4位小数，其他原样
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                self._writeln(f"  {k:<14s}: {val_str}")
        self._writeln("")  # 空行分隔

    def summary(self, total: int, used: int, rejected: int) -> None:
        """
        在日志末尾写入处理汇总。

        注：自 peak_finder.py 改为"全部保留+打标签"设计后，used/rejected
        的含义变为"完全通过质量检测"/"存在至少一项质量标记"，而非真正
        被剔除（所有候选峰都会出现在最终结果中）。
        """
        self._writeln(self._SEP)
        self._writeln("[SUMMARY]")
        self._writeln(f"  候选峰总数     : {total}")
        self._writeln(f"  完全通过检测   : {used}")
        self._writeln(f"  存在质量标记   : {rejected}")
        self._writeln(self._SEP)

    def close(self) -> None:
        """显式关闭文件（pipeline 结束时调用）。"""
        if not self._file.closed:
            self._file.close()

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    def _write_header(self) -> None:
        self._writeln(self._SEP)
        self._writeln(f"光谱处理日志  生成时间: {self._ts()}")
        self._writeln(self._SEP)
        self._writeln("")

    @staticmethod
    def _ts() -> str:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _writeln(self, line: str) -> None:
        self._file.write(line + "\n")
        self._file.flush()  # 实时写盘，程序崩溃也不丢日志

    # ── 支持 with 语句 ────────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()