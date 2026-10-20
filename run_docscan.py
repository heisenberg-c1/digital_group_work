"""PyInstaller 入口（GUI）。spec 文件引用本脚本。"""

import multiprocessing  # noqa: F401 —— PyInstaller 在 Windows 上冻结多进程需提前 import
import sys
from pathlib import Path

# 打包后 src 不存在，docscan 由 PyInstaller 收进包里
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from docscan.app import run_gui  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_gui())
