# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置。

用法（在项目根目录）::

    # lite 版（无 OCR，体积小）：默认
    pyinstaller docscan.spec --noconfirm

    # full 版（含 RapidOCR，支持 CER 评测）：
    DOCSCAN_FULL=1 pyinstaller docscan.spec --noconfirm

产物：dist/docscan/docscan.exe（onedir，console=False）
注意：断网可用——OCR 走本地 onnxruntime，全流程无网络依赖。
"""

import os

FULL = os.environ.get("DOCSCAN_FULL", "") not in ("", "0")

block_cipher = None

hiddenimports = ["docscan.experiments.exp_e2e"]
excludedlibs = [
    "skimage",
    "scipy",
    "matplotlib",
    "pytest",
    "tkinterdnd2",
    "sv_ttk",
    "pandas",
    "IPython",
]
if FULL:
    hiddenimports += [
        "rapidocr_onnxruntime",
        "rapidocr_onnxruntime.ch_ppocr_cls",
        "rapidocr_onnxruntime.ch_ppocr_det",
        "rapidocr_onnxruntime.ch_ppocr_rec",
        "rapidocr_onnxruntime.utils",
    ]
else:
    excludedlibs += ["rapidocr_onnxruntime", "onnxruntime", "rapidocr"]

a = Analysis(
    ["run_docscan.py"],
    pathex=["src"],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludedlibs,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="docscan",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # GUI 程序：不弹黑窗
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="docscan",
)
