# docscan · 手机文档扫描与可读性增强系统

《数字图像处理》课程设计（题目 1）：把手机随手拍的文档照片，处理成「平整、清晰、可读」的扫描件。

## 功能

- **几何校正**：自动检测页面四角（带置信度），透视变换拉平；失败时明确提示而非静默出错
- **光照归一**：背景估计去阴影、Retinex、灰世界 / 白块白平衡、直方图均衡 / 伽马 / 对比度拉伸 / CLAHE
- **色彩空间**：BGR ↔ YCbCr(BT.601) 与 BGR ↔ HSV 两套亮度/色度分离通道（`apply_on_luma` / `apply_on_value`）
- **去噪**：中值 / 双边（排在增强之后、去网纹与复原之前，避免噪声被锐化与维纳放大）
- **频域去网纹**：FFT 幅度谱 + 自动周期峰检测（DC 半径按最大待检周期推算 + 显著性校验）+ 理想 / Butterworth 陷波器，可频谱图手动选点；彩色图只在亮度通道陷波
- **退化复原**：运动 / 散焦 PSF 建模、逆滤波、维纳滤波、倒谱法估计模糊方向
- **二值化与评测**：Otsu / 自适应 / Sauvola，PSNR / SSIM / CER 全套指标
- **导出**：彩色 / 灰度 / 黑白三档，PNG / JPG 与 1-bit PDF（CCITT G4）；支持 PDF 输入
- **界面**：三栏工作台，阶段参数面板（滑杆 / 下拉框热更新）、频谱选点、参数预设（JSON 存/取）、滚轮缩放、明暗主题、高分屏 DPI 适配
- **CLI**：与 GUI 共用同一条处理链（`--enable` 逐项开阶段、`--preset` 载入预设、`-o out.pdf` 直接出 PDF）

## 架构

```
io/            装载（EXIF/透明底/BGR 统一）与导出（PNG/PDF）
core/          纯算法层，只依赖 numpy + opencv + scipy
  geometry       四点检测与透视校正（边缘支撑度过滤伪边界）
  colorspace     YCbCr / HSV 亮度-色度分离
  color          白平衡（灰世界 / 白块）
  illumination   光照归一
  enhance        伽马 / 拉伸 / 直方图均衡 / CLAHE
  denoise        中值 / 双边
  freq           频域分析与陷波
  restore        退化建模与复原
  binarize       二值化
  sharpen        USM
  metrics        评测指标
pipeline/      Runner 编排：阶段开关、参数热更新、进度回调
ui/            Tkinter 三栏界面（处理链与参数 / 画布 / 指标），sv-ttk 主题可退回自绘
experiments/   实验脚本：采样量化、色彩空间、空域增强、陷波对比、维纳 K 扫描、端到端 CER
```

## 快速开始

```bash
# 1) 创建虚拟环境（所有依赖只装进 .venv，不污染全局）
uv venv
uv pip install -e ".[dev,ocr]"

# 2) 跑测试
.venv/Scripts/python -m pytest -q

# 3) 命令行处理一张图（与界面同一条处理链）
.venv/Scripts/python -m docscan --cli samples/demo/tilted_doc.jpg -o out.png --mode bw
.venv/Scripts/python -m docscan --cli samples/demo/tilted_doc.jpg -o out.pdf --mode bw \
    --enable denoise,freq,restore          # 逐项打开附加阶段
.venv/Scripts/python -m docscan --cli in.pdf -o out.pdf --preset my.json   # 多页 PDF 进出

# 4) 图形界面
.venv/Scripts/python -m docscan
```

## 实验脚本（报告图表）

```bash
.venv/Scripts/python -m docscan.experiments.exp_sampling   --out data/results  # 要求① DPI×位深 CER 热力图
.venv/Scripts/python -m docscan.experiments.exp_colorspace --out data/results  # 要求② 亮度/色度分离与色相漂移
.venv/Scripts/python -m docscan.experiments.exp_enhance    --out data/results  # 要求③ 四种空域增强对比
.venv/Scripts/python -m docscan.experiments.exp_notch      --out data/results  # 要求④ 三种滤波对比
.venv/Scripts/python -m docscan.experiments.exp_wiener     --out data/results  # 要求⑤ K 扫描曲线
.venv/Scripts/python -m docscan.experiments.exp_e2e        --out data/results  # 要求⑦ 七场景端到端
.venv/Scripts/python tools/make_report_assets.py                              # 各模块 before/after 素材
.venv/Scripts/python tools/bench.py                                           # 冷启动 + 12MP 基准
```

OCR 用 RapidOCR（本地 onnxruntime，**断网可用**）；未安装时实验自动降级为 SSIM 代理指标。

## 打包

```bash
.venv/Scripts/python -m PyInstaller docscan.spec --noconfirm          # lite 版（无 OCR）
DOCSCAN_FULL=1 .venv/Scripts/python -m PyInstaller docscan.spec --noconfirm  # full 版（含 OCR）
# 产物：dist/docscan/docscan.exe（onedir，双击即用）
```

## 开发约定

- 分支模型：`main`（只接受 dev 合并）← `dev`（集成）← `feature/<模块>/<简述>`
- 提交信息：`<type>: <中文简述>`，动词开头、一条提交一件事
- 测试红线：新增 `core/` 函数必须有测试，且至少一个用例与 oracle 或解析解对拍
- 静默降级禁止：枚举参数遇到未知值必须抛 `ValueError`，不得回退默认值——`method="he"`
  静默按 CLAHE 跑、`psf_type="disk"` 静默用线段 PSF 处理散焦，都会产出「看着正常但
  不可复现」的数据（对应 `stage_enhance` / `stage_restore` 的白名单校验）
- 质量门槛：`ruff check src tests tools` 与 `pytest` 必须双绿；UI 改动至少跑一遍
  `tests/test_ui_smoke.py`（构造级冒烟，能拦住「漏导入导致对话框崩溃」这类问题）
- 迭代计划与 DoD 见 `项目开发规划-迭代与分支管理.md`
