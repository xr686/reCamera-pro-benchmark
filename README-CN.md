# reCamera Pro RKNN 模型基准测试

[![模型](https://img.shields.io/badge/models-25-blue)](./models/)
[![设备](https://img.shields.io/badge/device-reCamera%20Pro%20(RV1126B)-orange)](https://www.seeedstudio.com/)

对 Seeed reCamera Pro（Rockchip RV1126B，2 GB RAM，aarch64）进行了全面的 RKNN 模型基准测试。所有模型均转换为 RKNN 格式，并在设备上使用真实的 NPU 推理进行基准测试。

下文所有 YOLO 数据均来自 [`native/`](./native/) 中的原生 C++ 基准程序：RGA 直接把 letterbox
结果写入零拷贝的 RKNN 输入缓冲、int8 输出原地解码、NPU/CPU 频率锁定到最高档。

---

## 快速入门

```bash
# 克隆仓库（使用 Git LFS 下载模型权重）
git lfs install
git clone https://github.com/Seeed-Projects/recamera-pro-benchmark.git
cd recamera-pro-benchmark

# 在设备上跑完整个 YOLO 矩阵（交叉编译 + 推送 + 基准测试）
DEVICE_IP=192.168.42.1 ./scripts/bench_native.sh

# 也可以只跑单个模型或单个任务族
./scripts/bench_native.sh --only YOLOv8n
./scripts/bench_native.sh --only det
```

---

## 仓库结构

```
recamera-pro-benchmark/
├── README.md                           # 本文件
├── AGENTS.md                           # 详细的基准测试方法论与规则
│
├── models/                             # 所有模型按系列组织
│   ├── yolo/                           # YOLO 系列（v8, 11, 26）
│   │   ├── tools/                      # ONNX 优化脚本
│   │   │   ├── optimize_yolo_onnx.py   # YOLOv8/YOLO11 ONNX 优化器
│   │   │   └── optimize_yolo26_onnx.py # YOLO26 ONNX 优化器
│   │   ├── yolov8_det/                 # YOLOv8 目标检测 (n, s)
│   │   ├── yolov8_pose/                # YOLOv8 姿态估计
│   │   ├── yolov8_obb/                 # YOLOv8 旋转边界框
│   │   ├── yolov8_seg/                 # YOLOv8 实例分割
│   │   ├── yolo11_det/                 # YOLO11 目标检测 (n, s)
│   │   ├── yolo11_pose/               # YOLO11 姿态估计
│   │   ├── yolo11_obb/                # YOLO11 旋转边界框
│   │   ├── yolo11_seg/                # YOLO11 实例分割
│   │   ├── yolo11_cls/                # YOLO11 图像分类
│   │   ├── yolo26_det/                # YOLO26 目标检测
│   │   ├── yolo26_pose/               # YOLO26 姿态估计
│   │   ├── yolo26_obb/                # YOLO26 旋转边界框
│   │   ├── yolo26_seg/                # YOLO26 实例分割
│   │   ├── yolo26_cls/                # YOLO26 图像分类
│   │   └── yolo26_depth/              # YOLO26 深度估计
│   │
│   ├── ppocr/                          # OCR 流水线
│   │   └── ppocrv4/                    # PPOCRv4 检测 + 识别 (INT8)
│   │
│   └── zipformer/                      # 语音转文字
│       └── (encoder + decoder + joiner, FP16)
│
├── native/                             # C++ 基准测试程序（YOLO 推理路径）
│   ├── bench_native.cpp                # 零拷贝 RKNN + RGA + int8 解码，覆盖全部 YOLO 任务
│   ├── fetch_libs.sh                   # 获取交叉编译所需的 rknn_api.h / librknnrt.so / RGA
│   ├── CMakeLists.txt                  # aarch64 交叉构建，RPATH = /oem/usr/lib
│   └── build.sh                        # 使用 reCamera Pro SDK 工具链交叉编译
│
├── benchmark_results/                  # 基准测试数据与日志
│   └── native/                         # 每个模型的日志 + results.md / results.csv
├── scripts/                            # 转换与基准测试脚本
│   └── bench_native.sh                 # 构建 + 推送 + 跑完整个 YOLO 矩阵
├── assets/                             # 测试图像
├── calibration/                        # 校准数据集
└── doc/                                # 参考文档
```

每个模型子目录均为自包含的，包含：
- `*.onnx` — 原始 ONNX 模型
- `*_optimized.onnx` — 优化后的 ONNX（如适用）
- `*_W8A8.rknn` / `*_FP16.rknn` — 转换后的 RKNN 模型
- `convert_to_rknn.py` — ONNX → RKNN 转换脚本
- `infer.py` — 设备推理脚本（仅 OCR / 语音；YOLO 使用 `native/bench_native`）
- `calibration_200.txt` — 校准图像列表
- `result_*.jpg` — 示例推理结果
- `README.md` — 模型专项文档

---

## 模型概览

**25 个 RKNN 模型**，覆盖 3 个系列、9 个任务类别，全部在 RV1126B NPU 上进行基准测试。

延迟为完整的串行单帧耗时——**预处理**（RGA letterbox 写入零拷贝输入缓冲）+ **NPU**
（一次 `rknn_run`）+ **后处理**（解码、NMS、掩码/深度图），测试参数 `--runs 100 --warmup 20`，
FPS = 1000 / 延迟。原始日志见 `benchmark_results/native/results.md`。

### 目标检测

| 模型 | 输入 | 精度 | 预处理 (ms) | NPU (ms) | 后处理 (ms) | 延迟 (ms) | FPS |
|------|------|-----|-----------:|--------:|-----------:|----------:|----:|
| YOLO26n | 640×640 | INT8 | 1.81 | 18.87 | 0.49 | **21.17** | **47.2** |
| YOLO11n | 640×640 | INT8 | 1.79 | 17.67 | 0.65 | **20.11** | **49.7** |
| YOLO11s | 640×640 | INT8 | 1.81 | 33.54 | 0.73 | **36.08** | **27.7** |
| YOLOv8n | 640×640 | INT8 | 1.82 | 15.30 | 0.68 | **17.80** | **56.2** |
| YOLOv8s | 640×640 | INT8 | 1.83 | 30.05 | 0.67 | **32.56** | **30.7** |

### 姿态估计

| 模型 | 输入 | 精度 | 预处理 (ms) | NPU (ms) | 后处理 (ms) | 延迟 (ms) | FPS |
|------|------|-----|-----------:|--------:|-----------:|----------:|----:|
| YOLO26n-pose | 640×640 | INT8 | 1.80 | 20.06 | 0.24 | **22.10** | **45.2** |
| YOLO11n-pose | 640×640 | INT8 | 1.77 | 18.05 | 0.41 | **20.23** | **49.4** |
| YOLOv8n-pose | 640×640 | INT8 | 1.81 | 15.74 | 0.33 | **17.89** | **55.9** |

### 实例分割

| 模型 | 输入 | 精度 | 预处理 (ms) | NPU (ms) | 后处理 (ms) | 延迟 (ms) | FPS |
|------|------|-----|-----------:|--------:|-----------:|----------:|----:|
| YOLO26n-seg | 640×640 | INT8 | 1.82 | 24.17 | 3.65 | **29.63** | **33.7** |
| YOLO11n-seg | 640×640 | INT8 | 1.81 | 22.11 | 4.12 | **28.04** | **35.7** |
| YOLOv8n-seg | 640×640 | INT8 | 1.82 | 19.90 | 2.88 | **24.59** | **40.7** |

后处理含每个检测框的掩码合成（框内 `sigmoid(coef · proto)`）。

### 旋转边界框 (OBB)

| 模型 | 输入 | 精度 | 预处理 (ms) | NPU (ms) | 后处理 (ms) | 延迟 (ms) | FPS |
|------|------|-----|-----------:|--------:|-----------:|----------:|----:|
| YOLO26n-obb | 640×640 | INT8 | 1.80 | 18.33 | 0.13 | **20.27** | **49.3** |
| YOLO11n-obb | 640×640 | INT8 | 1.81 | 16.98 | 0.13 | **18.92** | **52.9** |
| YOLOv8n-obb | 640×640 | INT8 | 1.79 | 14.91 | 0.12 | **16.81** | **59.5** |

OBB 模型基于 DOTA 航拍数据训练，街景 `bus.jpg` 在 conf 0.25 下检出 0 个目标（最高类别分
仅 0.01–0.13）；解码与旋转 NMS 每帧仍照常执行。

### 图像分类

| 模型 | 输入 | 精度 | 预处理 (ms) | NPU (ms) | 后处理 (ms) | 延迟 (ms) | FPS |
|------|------|-----|-----------:|--------:|-----------:|----------:|----:|
| YOLO26n-cls | 224×224 | INT8 | 1.74 | 2.16 | 0.02 | **3.93** | **254.3** |
| YOLO11n-cls | 224×224 | FP16 输入 | 1.79 | 9.97 | 0.03 | **11.78** | **84.9** |

### 深度估计

| 模型 | 输入 | 精度 | 预处理 (ms) | NPU (ms) | 后处理 (ms) | 延迟 (ms) | FPS |
|------|------|-----|-----------:|--------:|-----------:|----------:|----:|
| YOLO26n-depth | 640×640 | INT8 | 1.73 | 84.80 | 2.32 | **88.85** | **11.3** |

后处理为米制深度解码（`exp(clip(raw, ±10)) · e^cal_b`，用 256 项 int8 查找表完成）
加上一次 RGA 放大，把归一化后的深度图放大到源帧分辨率。

> **YOLO 表格不再给出功耗列。** 板子上没有功耗遥测节点（只有 SARADC 与 IMU），此前
> 5.0–5.6 W 的数据是 Python 测试时用外部 USB-C 功率计测得的，无法在此按模型复现。
> 持续 NPU 负载下整板功耗仍在 5 W 左右。

### OCR (PPOCRv4)

| 组件 | 输入 | 精度 | 延迟 (ms)* | FPS* | 功耗 (W) |
|------|------|------|-----------|------|----------|
| ppocrv4_det | 480×480 | INT8 | 58.62 | 17.1 | 5.0 |
| ppocrv4_rec | 48×320 | INT8 | (合计) | (合计) | |

\* OCR 延迟/FPS 为检测 + 识别完整流水线的合计值，速度随文本内容变化。

### 语音转文字 (Zipformer)

| 组件 | 输入 | 精度 | 延迟 (ms) | FPS | 功耗 (W) |
|------|------|------|-----------|-----|----------|
| zipformer_encoder | 103×80 | FP16 | 1834.9（整组合计） | / | 5.2 |
| zipformer_decoder | 1×2 | FP16 | | / | |
| zipformer_joiner | 1×512 | FP16 | | / | |

| 音频 | 总延迟 | RTF |
|------|--------|-----|
| 5.61 s test.wav | 1834.9 ms | 0.33 |
| 5.00 s 麦克风 | 1580.1 ms | 0.32 |

RTF < 1.0 表示实时处理（推理速度快于音频时长）。

---

## 模型目录使用方法

每个模型目录包含转换和部署所需的全部文件：

### ONNX → RKNN 转换（x86_64 主机）

```bash
cd models/yolo/yolov8_det
python3 convert_to_rknn.py
```

使用 RKNN-Toolkit2，`target_platform='rv1126b'`，200 张校准图像。

### 构建与运行原生基准程序

```bash
# 1. 交叉编译（需要 reCamera Pro SDK 工具链；可用 RECAMERA_SDK_ROOT 覆盖路径）
cd native && ./build.sh          # 首次运行会先执行 fetch_libs.sh 拉取头文件与库

# 2. 在仓库根目录：构建 + 推送二进制/测试帧/模型 + 跑完全部模型
./scripts/bench_native.sh                       # 每模型 100 次运行 / 20 次预热
RUNS=30 WARMUP=10 ./scripts/bench_native.sh     # 快速跑一遍
./scripts/bench_native.sh --only seg            # 只跑某个任务族
./scripts/bench_native.sh --report-only         # 用已有日志重新生成 results.md
```

也可以直接在设备上调用二进制：

```bash
scp native/build/bench_native root@192.168.42.1:/userdata/benchmark/
ssh root@192.168.42.1 \
  'cd /userdata/benchmark && LD_LIBRARY_PATH=/oem/usr/lib ./bench_native \
     --model models/yolov8n_640x640_W8A8.rknn --image bench.rgb --runs 100 --warmup 20'
```

### bench_native 参数

| 参数 | 说明 |
|------|------|
| `--model <file.rknn>` | 待测模型（必填） |
| `--image <raw.rgb>` | 原始 RGB888 输入帧（默认 1280×720，见 `--src`） |
| `--src WxH` | 原始帧分辨率，默认 `1280x720` |
| `--runs N` / `--warmup N` | 计时次数 / 预热次数（默认 100 / 20） |
| `--conf F` / `--nms F` | 检测阈值（默认 0.25 / 0.45） |
| `--task T` | 强制指定 `det`/`pose`/`seg`/`obb`/`cls`/`depth`（默认按输出头自动识别） |
| `--dump-io` | 打印输入/输出张量属性后退出 |
| `--trace-post` | 后处理分阶段耗时（深度模型） |

一个二进制覆盖全部六种 YOLO 任务：DFL/直接回归框、类别与 objectness 通道、关键点、
掩码系数、角度、proto 等输出头都由张量名与形状自动推断，YOLOv8 / YOLO11 / YOLO26
无需任何按模型定制的代码。

---

## ONNX 优化

从 PyTorch（`*.pt`）导出的 YOLO 模型包含不适合 NPU 执行的后处理解码操作。`models/yolo/tools/` 中的优化脚本会剥离这些 CPU 端操作：

```bash
# 优化 YOLOv8/YOLO11 ONNX
python3 models/yolo/tools/optimize_yolo_onnx.py --model yolov8n.onnx --output yolov8n_optimized.onnx

# 优化 YOLO26 ONNX
python3 models/yolo/tools/optimize_yolo26_onnx.py --model yolo26n.onnx --output yolo26n_optimized.onnx
```

主要优化内容：
- 剥离 DFL 解码、锚点生成、NMS 和 sigmoid 操作
- 输出原始空间特征图供 NPU 加速推理
- 后处理在推理脚本中由 CPU 执行

---

## 设备环境

| 组件 | 规格 |
|------|------|
| 设备 | Seeed reCamera Pro |
| SoC | Rockchip RV1126B |
| CPU | ARM Cortex-A7, 4 核 |
| RAM | 1.9 GB（总计） |
| NPU | 2.0 TOPS |
| RKNN Runtime | 2.3.2 |
| 内核 | Linux 6.1.x |
| 主机工具包 | RKNN-Toolkit2 2.3.2 (x86_64) |

---

## 主机环境与工具链

在转换或基准测试模型之前，主机必须安装以下工具链。

| 组件 | 版本 | 用途 |
|------|------|------|
| 主机操作系统 | Ubuntu 22.04 (x86_64) / WSL2 | 构建与转换主机 |
| Python | 3.10 | 转换脚本 + OCR/语音推理 |
| reCamera Pro SDK | v1.0.10 | `native/` 的 aarch64 交叉工具链与 sysroot（`RECAMERA_SDK_ROOT`） |
| CMake | ≥ 3.16 | 原生基准程序构建 |
| ffmpeg | 任意 | 把 `assets/bus.jpg` 解码为原始 RGB888 测试帧 |
| sshpass | 任意 | `scripts/bench_native.sh` 的设备推送 |
| Conda 环境 | `recamera-rknn-2.3.2` (miniforge3) | ONNX → RKNN 转换 |
| RKNN-Toolkit2 | **2.3.2** | 模型转换，`target_platform='rv1126b'` |
| ONNX | ≥ 1.14 | ONNX 图检查与优化 |
| ONNX Runtime | ≥ 1.16 | CPU 端精度对比 |
| NumPy | 1.24 | 张量运算 |
| OpenCV | 4.x | 图像 I/O、绘制、NMS |
| PyTorch（可选） | 2.x | 通过 ultralytics 导出 YOLO `.pt` → `.onnx` |
| Ultralytics | ≥ 8.4 | YOLO 导出与参考后处理 |
| PaddlePaddle + Paddle2ONNX | 可选 | PPOCRv4 导出流水线 |

转换环境的激活方式：

```bash
# 基于 Miniforge（推荐，recamera-rknn-dev skill 使用）
conda activate recamera-rknn-2.3.2

# 或基于 Miniconda（旧版基准测试环境）
eval "$(/home/seeed/miniconda3/bin/conda shell.bash hook)"
conda activate rknn_bench
```

> **约定**：本仓库中的所有模型均使用 **RKNN-Toolkit2 2.3.2** 转换，目标平台为 `rv1126b`。设备端运行时（`librknnrt.so`）必须与工具包版本匹配——混用版本会在 `load_rknn()` 时失败。

---

## 测试与验证

测试分为三个级别，从快速验证到完整 NPU 基准测试。

### 1. 转换验证（主机，无需设备）

运行 `convert_to_rknn.py` 后，验证 `.rknn` 是否正确生成：

```bash
# 文件存在且大小合理（例如 YOLO nano 约 4 MB）
ls -lh yolov8n_640x640_W8A8.rknn

# 可选：在 Toolkit 模拟器上对比 ONNX 与 RKNN 输出
python3 scripts/compare_onnx_rknn.py \
    --onnx yolov8n_optimized.onnx \
    --rknn yolov8n_640x640_W8A8.rknn \
    --image assets/bus.jpg
```

### 2. 设备端推理测试（功能正确性）

`bench_native` 会打印最后一帧的解码结果，即为正确性检查：

```bash
./scripts/bench_native.sh --only YOLOv8n
cat benchmark_results/native/YOLOv8n.log
```

预期输出结尾（bus.jpg → 4 个人 + 1 辆公交车）：

```
detections : 5 (best raw score 0.898)
  [0] person         0.808 box(1059,256,1278,581)
  [1] person         0.712 box(82,241,379,607)
  ...
```

框坐标已还原到源帧坐标系（解码后撤销 letterbox）。姿态模型附加 `+17kpt`，OBB 模型附加
`rot`，分割会为每个检测合成掩码，分类打印 top-5，深度打印米制 min/mean/max。

### 3. NPU 基准测试（性能测量）

```bash
./scripts/bench_native.sh            # 完整矩阵，100 次运行 / 20 次预热
```

每个模型的预期输出：

```
================ BENCHMARK ================
model      : models/yolov8n_640x640_W8A8.rknn
task       : det
input      : 640x640
runs       : 100 (warmup 20)
preprocess : avg   1.82 ms  min   1.75
npu        : avg  15.30 ms  min  14.48  (steady-state 65.4 fps)
postprocess: avg   0.68 ms  min   0.60
total      : avg  17.80 ms  min  17.00  max  21.48
fps        : 56.2
detections : 5 (best raw score 0.898)
===========================================
```

`scripts/bench_runner.py` 保留给语音（Zipformer）与 OCR 流水线，它们仍使用 Python 运行时。

### 通过 / 失败标准

- **转换通过**：`rknn.export_rknn()` 返回 0，文件大小符合预期。
- **精度通过**：在 `bus.jpg` 上 ONNX 与 RKNN 的余弦相似度 ≥ 0.99；在 conf ≥ 0.3 时检测结果差异不超过 ±1 个目标。
- **性能通过**：延迟在上表数值的 ±5% 以内（须锁频；NPU 若停留在 `dvfs` 调频会慢 25–35%）。

---

## 测试方法论

- **测试程序**：`native/bench_native.cpp`，aarch64 交叉编译，链接 Seeed 认证的
  RKNN Runtime 2.3.2（`librknnrt.so`，由 `native/fetch_libs.sh` 校验 SHA-256）
- **迭代次数**：每模型 100 次运行，20 次预热不计入
- **延迟**：单帧串行耗时——RGA letterbox 预处理 + 一次 `rknn_run` + 完整后处理
  （解码、NMS、掩码 / 深度图）
- **FPS**：`1000 / avg_latency_ms`
- **频率**：NPU devfreq 锁到 `userspace` @ 950 MHz，CPU governor 为 `performance`
  （1608 MHz），VENC/VPU 为 `performance`——每次运行前由程序自动设置
- **零拷贝**：输入是 `rknn_create_mem` 缓冲，RGA 直接写入（声明为 uint8 NHWC，由运行时
  映射到模型自身的 int8/fp16 输入）；输出同样零拷贝绑定，并保持原生 int8
- **解码**：候选筛选直接在 int8 上做（量化 scale 为正，排序不变），只有超过阈值的少量
  网格才反量化；分割与深度采用通道在外的顺序访存
- **输入帧**：`assets/bus.jpg` 解码为 1280×720 原始 RGB888 帧，再由 RGA 做 letterbox
- **INT8 校准**：200 张真实图像（COCO val2017 / ImageNet val），非随机噪声
- **转换**：RKNN-Toolkit2 流水线，`target_platform='rv1126b'`

### YOLO 代际对比

各任务最佳模型（端到端延迟，含预处理/后处理）：

| 任务 | 最佳模型 | FPS | 优势 |
|------|---------|-----|------|
| 目标检测 | YOLOv8n | 56.2 | Zoo 导出，DFL 头融合充分 |
| 姿态估计 | YOLOv8n-pose | 55.9 | 单个 65 通道输出头，张量数最少 |
| 旋转边界框 | YOLOv8n-obb | 59.5 | 框与类别拼接为单个输出头 |
| 实例分割 | YOLOv8n-seg | 40.7 | proto/系数开销最小 |
| 图像分类 | YOLO26n-cls | 254.3 | 轻量骨干，NPU 仅 2.2 ms |
| 深度估计 | YOLO26n-depth | 11.3 | 本仓库唯一的单目深度模型 |

---

## 许可证

RKNN 模型来源于各自的开源项目，许可证请参阅各模型的原始来源。基准测试脚本和结果采用 MIT 许可证。

---

## 相关资源

- [Seeed reCamera Pro](https://www.seeedstudio.com/reCamera-Pro-2GB.html)
- [Rockchip RKNN 模型库](https://github.com/airockchip/rknn_model_zoo)
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [K2 Zipformer](https://github.com/k2-fsa/icefall)
