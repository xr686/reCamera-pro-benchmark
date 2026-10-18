# YOLO11 Detection Models

## 模型信息
- **模型**: yolo11n, yolo11s
- **任务**: 目标检测 (80 classes, COCO)
- **输入**: RGB 图像, 640×640
- **量化**: W8A8 (INT8)

## 文件说明
- `yolo11n.onnx` / `yolo11s.onnx`: ONNX 格式模型
- `yolo11n_640x640_W8A8.rknn` / `yolo11s_640x640_W8A8.rknn`: RKNN 量化模型
- `../../native/bench_native` — 原生 C++ 推理/基准测试程序（见仓库 `native/`）
- `convert_to_rknn.py`: ONNX → RKNN 转换脚本
- `calibration_200.txt`: 量化校准数据列表
- `result_bus_n.jpg` / `result_bus_s.jpg`: 推理结果示例

## 使用方法

### 推理
```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO11n
./scripts/bench_native.sh --only YOLO11s
```

### 转换
```bash
python3 convert_to_rknn.py
```

## 性能 (RV1126B NPU)
- **yolo11n**: 49.7 FPS, 20.11 ms
- **yolo11s**: 27.7 FPS, 36.08 ms

## 原生 C++ 基准测试 (RV1126B)

本目录原有的 Python `infer.py` 推理路径已替换为 `native/` 下的 C++ 基准程序：
RGA 直接把 letterbox 结果写入零拷贝的 RKNN 输入缓冲，int8 输出原地解码，
NPU/CPU 频率锁定到最高档。

| 模型 | 预处理 (ms) | NPU (ms) | 后处理 (ms) | 端到端 (ms) | FPS |
|------|-----------:|--------:|-----------:|-----------:|----:|
| YOLO11n | 1.79 | 17.67 | 0.65 | **20.11** | **49.7** |
| YOLO11s | 1.81 | 33.54 | 0.73 | **36.08** | **27.7** |

`--runs 100 --warmup 20`，输入帧取自 `assets/bus.jpg`，NPU 锁定 950 MHz，CPU governor 为 `performance`。完整数据见 `benchmark_results/native/results.md`。

```bash
# 在仓库根目录执行：交叉编译 + 推送到设备 + 跑基准
./scripts/bench_native.sh --only YOLO11n
./scripts/bench_native.sh --only YOLO11s
```
