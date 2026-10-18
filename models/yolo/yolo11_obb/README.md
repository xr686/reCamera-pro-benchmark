# YOLO11 Oriented Bounding Box (OBB)

## 模型信息
- **模型**: yolo11n-obb
- **任务**: 旋转边界框检测 (15 classes, DOTA aerial)
- **输入**: RGB 图像, 640×640
- **输出**: 旋转边界框 + 类别
- **量化**: W8A8 (INT8)

## 文件说明
- `yolo11n-obb.onnx`: ONNX 格式模型
- `yolo11n-obb_640x640_W8A8.rknn`: RKNN 量化模型
- `../../native/bench_native` — 原生 C++ 推理/基准测试程序（见仓库 `native/`）
- `convert_to_rknn.py`: ONNX → RKNN 转换脚本
- `calibration_200.txt`: 量化校准数据列表

## 使用方法

### 推理
```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO11n-obb
```

### 转换
```bash
python3 convert_to_rknn.py
```

## 性能 (RV1126B NPU)
- 待测试

## 原生 C++ 基准测试 (RV1126B)

本目录原有的 Python `infer.py` 推理路径已替换为 `native/` 下的 C++ 基准程序：
RGA 直接把 letterbox 结果写入零拷贝的 RKNN 输入缓冲，int8 输出原地解码，
NPU/CPU 频率锁定到最高档。

| 模型 | 预处理 (ms) | NPU (ms) | 后处理 (ms) | 端到端 (ms) | FPS |
|------|-----------:|--------:|-----------:|-----------:|----:|
| YOLO11n-obb | 1.81 | 16.98 | 0.13 | **18.92** | **52.9** |

`--runs 100 --warmup 20`，输入帧取自 `assets/bus.jpg`，NPU 锁定 950 MHz，CPU governor 为 `performance`。完整数据见 `benchmark_results/native/results.md`。

```bash
# 在仓库根目录执行：交叉编译 + 推送到设备 + 跑基准
./scripts/bench_native.sh --only YOLO11n-obb
```
