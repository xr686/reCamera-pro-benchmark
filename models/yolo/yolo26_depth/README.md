# YOLO26n-Depth - RV1126B 深度估计

## 模型信息
- **架构**: YOLO26n-depth (DepthAnythingV2-based)
- **任务**: 单目深度估计
- **输入**: 640×640 RGB 图像
- **输出**: 160×160 深度图 (NPU) → 上采样至原图尺寸
- **量化**: INT8 W8A8
- **参数量**: 6.36M
- **来源**: ultralytics v8.4.154 (`yolo26n-depth.pt`)

## 性能 (RV1126B, 640×640, 100次推理)
| 指标 | 值 |
|------|-----|
| 平均延迟 | 88.85 ms |
| 最小延迟 | 86.48 ms |
| 最大延迟 | 95.51 ms |
| FPS | 11.3 |
| 模型大小 | 6.7 MB |

## 深度解码
NPU 输出原始深度预测 [1,1,160,160]，需在 CPU 端解码：
```python
depth = exp(clip(raw_depth, -10, 10))
depth = depth ** cal_a  # cal_a = 1.0
depth = depth * exp(cal_b)  # cal_b = -0.1938
```
解码后深度范围约 1.0-13.6 米。

## 文件说明
- `yolo26n-depth.pt` - PyTorch 权重
- `yolo26n-depth.onnx` - 原始 ONNX (opset 12)
- `yolo26n-depth_optimized.onnx` - 优化 ONNX (剥离后处理)
- `yolo26n-depth_640x640_W8A8.rknn` - RKNN INT8 模型
- `../../native/bench_native` — 原生 C++ 推理/基准测试程序（见仓库 `native/`）
- `convert_to_rknn.py` - RKNN 转换脚本
- `optimize_onnx.py` - ONNX 优化脚本

## 使用方法
```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO26n-depth
```

## 输出格式
- 终端报告：分阶段耗时（预处理 / NPU / 后处理）与深度统计（mean、min、max，单位米）
- `scripts/bench_native.sh` 会把完整日志写入 `benchmark_results/native/YOLO26n-depth.log`
- C++ 后处理：int8 → 米制深度用 256 项查找表完成（等价于 `exp(clip(raw,-10,10)) * exp(cal_b)`），
  再由 RGA 把归一化后的 uint8 深度图放大到源帧分辨率
- `result_bus.jpg`、`result_bus_depth.jpg` - 早期 Python 脚本生成的对比图（保留作参考）

## 优化说明
ONNX 优化剥离了 10 个后处理节点：
- Clip, Exp, Pow, Mul (深度解码)
- Resize (160×160 → 640×640 上采样)

这些操作在 CPU 端执行，减少 NPU 负担并提高量化精度。

## 原生 C++ 基准测试 (RV1126B)

本目录原有的 Python `infer.py` 推理路径已替换为 `native/` 下的 C++ 基准程序：
RGA 直接把 letterbox 结果写入零拷贝的 RKNN 输入缓冲，int8 输出原地解码，
NPU/CPU 频率锁定到最高档。

| 模型 | 预处理 (ms) | NPU (ms) | 后处理 (ms) | 端到端 (ms) | FPS |
|------|-----------:|--------:|-----------:|-----------:|----:|
| YOLO26n-depth | 1.73 | 84.80 | 2.32 | **88.85** | **11.3** |

`--runs 100 --warmup 20`，输入帧取自 `assets/bus.jpg`，NPU 锁定 950 MHz，CPU governor 为 `performance`。完整数据见 `benchmark_results/native/results.md`。

```bash
# 在仓库根目录执行：交叉编译 + 推送到设备 + 跑基准
./scripts/bench_native.sh --only YOLO26n-depth
```
