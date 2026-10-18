# YOLO26n Pose - RV1126B

## Model Info
- **Architecture**: YOLO26n-pose
- **Input**: 640x640 RGB
- **Output**: 9 tensors (keypoint×3 + bbox×3 + objectness×3)
- **Quantization**: INT8 W8A8
- **Source**: ultralytics v8.4.154 (`yolo26n-pose.pt`)

## ONNX Optimization
- `keypoint_stride_N`: [1, 51, H, W] (17 keypoints × 3 values)
- `bbox_stride_N`: [1, 4, H, W]
- `objectness_stride_N`: [1, 1, H, W] (needs sigmoid)

## Files
- `yolo26n-pose.onnx` - Original ONNX export
- `yolo26n-pose_optimized.onnx` - Optimized ONNX
- `yolo26n-pose_640x640_W8A8.rknn` - RKNN INT8 model (4.81 MB)
- `../../native/bench_native` — native C++ inference/benchmark harness (repo `native/`)

## Benchmark (RV1126B Device, 100 runs)
- Avg latency: 21.17 ms
- FPS: 47.2
- Detections: 4 persons with 17 keypoints each

## Usage
```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO26n-pose
```

## Native C++ Benchmark (RV1126B)

The Python `infer.py` path in this directory was replaced by the native harness in
[`native/`](../../native/): RGA letterboxes straight into a zero-copy RKNN input
buffer, the int8 outputs are decoded in place, and the NPU/CPU clocks are pinned to
their top OPPs.

| Model | Pre (ms) | NPU (ms) | Post (ms) | End-to-end (ms) | FPS |
|-------|---------:|---------:|----------:|----------------:|----:|
| YOLO26n-pose | 1.80 | 20.06 | 0.24 | **22.10** | **45.2** |

`--runs 100 --warmup 20`, input frame from `assets/bus.jpg`, NPU pinned to 950 MHz, CPU governor `performance`. Full matrix: `benchmark_results/native/results.md`.

```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO26n-pose
```
