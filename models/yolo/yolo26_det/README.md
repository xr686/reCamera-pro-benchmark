# YOLO26n Detection - RV1126B

## Model Info
- **Architecture**: YOLO26n (End-to-end, NMS-free, no DFL)
- **Input**: 640x640 RGB
- **Output**: 6 tensors (bbox×3 + cls×3 strides), grouped by branch
- **Quantization**: INT8 W8A8
- **Source**: ultralytics v8.4.154 (`yolo26n.pt`)

## Key Architecture Differences from YOLOv8/YOLO11
- **No DFL**: 4-channel direct bbox regression (not 64-channel DFL)
- **No anchor grid offsets**: Direct ltrb distance regression from cell centers
- **End-to-end**: Model includes decode in-graph (but we strip it for NPU optimization)

## ONNX Optimization
The optimized ONNX strips the post-decode nodes and outputs spatial features:
- `bbox_stride_N`: [1, 4, H, W] - direct bbox regression
- `cls_stride_N`: [1, 80, H, W] - sigmoid class scores

## Files
- `yolo26n.pt` - PyTorch weights
- `yolo26n.onnx` - Original ONNX export (opset 12)
- `yolo26n_optimized.onnx` - Optimized ONNX for RKNN
- `yolo26n_640x640_W8A8.rknn` - RKNN INT8 model (4.22 MB)
- `../../native/bench_native` — native C++ inference/benchmark harness (repo `native/`)

## Benchmark (RV1126B Device, 100 runs)
- Avg latency: 21.17 ms
- FPS: 47.2
- Detections: 6 (best raw score 0.872; the log prints the top 5: 4 persons + bus 0.742)

## Usage
```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO26n
```

## Native C++ Benchmark (RV1126B)

The Python `infer.py` path in this directory was replaced by the native harness in
[`native/`](../../native/): RGA letterboxes straight into a zero-copy RKNN input
buffer, the int8 outputs are decoded in place, and the NPU/CPU clocks are pinned to
their top OPPs.

| Model | Pre (ms) | NPU (ms) | Post (ms) | End-to-end (ms) | FPS |
|-------|---------:|---------:|----------:|----------------:|----:|
| YOLO26n | 1.81 | 18.87 | 0.49 | **21.17** | **47.2** |

`--runs 100 --warmup 20`, input frame from `assets/bus.jpg`, NPU pinned to 950 MHz, CPU governor `performance`. Full matrix: `benchmark_results/native/results.md`.

```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO26n
```
