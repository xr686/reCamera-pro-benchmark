# YOLO26n OBB - RV1126B

## Model Info
- **Architecture**: YOLO26n-obb (DOTA dataset, 15 classes)
- **Input**: 640x640 RGB
- **Output**: 9 tensors (cls×3 + bbox×3 + angle×3)
- **Quantization**: INT8 W8A8
- **Source**: ultralytics v8.4.154 (`yolo26n-obb.pt`)

## ONNX Optimization
- `cls_stride_N`: [1, 15, H, W] (sigmoid, DOTA classes)
- `bbox_stride_N`: [1, 4, H, W]
- `angle_stride_N`: [1, 1, H, W] (needs sigmoid decode)

Angle decode: `(sigmoid(raw) - 0.25) * π` → range [-π/4, 3π/4]

## Files
- `yolo26n-obb.onnx` - Original ONNX export
- `yolo26n-obb_optimized.onnx` - Optimized ONNX
- `yolo26n-obb_640x640_W8A8.rknn` - RKNN INT8 model (4.36 MB)
- `../../native/bench_native` — native C++ inference/benchmark harness (repo `native/`)

## Benchmark (RV1126B Device, 100 runs)
- Avg latency: 21.17 ms
- FPS: 47.2

## DOTA Classes
plane, ship, storage-tank, baseball-diamond, tennis-court,
basketball-court, ground-track-field, harbor, bridge,
large-vehicle, small-vehicle, helicopter, roundabout,
soccer-ball-field, swimming-pool

## Usage
```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO26n-obb
```

## Native C++ Benchmark (RV1126B)

The Python `infer.py` path in this directory was replaced by the native harness in
[`native/`](../../native/): RGA letterboxes straight into a zero-copy RKNN input
buffer, the int8 outputs are decoded in place, and the NPU/CPU clocks are pinned to
their top OPPs.

| Model | Pre (ms) | NPU (ms) | Post (ms) | End-to-end (ms) | FPS |
|-------|---------:|---------:|----------:|----------------:|----:|
| YOLO26n-obb | 1.80 | 18.33 | 0.13 | **20.27** | **49.3** |

`--runs 100 --warmup 20`, input frame from `assets/bus.jpg`, NPU pinned to 950 MHz, CPU governor `performance`. Full matrix: `benchmark_results/native/results.md`.

```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO26n-obb
```
