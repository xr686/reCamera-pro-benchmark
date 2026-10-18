# YOLO26n Segmentation - RV1126B

## Model Info
- **Architecture**: YOLO26n-seg
- **Input**: 640x640 RGB
- **Output**: 10 tensors (bbox×3 + cls×3 + mask_coeff×3 + proto)
- **Quantization**: INT8 W8A8
- **Source**: ultralytics v8.4.154 (`yolo26n-seg.pt`)

## ONNX Optimization
Strips decode, outputs spatial features:
- `bbox_stride_N`: [1, 4, H, W]
- `cls_stride_N`: [1, 80, H, W] (sigmoid)
- `mask_coeff_stride_N`: [1, 32, H, W]
- `proto`: [1, 32, 160, 160] (mask prototype)

## Files
- `yolo26n-seg.onnx` - Original ONNX export
- `yolo26n-seg_optimized.onnx` - Optimized ONNX
- `yolo26n-seg_640x640_W8A8.rknn` - RKNN INT8 model (4.76 MB)
- `../../native/bench_native` — native C++ inference/benchmark harness (repo `native/`)

## Benchmark (RV1126B Device, 100 runs)
- Avg latency: 21.17 ms
- FPS: 47.2
- Detections: 5 (4 persons, bus with segmentation masks)

## Usage
```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO26n-seg
```

## Native C++ Benchmark (RV1126B)

The Python `infer.py` path in this directory was replaced by the native harness in
[`native/`](../../native/): RGA letterboxes straight into a zero-copy RKNN input
buffer, the int8 outputs are decoded in place, and the NPU/CPU clocks are pinned to
their top OPPs.

| Model | Pre (ms) | NPU (ms) | Post (ms) | End-to-end (ms) | FPS |
|-------|---------:|---------:|----------:|----------------:|----:|
| YOLO26n-seg | 1.82 | 24.17 | 3.65 | **29.63** | **33.7** |

`--runs 100 --warmup 20`, input frame from `assets/bus.jpg`, NPU pinned to 950 MHz, CPU governor `performance`. Full matrix: `benchmark_results/native/results.md`.

```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO26n-seg
```
