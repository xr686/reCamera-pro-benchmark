# YOLO26n Classification - RV1126B

## Model Info
- **Architecture**: YOLO26n-cls (ImageNet 1000 classes)
- **Input**: 224x224 RGB
- **Output**: [1, 1000] classification scores
- **Quantization**: INT8 W8A8
- **Source**: ultralytics v8.4.154 (`yolo26n-cls.pt`)

## Files
- `yolo26n-cls.onnx` - ONNX export (opset 12)
- `yolo26n-cls_224x224_W8A8.rknn` - RKNN INT8 model (3.65 MB)
- `../../native/bench_native` — native C++ inference/benchmark harness (repo `native/`)

## Benchmark (RV1126B Device, 100 runs)
- Avg latency: 21.17 ms
- FPS: 47.2
- Top-5 predictions printed on image

## Usage
```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO26n-cls
```

## Native C++ Benchmark (RV1126B)

The Python `infer.py` path in this directory was replaced by the native harness in
[`native/`](../../native/): RGA letterboxes straight into a zero-copy RKNN input
buffer, the int8 outputs are decoded in place, and the NPU/CPU clocks are pinned to
their top OPPs.

| Model | Pre (ms) | NPU (ms) | Post (ms) | End-to-end (ms) | FPS |
|-------|---------:|---------:|----------:|----------------:|----:|
| YOLO26n-cls | 1.74 | 2.16 | 0.02 | **3.93** | **254.3** |

`--runs 100 --warmup 20`, input frame from `assets/bus.jpg`, NPU pinned to 950 MHz, CPU governor `performance`. Full matrix: `benchmark_results/native/results.md`.

```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLO26n-cls
```
