# YOLOv8n-seg — RV1126B RKNN Benchmark

## Model Info

| Field | Value |
|-------|-------|
| Model | YOLOv8n-seg |
| Source | Rockchip Model Zoo (`yolov8n-seg_zoo.onnx`) |
| Input | 640×640 RGB uint8 |
| Quantization | W8A8 (INT8) |
| Target | RV1126B |
| Calibration | 200 COCO val2017 images |

## Output Format

13 output tensors (grouped by stride):

Per stride (8, 16, 32):
- bbox DFL: `[1, 64, H, W]` — needs DFL softmax decode
- cls sigmoid: `[1, 80, H, W]` — per-class probabilities (sigmoid in ONNX)
- cls sum: `[1, 1, H, W]` — confidence = sum of sigmoid scores (ReduceSum + Clip)
- mask coeffs: `[1, 32, H, W]` — mask weights

Plus proto: `[1, 32, 160, 160]` — proto mask feature map

## Post-Processing

1. DFL decode (softmax on 64 channels → 4 distance values)
2. Anchor-based bbox decode (distances → xyxy)
3. Confidence from cls_sum output (already computed in ONNX)
4. Class = argmax of 80 sigmoid class scores
5. Mask = sigmoid(mask_coeffs @ proto)
6. NMS on bboxes

## Results (bus.jpg)

| Metric | Value |
|--------|-------|
| Avg latency | 24.59 ms |
| Min latency | 23.34 ms |
| Max latency | 26.66 ms |
| FPS | 40.7 |
| Model size | 4.61 MB |

### Detection Comparison

| # | Ultralytics FP32 | RKNN INT8 |
|---|-----------------|-----------|
| 1 | person 0.888 | person 0.871 |
| 2 | person 0.852 | bus 0.847 |
| 3 | person 0.842 | person 0.839 |
| 4 | bus 0.807 | person 0.804 |
| 5 | person 0.343 | skateboard 0.523 |
| 6 | skateboard 0.335 | person 0.296 |

6/6 objects detected. Same classes, similar confidence. Mask pixel counts differ due to resolution scaling.

## Usage

### Convert

```bash
python convert_to_rknn.py
```

### Benchmark (on device)
```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLOv8n-seg
```

## Native C++ Benchmark (RV1126B)

The Python `infer.py` path in this directory was replaced by the native harness in
[`native/`](../../native/): RGA letterboxes straight into a zero-copy RKNN input
buffer, the int8 outputs are decoded in place, and the NPU/CPU clocks are pinned to
their top OPPs.

| Model | Pre (ms) | NPU (ms) | Post (ms) | End-to-end (ms) | FPS |
|-------|---------:|---------:|----------:|----------------:|----:|
| YOLOv8n-seg | 1.82 | 19.90 | 2.88 | **24.59** | **40.7** |

`--runs 100 --warmup 20`, input frame from `assets/bus.jpg`, NPU pinned to 950 MHz, CPU governor `performance`. Full matrix: `benchmark_results/native/results.md`.

```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLOv8n-seg
```
