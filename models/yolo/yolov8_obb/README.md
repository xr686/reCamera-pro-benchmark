# YOLOv8n-obb — RV1126B RKNN Benchmark

## Model Info

| Field | Value |
|-------|-------|
| Model | YOLOv8n-obb |
| Source | Rockchip Model Zoo (`yolov8n-obb_zoo.onnx`) |
| Input | 640×640 RGB uint8 |
| Quantization | W8A8 (INT8) |
| Target | RV1126B |
| Calibration | 200 COCO val2017 images |
| Training | DOTA (aerial imagery, 15 classes) |

## Output Format

4 output tensors:
- `[1, 79, 80, 80]` — stride 8: 64 DFL + 15 DOTA class logits
- `[1, 79, 40, 40]` — stride 16
- `[1, 79, 20, 20]` — stride 32
- `[1, 1, 8400]` — angle sigmoid (maps to [0, π/4])

## Post-Processing

1. DFL decode (softmax on 64 channels → 4 distance values)
2. Sigmoid on 15 class logits → confidence
3. Anchor-based bbox decode → xyxy
4. Angle = sigmoid_value × π/4
5. Convert to rotated rectangle (cx, cy, w, h, angle)
6. Rotated NMS (cv2.rotatedRectangleIntersection)

## Results (bus.jpg)

Note: OBB is trained on DOTA aerial imagery, so street-level images produce low confidence detections.

| Metric | Value |
|--------|-------|
| Avg latency | 16.81 ms |
| Min latency | 16.06 ms |
| Max latency | 19.54 ms |
| FPS | 59.5 |
| Model size | 4.26 MB |

At conf=0.25: 0 detections (expected for non-aerial images)
At conf=0.05: 13 "small-vehicle" detections (low confidence)

## Usage

### Convert

```bash
python convert_to_rknn.py
```

### Benchmark (on device)
```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLOv8n-obb
```

## Native C++ Benchmark (RV1126B)

The Python `infer.py` path in this directory was replaced by the native harness in
[`native/`](../../native/): RGA letterboxes straight into a zero-copy RKNN input
buffer, the int8 outputs are decoded in place, and the NPU/CPU clocks are pinned to
their top OPPs.

| Model | Pre (ms) | NPU (ms) | Post (ms) | End-to-end (ms) | FPS |
|-------|---------:|---------:|----------:|----------------:|----:|
| YOLOv8n-obb | 1.79 | 14.91 | 0.12 | **16.81** | **59.5** |

`--runs 100 --warmup 20`, input frame from `assets/bus.jpg`, NPU pinned to 950 MHz, CPU governor `performance`. Full matrix: `benchmark_results/native/results.md`.

```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLOv8n-obb
```
