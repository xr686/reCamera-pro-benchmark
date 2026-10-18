# YOLOv8 Detection Models for reCamera Pro (RV1126B)

## Models

### YOLOv8n (Nano)
- **Model**: `yolov8n_W8A8.rknn` (4.2 MB)
- **Source**: Rockchip Model Zoo (`yolov8n.onnx`)
- **Quantization**: INT8 W8A8
- **Input**: 640x640, RGB, uint8
- **Performance**: 17.80ms / 56.2 FPS (native C++ harness, pinned clocks)
- **Test Result**: 5 detections (4 persons + 1 bus)

### YOLOv8s (Small)
- **Model**: `yolov8s_W8A8.rknn` (12 MB)
- **Source**: Rockchip Model Zoo (`yolov8s.onnx`)
- **Quantization**: INT8 W8A8
- **Input**: 640x640, RGB, uint8
- **Performance**: 32.56ms / 30.7 FPS (native C++ harness, pinned clocks)
- **Test Result**: 4 detections (3 persons + 1 bus)

## Files

- `*.onnx` - Original ONNX models from Rockchip Zoo
- `*_W8A8.rknn` - Converted INT8 RKNN models for RV1126B
- `convert_to_rknn.py` - Conversion script
- `../../native/bench_native` — native C++ inference/benchmark harness (repo `native/`)
- `calibration_200.txt` - Calibration image list (200 COCO images)
- `result_bus.jpg` - YOLOv8n inference result
- `result_bus_s.jpg` - YOLOv8s inference result

## Usage

### Convert ONNX to RKNN
```bash
python convert_to_rknn.py \
  --model yolov8n.onnx \
  --calibration calibration_200.txt \
  --output yolov8n_W8A8.rknn \
  --target rv1126b
```

### Run Inference on Device
```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLOv8n
./scripts/bench_native.sh --only YOLOv8s
```

## Post-processing

The YOLOv8 Zoo models output raw feature maps (9 outputs for 3 scales):
- Bounding box features (64 channels, DFL with reg_max=16)
- Class scores (80 classes, sigmoid applied)
- Objectness scores (1 channel, sigmoid applied)

The inference script performs:
1. DFL decode (softmax + weighted sum)
2. Anchor generation for 3 scales (8, 16, 32)
3. Bounding box decoding
4. Confidence filtering (objectness × max_class_score)
5. Non-maximum suppression (NMS)

## Performance Comparison

| Model | Size | FPS | Latency | Detections |
|-------|------|-----|---------|------------|
| YOLOv8n | 4.2 MB | 56.2 | 17.80ms | 5 |
| YOLOv8s | 12 MB | 30.7 | 32.56ms | 4 |

## Notes

- Models use Rockchip Zoo preprocessing: mean=[0,0,0], std=[255,255,255]
- Input should be uint8 [0-255], normalized internally
- Calibration used 200 COCO val2017 images
- Tested on reCamera Pro (RV1126B) with NPU

## Native C++ Benchmark (RV1126B)

The Python `infer.py` path in this directory was replaced by the native harness in
[`native/`](../../native/): RGA letterboxes straight into a zero-copy RKNN input
buffer, the int8 outputs are decoded in place, and the NPU/CPU clocks are pinned to
their top OPPs.

| Model | Pre (ms) | NPU (ms) | Post (ms) | End-to-end (ms) | FPS |
|-------|---------:|---------:|----------:|----------------:|----:|
| YOLOv8n | 1.82 | 15.30 | 0.68 | **17.80** | **56.2** |
| YOLOv8s | 1.83 | 30.05 | 0.67 | **32.56** | **30.7** |

`--runs 100 --warmup 20`, input frame from `assets/bus.jpg`, NPU pinned to 950 MHz, CPU governor `performance`. Full matrix: `benchmark_results/native/results.md`.

```bash
# from the repo root: cross-compiles, stages and benchmarks on the device
./scripts/bench_native.sh --only YOLOv8n
./scripts/bench_native.sh --only YOLOv8s
```
