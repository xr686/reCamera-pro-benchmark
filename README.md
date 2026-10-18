# reCamera Pro RKNN Model Benchmark

[![Models](https://img.shields.io/badge/models-25-blue)](./models/)
[![Device](https://img.shields.io/badge/device-reCamera%20Pro%20(RV1126B)-orange)](https://www.seeedstudio.com/)

Comprehensive RKNN model benchmarking for the **Seeed reCamera Pro** (Rockchip RV1126B, 2 GB RAM, aarch64). All models converted to RKNN format and benchmarked on-device with real NPU inference.

All YOLO numbers below come from the native C++ harness in [`native/`](./native/) — RGA
preprocessing into a zero-copy RKNN input buffer, int8 outputs decoded in place, and the
NPU/CPU clocks pinned to their top OPPs. The earlier Python (`rknn-toolkit-lite2`) YOLO
path was removed; OCR and STT still use their Python inference scripts.

---

## Quick Start

```bash
# Clone the repo (with Git LFS for model weights)
git lfs install
git clone https://github.com/Seeed-Projects/recamera-pro-benchmark.git
cd recamera-pro-benchmark

# Benchmark the whole YOLO matrix on the device (cross-compiles, stages, runs)
DEVICE_IP=192.168.42.1 ./scripts/bench_native.sh

# Or a single model / task family
./scripts/bench_native.sh --only YOLOv8n
./scripts/bench_native.sh --only det
```

---

## Repository Structure

```
recamera-pro-benchmark/
├── README.md                           # This file
├── AGENTS.md                           # Detailed benchmark methodology & rules
│
├── models/                             # All models organized by family
│   ├── yolo/                           # YOLO family (v8, 11, 26)
│   │   ├── tools/                      # ONNX optimization scripts
│   │   │   ├── optimize_yolo_onnx.py   # YOLOv8/YOLO11 ONNX optimizer
│   │   │   └── optimize_yolo26_onnx.py # YOLO26 ONNX optimizer
│   │   ├── yolov8_det/                 # YOLOv8 Detection (n, s)
│   │   ├── yolov8_pose/                # YOLOv8 Pose Estimation
│   │   ├── yolov8_obb/                 # YOLOv8 Oriented Bounding Box
│   │   ├── yolov8_seg/                 # YOLOv8 Instance Segmentation
│   │   ├── yolo11_det/                 # YOLO11 Detection (n, s)
│   │   ├── yolo11_pose/               # YOLO11 Pose Estimation
│   │   ├── yolo11_obb/                # YOLO11 Oriented Bounding Box
│   │   ├── yolo11_seg/                # YOLO11 Instance Segmentation
│   │   ├── yolo11_cls/                # YOLO11 Classification
│   │   ├── yolo26_det/                # YOLO26 Detection
│   │   ├── yolo26_pose/               # YOLO26 Pose Estimation
│   │   ├── yolo26_obb/                # YOLO26 Oriented Bounding Box
│   │   ├── yolo26_seg/                # YOLO26 Instance Segmentation
│   │   ├── yolo26_cls/                # YOLO26 Classification
│   │   └── yolo26_depth/              # YOLO26 Depth Estimation
│   │
│   ├── ppocr/                          # OCR pipeline
│   │   └── ppocrv4/                    # PPOCRv4 Det + Rec (INT8)
│   │
│   └── zipformer/                      # Speech-to-Text
│       └── (encoder + decoder + joiner, FP16)
│
├── native/                             # C++ benchmark harness (YOLO inference path)
│   ├── bench_native.cpp                # zero-copy RKNN + RGA + int8 decode, every YOLO task
│   ├── fetch_libs.sh                   # fetches rknn_api.h / librknnrt.so / RGA for the build
│   ├── CMakeLists.txt                  # aarch64 cross build, RPATH = /oem/usr/lib
│   └── build.sh                        # cross-compile with the reCamera Pro SDK toolchain
│
├── benchmark_results/                  # Benchmark data & logs
│   └── native/                         # per-model logs + results.md / results.csv
├── scripts/                            # Conversion & benchmark scripts
│   └── bench_native.sh                 # build + stage + run the full YOLO matrix
├── assets/                             # Test images
├── calibration/                        # Calibration datasets
└── doc/                                # Reference documents
```

Each model sub-directory is self-contained with:
- `*.onnx` — Original ONNX model
- `*_optimized.onnx` — Optimized ONNX for RKNN (where applicable)
- `*_W8A8.rknn` / `*_FP16.rknn` — Converted RKNN model
- `convert_to_rknn.py` — ONNX → RKNN conversion script
- `infer.py` — Device inference script (OCR / STT only; YOLO uses `native/bench_native`)
- `calibration_200.txt` — Calibration image list
- `result_*.jpg` — Sample inference results
- `README.md` — Model-specific documentation

---

## Models Overview

**25 RKNN models** across 3 families and 9 task categories, benchmarked on RV1126B NPU.

Latency is the full serial frame path — **Pre** (RGA letterbox into the zero-copy input
buffer) + **NPU** (one `rknn_run`) + **Post** (decode, NMS, masks/depth map) — measured with
`--runs 100 --warmup 20`. FPS = 1000 / latency. Raw logs: `benchmark_results/native/results.md`.

### Object Detection

| Model | Input | Precision | Pre (ms) | NPU (ms) | Post (ms) | Latency (ms) | FPS |
|-------|-------|-----------|---------:|---------:|----------:|-------------:|----:|
| YOLO26n | 640×640 | INT8 | 1.81 | 18.87 | 0.49 | **21.17** | **47.2** |
| YOLO11n | 640×640 | INT8 | 1.79 | 17.67 | 0.65 | **20.11** | **49.7** |
| YOLO11s | 640×640 | INT8 | 1.81 | 33.54 | 0.73 | **36.08** | **27.7** |
| YOLOv8n | 640×640 | INT8 | 1.82 | 15.30 | 0.68 | **17.80** | **56.2** |
| YOLOv8s | 640×640 | INT8 | 1.83 | 30.05 | 0.67 | **32.56** | **30.7** |

### Pose Estimation

| Model | Input | Precision | Pre (ms) | NPU (ms) | Post (ms) | Latency (ms) | FPS |
|-------|-------|-----------|---------:|---------:|----------:|-------------:|----:|
| YOLO26n-pose | 640×640 | INT8 | 1.80 | 20.06 | 0.24 | **22.10** | **45.2** |
| YOLO11n-pose | 640×640 | INT8 | 1.77 | 18.05 | 0.41 | **20.23** | **49.4** |
| YOLOv8n-pose | 640×640 | INT8 | 1.81 | 15.74 | 0.33 | **17.89** | **55.9** |

### Instance Segmentation

| Model | Input | Precision | Pre (ms) | NPU (ms) | Post (ms) | Latency (ms) | FPS |
|-------|-------|-----------|---------:|---------:|----------:|-------------:|----:|
| YOLO26n-seg | 640×640 | INT8 | 1.82 | 24.17 | 3.65 | **29.63** | **33.7** |
| YOLO11n-seg | 640×640 | INT8 | 1.81 | 22.11 | 4.12 | **28.04** | **35.7** |
| YOLOv8n-seg | 640×640 | INT8 | 1.82 | 19.90 | 2.88 | **24.59** | **40.7** |

Post includes the per-detection mask assembly (`sigmoid(coef · proto)` inside each box).

### Oriented Bounding Box (OBB)

| Model | Input | Precision | Pre (ms) | NPU (ms) | Post (ms) | Latency (ms) | FPS |
|-------|-------|-----------|---------:|---------:|----------:|-------------:|----:|
| YOLO26n-obb | 640×640 | INT8 | 1.80 | 18.33 | 0.13 | **20.27** | **49.3** |
| YOLO11n-obb | 640×640 | INT8 | 1.81 | 16.98 | 0.13 | **18.92** | **52.9** |
| YOLOv8n-obb | 640×640 | INT8 | 1.79 | 14.91 | 0.12 | **16.81** | **59.5** |

OBB models are DOTA-trained, so a street-level `bus.jpg` yields 0 detections at conf 0.25
(best raw class score 0.01–0.13); the decode + rotated-NMS path still runs on every frame.

### Classification

| Model | Input | Precision | Pre (ms) | NPU (ms) | Post (ms) | Latency (ms) | FPS |
|-------|-------|-----------|---------:|---------:|----------:|-------------:|----:|
| YOLO26n-cls | 224×224 | INT8 | 1.74 | 2.16 | 0.02 | **3.93** | **254.3** |
| YOLO11n-cls | 224×224 | FP16 in | 1.79 | 9.97 | 0.03 | **11.78** | **84.9** |

### Depth Estimation

| Model | Input | Precision | Pre (ms) | NPU (ms) | Post (ms) | Latency (ms) | FPS |
|-------|-------|-----------|---------:|---------:|----------:|-------------:|----:|
| YOLO26n-depth | 640×640 | INT8 | 1.73 | 84.80 | 2.32 | **88.85** | **11.3** |

Post is the metric decode (`exp(clip(raw, ±10)) · e^cal_b`, resolved through a 256-entry
int8 lookup table) plus an RGA upscale of the normalised map to the source frame.

> **Power column dropped for the YOLO tables.** The board exposes no power telemetry
> (only SARADC and the IMU), so the earlier 5.0–5.6 W figures came from an external USB-C
> meter during the Python runs and cannot be reproduced per-model here. Expect the same
> ~5 W board draw under sustained NPU load.

### OCR (PPOCRv4)

| Component | Input | Precision | Latency (ms)* | FPS* | Power (W) |
|-----------|-------|-----------|---------------|------|-----------|
| ppocrv4_det | 480×480 | INT8 | 58.62 | 17.1 | 5.0 |
| ppocrv4_rec | 48×320 | INT8 | (combined) | (combined) | |

\* OCR latency/FPS are the combined detection + recognition pipeline; speed varies with text content.

### Speech-to-Text (Zipformer)

| Component | Input | Precision | Latency (ms) | FPS | Power (W) |
|-----------|-------|-----------|--------------|-----|-----------|
| zipformer_encoder | 103×80 | FP16 | 1834.9 (group total) | / | 5.2 |
| zipformer_decoder | 1×2 | FP16 | | / | |
| zipformer_joiner | 1×512 | FP16 | | / | |

| Audio | Total Latency | RTF |
|-------|--------------|-----|
| 5.61 s test.wav | 1834.9 ms | 0.33 |
| 5.00 s microphone | 1580.1 ms | 0.32 |

RTF < 1.0 means real-time capable (inference faster than audio duration).

---

## Per-Model Directory Usage

Each model directory contains everything needed for conversion and deployment:

### Convert ONNX → RKNN (on x86_64 host)

```bash
cd models/yolo/yolov8_det
python3 convert_to_rknn.py
```

Uses RKNN-Toolkit2 with `target_platform='rv1126b'` and 200-image calibration.

### Build & Run the native harness

```bash
# 1. Cross-compile (needs the reCamera Pro SDK toolchain; RECAMERA_SDK_ROOT overrides)
cd native && ./build.sh          # fetch_libs.sh pulls rknn_api.h / librknnrt.so / RGA first

# 2. From the repo root: build + push binary, test frame and models, run everything
./scripts/bench_native.sh                       # 100 runs / 20 warmup per model
RUNS=30 WARMUP=10 ./scripts/bench_native.sh     # quick pass
./scripts/bench_native.sh --only seg            # one task family
./scripts/bench_native.sh --report-only         # regenerate results.md from the logs
```

Or drive the binary directly on the device:

```bash
scp native/build/bench_native root@192.168.42.1:/userdata/benchmark/
ssh root@192.168.42.1 \
  'cd /userdata/benchmark && LD_LIBRARY_PATH=/oem/usr/lib ./bench_native \
     --model models/yolov8n_640x640_W8A8.rknn --image bench.rgb --runs 100 --warmup 20'
```

### bench_native options

| Option | Meaning |
|--------|---------|
| `--model <file.rknn>` | Model to benchmark (required) |
| `--image <raw.rgb>` | Raw RGB888 input frame (default 1280×720, see `--src`) |
| `--src WxH` | Geometry of the raw frame, default `1280x720` |
| `--runs N` / `--warmup N` | Timed / discarded iterations (default 100 / 20) |
| `--conf F` / `--nms F` | Detection thresholds (default 0.25 / 0.45) |
| `--task T` | Force `det`/`pose`/`seg`/`obb`/`cls`/`depth` (auto-detected from the head layout) |
| `--dump-io` | Print input/output tensor attributes and exit |
| `--trace-post` | Per-stage post-processing timing (depth) |

One binary covers all six YOLO tasks: the head layout (DFL vs direct boxes, class/objectness
channels, keypoint, mask-coefficient, angle and proto tensors) is inferred from the output
tensor names and shapes, so YOLOv8 / YOLO11 / YOLO26 exports need no per-model code.

---

## ONNX Optimization

YOLO models exported from PyTorch (`*.pt`) produce ONNX graphs that include post-processing decode operations not well-suited for NPU execution. The optimization scripts in `models/yolo/tools/` strip these CPU-bound operations:

```bash
# Optimize YOLOv8/YOLO11 ONNX
python3 models/yolo/tools/optimize_yolo_onnx.py --model yolov8n.onnx --output yolov8n_optimized.onnx

# Optimize YOLO26 ONNX
python3 models/yolo/tools/optimize_yolo26_onnx.py --model yolo26n.onnx --output yolo26n_optimized.onnx
```

Key optimizations:
- Strip DFL decode, anchor generation, NMS, and sigmoid operations
- Output raw spatial feature maps for NPU-accelerated inference
- Post-processing runs on CPU in the inference script

---

## Device Environment

| Component | Spec |
|-----------|------|
| Device | Seeed reCamera Pro |
| SoC | Rockchip RV1126B |
| CPU | ARM Cortex-A7, 4 cores |
| RAM | 1.9 GB (total) |
| NPU | 2.0 TOPS |
| RKNN Runtime | 2.3.2 |
| Kernel | Linux 6.1.x |
| Host Toolkit | RKNN-Toolkit2 2.3.2 (x86_64) |

---

## Host Environment & Toolchain

Before converting or benchmarking models, the host must have the following toolchain installed.

| Component | Version | Purpose |
|-----------|---------|---------|
| Host OS | Ubuntu 22.04 (x86_64) / WSL2 | Build & conversion host |
| Python | 3.10 | Conversion scripts + OCR/STT inference |
| reCamera Pro SDK | v1.0.10 | aarch64 cross toolchain & sysroot for `native/` (`RECAMERA_SDK_ROOT`) |
| CMake | ≥ 3.16 | Native harness build |
| ffmpeg | any | Decodes `assets/bus.jpg` into the raw RGB888 benchmark frame |
| sshpass | any | Device staging in `scripts/bench_native.sh` |
| Conda env | `recamera-rknn-2.3.2` (miniforge3) | ONNX → RKNN conversion |
| RKNN-Toolkit2 | **2.3.2** | Model conversion, `target_platform='rv1126b'` |
| ONNX | ≥ 1.14 | ONNX graph inspection & optimization |
| ONNX Runtime | ≥ 1.16 | CPU-side accuracy comparison |
| NumPy | 1.24 | Tensor math |
| OpenCV | 4.x | Image I/O, drawing, NMS |
| PyTorch (optional) | 2.x | Export YOLO `.pt` → `.onnx` via ultralytics |
| Ultralytics | ≥ 8.4 | YOLO export & reference post-processing |
| PaddlePaddle + Paddle2ONNX | optional | PPOCRv4 export pipeline |

The conversion environment is activated with:

```bash
# Miniforge-based (recommended, used by recamera-rknn-dev skill)
conda activate recamera-rknn-2.3.2

# Or Miniconda-based (legacy benchmark environment)
eval "$(/home/seeed/miniconda3/bin/conda shell.bash hook)"
conda activate rknn_bench
```

> **Contract**: All models in this repo are converted with **RKNN-Toolkit2 2.3.2** targeting `rv1126b`. The device runtime (`librknnrt.so`) must match the toolkit version — mixing versions will fail at `load_rknn()`.

---

## Testing & Validation

There are three levels of testing, from quick sanity to full NPU benchmark.

### 1. Conversion Sanity (host, no device)

After `convert_to_rknn.py`, verify the `.rknn` was produced correctly:

```bash
# File exists and has reasonable size (e.g., ~4 MB for YOLO nano)
ls -lh yolov8n_640x640_W8A8.rknn

# Optional: compare ONNX vs RKNN outputs on the Toolkit simulator
python3 scripts/compare_onnx_rknn.py \
    --onnx yolov8n_optimized.onnx \
    --rknn yolov8n_640x640_W8A8.rknn \
    --image assets/bus.jpg
```

### 2. On-device Inference Test (functional correctness)

`bench_native` prints the decoded result of the last frame, which is the correctness check:

```bash
./scripts/bench_native.sh --only YOLOv8n
cat benchmark_results/native/YOLOv8n.log
```

Expected tail (bus.jpg → 4 persons + 1 bus):

```
detections : 5 (best raw score 0.898)
  [0] person         0.808 box(1059,256,1278,581)
  [1] person         0.712 box(82,241,379,607)
  ...
```

Boxes are reported in source-frame coordinates (the letterbox is undone after decoding).
Pose models add `+17kpt`, OBB models add `rot`, segmentation assembles a mask per detection,
classification prints its top-5 and depth prints the metric min/mean/max.

### 3. NPU Benchmark (performance measurement)

```bash
./scripts/bench_native.sh            # full matrix, 100 runs / 20 warmup
```

Expected output per model:

```
================ BENCHMARK ================
model      : models/yolov8n_640x640_W8A8.rknn
task       : det
input      : 640x640
runs       : 100 (warmup 20)
preprocess : avg   1.82 ms  min   1.75
npu        : avg  15.30 ms  min  14.48  (steady-state 65.4 fps)
postprocess: avg   0.68 ms  min   0.60
total      : avg  17.80 ms  min  17.00  max  21.48
fps        : 56.2
detections : 5 (best raw score 0.898)
===========================================
```

`scripts/bench_runner.py` remains for the audio (Zipformer) and OCR pipelines, which still
use the Python runtime.

### Pass / Fail Criteria

- **Conversion pass**: `rknn.export_rknn()` returns 0 and file size matches expectation.
- **Accuracy pass**: ONNX vs RKNN cosine similarity ≥ 0.99 on `bus.jpg`; detections match within ±1 object at conf ≥ 0.3.
- **Performance pass**: Latency within ±5 % of the numbers in the tables above (clocks pinned;
  an unpinned `dvfs` NPU will read 25–35 % slower).

---

## Methodology

- **Harness**: `native/bench_native.cpp`, cross-compiled aarch64, linked against the
  Seeed-qualified RKNN Runtime 2.3.2 (`librknnrt.so`, SHA-256 verified by `native/fetch_libs.sh`)
- **Iterations**: 100 runs per model, 20 warmup excluded
- **Latency**: one serial frame — RGA letterbox preprocessing + a single `rknn_run` +
  full post-processing (decode, NMS, masks / depth map)
- **FPS**: `1000 / avg_latency_ms`
- **Clocks**: NPU devfreq pinned to `userspace` @ 950 MHz, CPU governor `performance`
  (1608 MHz), VENC/VPU `performance` — set by the harness before every run
- **Zero copy**: the input is an `rknn_create_mem` buffer that RGA writes into directly
  (declared uint8 NHWC so the runtime maps it onto the model's own int8/fp16 input);
  outputs are bound zero-copy and stay in their native int8 domain
- **Decode**: candidate filtering runs on raw int8 (ordering is preserved because the
  quantisation scale is positive), so only the cells above the confidence threshold are
  dequantised; segmentation and depth use cache-friendly channel-outer sweeps
- **Input frame**: `assets/bus.jpg` decoded to a 1280×720 raw RGB888 frame, letterboxed by RGA
- **INT8 Calibration**: 200 real images from COCO val2017 / ImageNet val — not random noise
- **Conversion**: RKNN-Toolkit2 pipeline, `target_platform='rv1126b'`

### YOLO Generation Comparison

Best-performing model per task (end-to-end latency, including pre/post-processing):

| Task | Best Model | FPS | Advantage |
|------|-----------|-----|-----------|
| Detection | YOLOv8n | 56.2 | Zoo export, well-fused DFL head |
| Pose | YOLOv8n-pose | 55.9 | Single 65-channel head, fewest outputs |
| OBB | YOLOv8n-obb | 59.5 | Concatenated box+class head |
| Segmentation | YOLOv8n-seg | 40.7 | Smallest proto/coeff cost |
| Classification | YOLO26n-cls | 254.3 | Lightweight backbone, 2.2 ms NPU |
| Depth | YOLO26n-depth | 11.3 | Only monocular depth model in the set |

### Why the native path is faster than the old Python numbers

The previous tables were produced by `rknn-toolkit-lite2` in Python with unpinned clocks.
Same models, same device, same input:

| Model | Python (ms / FPS) | Native C++ (ms / FPS) | Speedup |
|-------|------------------:|----------------------:|--------:|
| YOLO26n | 37.48 / 26.7 | 21.17 / 47.2 | 1.77× |
| YOLO11n | 38.70 / 25.8 | 20.11 / 49.7 | 1.92× |
| YOLO11s | 58.50 / 17.1 | 36.08 / 27.7 | 1.62× |
| YOLOv8n | 36.35 / 27.5 | 17.80 / 56.2 | 2.04× |
| YOLOv8s | 37.25 / 26.8 | 32.56 / 30.7 | 1.14× |
| YOLO26n-pose | 37.31 / 26.8 | 22.10 / 45.2 | 1.69× |
| YOLO11n-pose | 38.19 / 26.2 | 20.23 / 49.4 | 1.89× |
| YOLOv8n-pose | 36.92 / 27.1 | 17.89 / 55.9 | 2.06× |
| YOLO26n-seg | 55.11 / 18.1 | 29.63 / 33.7 | 1.86× |
| YOLO11n-seg | 54.59 / 18.3 | 28.04 / 35.7 | 1.95× |
| YOLOv8n-seg | 61.89 / 16.2 | 24.59 / 40.7 | 2.52× |
| YOLO26n-obb | 33.57 / 29.8 | 20.27 / 49.3 | 1.66× |
| YOLO11n-obb | 34.40 / 29.1 | 18.92 / 52.9 | 1.82× |
| YOLOv8n-obb | 35.56 / 28.1 | 16.81 / 59.5 | 2.12× |
| YOLO26n-cls | 6.57 / 152.2 | 3.93 / 254.3 | 1.67× |
| YOLO11n-cls | 15.20 / 65.7 | 11.78 / 84.9 | 1.29× |
| YOLO26n-depth | 99.16 / 10.1 | 88.85 / 11.3 | 1.12× |

Where the time went, in decreasing order of impact:

1. **Clocks** — leaving the NPU on `dvfs` costs 25–35 %; pinning it to 950 MHz is free speed.
2. **Output dequantisation** — `rknn-toolkit-lite2` converts every int8 output tensor to a
   fresh float32 NumPy array on each call (~5 ms at 640×640). The native path keeps int8 and
   only dequantises the few hundred cells that survive the threshold.
3. **Input copies** — the Python path built a NumPy uint8 array and copied it into the runtime
   buffer; RGA now writes straight into the zero-copy `rknn_create_mem` input.
4. **Post-processing in NumPy** — Python-level loops, `.astype(np.uint8)` round-trips and
   per-stride transposes cost milliseconds; the same decode is ~0.1–0.7 ms in C++.
5. **Language overhead** — the per-call interpreter cost is real but small (~0.5 ms) next to
   the four items above; most of the win is zero-copy + int8 + pinned clocks, not C++ itself.

---

## License

RKNN models are derived from their respective open-source projects. See each model's original source for license terms. Benchmark scripts and results are MIT-licensed.

---

## Related Resources

- [Seeed reCamera Pro](https://www.seeedstudio.com/reCamera-Pro-2GB.html)
- [Rockchip RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [K2 Zipformer](https://github.com/k2-fsa/icefall)
