# Native YOLO Benchmark Results (C++ / RV1126B)

Device: Seeed reCamera Pro (Rockchip RV1126B, Cortex-A7 x4 @ 1608 MHz, NPU @ 950 MHz).
Harness: `native/bench_native.cpp`, `--runs 100 --warmup 20`.
Input: `assets/bus.jpg` decoded to a 1280x720 raw RGB888 frame, letterboxed to the model input by RGA.
Latency = preprocess (RGA) + NPU inference + postprocess (decode + NMS), measured serially; FPS = 1000 / latency.

| Task | Model | Input | Pre (ms) | NPU (ms) | Post (ms) | Total (ms) | Min | Max | FPS |
|------|-------|-------|---------:|---------:|----------:|-----------:|----:|----:|----:|
| Object Detection | YOLO26n | 640x640 | 1.81 | 18.87 | 0.49 | **21.17** | 20.27 | 24.41 | **47.2** |
| Object Detection | YOLO11n | 640x640 | 1.79 | 17.67 | 0.65 | **20.11** | 19.32 | 21.71 | **49.7** |
| Object Detection | YOLO11s | 640x640 | 1.81 | 33.54 | 0.73 | **36.08** | 35.08 | 39.10 | **27.7** |
| Object Detection | YOLOv8n | 640x640 | 1.82 | 15.30 | 0.68 | **17.80** | 17.00 | 21.48 | **56.2** |
| Object Detection | YOLOv8s | 640x640 | 1.83 | 30.05 | 0.67 | **32.56** | 31.61 | 34.29 | **30.7** |
| Pose Estimation | YOLO26n-pose | 640x640 | 1.80 | 20.06 | 0.24 | **22.10** | 21.24 | 24.80 | **45.2** |
| Pose Estimation | YOLO11n-pose | 640x640 | 1.77 | 18.05 | 0.41 | **20.23** | 19.41 | 22.94 | **49.4** |
| Pose Estimation | YOLOv8n-pose | 640x640 | 1.81 | 15.74 | 0.33 | **17.89** | 16.96 | 20.05 | **55.9** |
| Instance Segmentation | YOLO26n-seg | 640x640 | 1.82 | 24.17 | 3.65 | **29.63** | 28.35 | 31.90 | **33.7** |
| Instance Segmentation | YOLO11n-seg | 640x640 | 1.81 | 22.11 | 4.12 | **28.04** | 26.61 | 30.28 | **35.7** |
| Instance Segmentation | YOLOv8n-seg | 640x640 | 1.82 | 19.90 | 2.88 | **24.59** | 23.34 | 26.66 | **40.7** |
| Oriented Bounding Box | YOLO26n-obb | 640x640 | 1.80 | 18.33 | 0.13 | **20.27** | 19.62 | 23.49 | **49.3** |
| Oriented Bounding Box | YOLO11n-obb | 640x640 | 1.81 | 16.98 | 0.13 | **18.92** | 18.02 | 20.48 | **52.9** |
| Oriented Bounding Box | YOLOv8n-obb | 640x640 | 1.79 | 14.91 | 0.12 | **16.81** | 16.06 | 19.54 | **59.5** |
| Classification | YOLO26n-cls | 224x224 | 1.74 | 2.16 | 0.02 | **3.93** | 3.47 | 4.90 | **254.3** |
| Classification | YOLO11n-cls | 224x224 | 1.79 | 9.97 | 0.03 | **11.78** | 11.24 | 14.66 | **84.9** |
| Depth Estimation | YOLO26n-depth | 640x640 | 1.73 | 84.80 | 2.32 | **88.85** | 86.48 | 95.51 | **11.3** |

## Per-model output sanity

- **YOLO26n** (det): 6 (best raw score 0.872)
- **YOLO11n** (det): 6 (best raw score 0.868)
- **YOLO11s** (det): 5 (best raw score 0.888)
- **YOLOv8n** (det): 5 (best raw score 0.898)
- **YOLOv8s** (det): 4 (best raw score 0.914)
- **YOLO26n-pose** (pose): 4 (best raw score 0.855)
- **YOLO11n-pose** (pose): 4 (best raw score 0.883)
- **YOLOv8n-pose** (pose): 3 (best raw score 0.863)
- **YOLO26n-seg** (seg): 5 (best raw score 0.896)
- **YOLO11n-seg** (seg): 5 (best raw score 0.860)
- **YOLOv8n-seg** (seg): 4 (best raw score 0.870)
- **YOLO26n-obb** (obb): 0 (best raw score 0.040)
- **YOLO11n-obb** (obb): 0 (best raw score 0.013)
- **YOLOv8n-obb** (obb): 0 (best raw score 0.133)
- **YOLO26n-cls** (cls): #654=0.1307 #908=0.1307 #408=0.1110 #717=0.0941 #627=0.0678
- **YOLO11n-cls** (cls): #864=0.0973 #751=0.0900 #404=0.0767 #717=0.0751 #627=0.0717
- **YOLO26n-depth** (depth): mean 4.928 m  min 1.668  max 16.093  (160x160 metric map -> 1280x720 uint8)
