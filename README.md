# reCamera Pro RKNN Model Benchmark

**English** · [简体中文](README-CN.md)

[![App benchmarks](https://img.shields.io/badge/app%20configs-270-blue)](#rv1126b-application-benchmarks)
[![Model directory](https://img.shields.io/badge/model%20directory-25-blue)](./models/)
[![Device](https://img.shields.io/badge/device-reCamera%20Pro%20(RV1126B)-orange)](https://www.seeedstudio.com/)

Comprehensive RKNN model benchmarking for the **Seeed reCamera Pro** (Rockchip RV1126B, 2 GB RAM, aarch64). All models converted to RKNN format and benchmarked on-device with real NPU inference.

The native YOLO tables below use results from the native C++ harness in [`native/`](./native/) — RGA
preprocessing into a zero-copy RKNN input buffer, int8 outputs decoded in place, and the
NPU/CPU clocks pinned to their top OPPs.

## RV1126B application benchmarks

The firmware V1.3.2 application suite covers **270 measured configurations across 100 task labels**. The 59 ImageNet classifier configurations are grouped under **50 representative models**. The CSV records batch-one Python application timings, sample counts, quality metrics and artifact hashes. The native YOLO tables below use their own timing protocol. [Per-model app demos](examples/README.md) · [Benchmark guide](benchmark_results/app/README.md) · [Model index](benchmark_results/app/RV1126B-ImageNet-model-index.csv) · [Test app](apps/npu-benchmark/README.md) · [RKNN downloads](models/app-architectures/README.md)

### Latest 16 classification models

Each model classified the same 100 Imagenette validation images against all 1,000 ImageNet classes. Rows are sorted by mean synchronous inference time.

All 270 configurations ran on firmware V1.3.2 at FP16, batch size 1. FPS uses the sum of mean preprocessing, synchronous inference and postprocessing times. The second FPS includes Python output cleanup and garbage collection. A dash marks an unrecorded value. Timing excludes camera capture and display.

| Model | Input | Weights (MiB) | Pre (ms) | Inference mean / P95 (ms) | Post (ms) | FPS / with GC | Top-1 / Top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| [EfficientViT (CGA)](examples/efficientvit-cga/README.md) | 224×224 | 6.31 | 13.35 | 16.24 / 18.04 | 0.27 | 33.49 / 8.27 | 59% / 86% |
| [StarNet](examples/starnet/README.md) | 224×224 | 5.83 | 13.47 | 17.43 / 18.48 | 0.27 | 32.08 / 8.26 | 69% / 90% |
| [SwiftFormer](examples/swiftformer/README.md) | 224×224 | 7.67 | 12.88 | 22.80 / 24.41 | 0.26 | 27.82 / 7.93 | 79% / 93% |
| [SelecSLS](examples/selecsls/README.md) | 224×224 | 62.15 | 13.44 | 29.50 / 32.46 | 0.27 | 23.14 / 7.49 | 83% / 95% |
| [PP-HGNet](examples/hgnet/README.md) | 224×224 | 28.79 | 12.89 | 33.12 / 35.44 | 0.27 | 21.61 / 7.20 | 83% / 99% |
| [XCiT](examples/xcit/README.md) | 224×224 | 9.98 | 12.78 | 35.89 / 39.49 | 0.28 | 20.43 / 7.05 | 71% / 93% |
| [FocalNet](examples/focalnet/README.md) | 224×224 | 57.51 | 13.36 | 71.38 / 74.69 | 0.45 | 11.74 / 5.46 | 84% / 96% |
| [Xception](examples/xception/README.md) | 299×299 | 52.40 | 22.07 | 72.85 / 76.19 | 0.47 | 10.48 / 5.19 | 83% / 96% |
| [ConvMixer](examples/convmixer/README.md) | 224×224 | 48.25 | 13.06 | 74.53 / 78.02 | 0.47 | 11.36 / 5.35 | 81% / 96% |
| [DaViT](examples/davit/README.md) | 224×224 | 60.33 | 13.20 | 94.91 / 98.36 | 0.48 | 9.21 / 4.86 | 81% / 99% |
| [gMLP](examples/gmlp/README.md) | 224×224 | 40.82 | 13.66 | 112.65 / 121.71 | 0.47 | 7.89 / 4.41 | 80% / 98% |
| [GC ViT](examples/gcvit/README.md) | 224×224 | 38.41 | 13.49 | 145.07 / 157.97 | 0.47 | 6.29 / 3.86 | 78% / 96% |
| [Hiera](examples/hiera/README.md) | 224×224 | 77.25 | 13.33 | 211.55 / 240.49 | 0.47 | 4.44 / 3.07 | 86% / 99% |
| [TNT](examples/tnt/README.md) | 224×224 | 183.48 | 13.43 | 340.78 / 352.78 | 0.50 | 2.82 / 2.21 | 83% / 95% |
| [MaxViT](examples/maxvit/README.md) | 224×224 | 87.89 | 13.31 | 533.83 / 581.74 | 0.48 | 1.83 / 1.55 | 85% / 97% |
| [VOLO](examples/volo/README.md) | 224×224 | 90.52 | 12.94 | 593.36 / 628.61 | 0.49 | 1.65 / 1.42 | 82% / 95% |

VOLO ran four transpose operations on the CPU. Its synchronous inference time includes that work. [Conversion and accuracy details](apps/npu-benchmark/ARCHITECTURES-1.3.2.md)

### Remaining 254 model configurations

Expand a task group to see its measured configurations. Samples / calls counts distinct inputs and timed calls. Quality metrics use each task’s test subset. Lower WER and error values are better. Model names link to runnable examples. [Full CSV](benchmark_results/app/measurements.csv)

<details>
<summary>Other ImageNet classifiers (43)</summary>

| Model | Task | Input | Weights (MiB) | Samples / calls | Pre (ms) | Inference mean / P95 (ms) | Post (ms) | FPS / with GC | Task metrics |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [ShuffleNet V2 ×0.5](examples/shufflenet_v2_x0_5/README.md) | Image classification | 224×224 | 3.40 | 100 / 100 | 11.14 | 13.53 / 14.37 | 0.26 | 40.12 / 8.25 | Top-1 53.0%<br>Top-5 76.0% |
| [MobileNetV4 Conv Small](examples/mobilenetv4-small/README.md) | Image classification | 224×224 | 7.47 | 100 / 100 | 13.47 | 14.16 / 15.33 | 0.26 | 35.85 / 8.30 | Top-1 73.0%<br>Top-5 94.0% |
| [MNASNet 0.5](examples/mnasnet0_5/README.md) | Image classification | 224×224 | 4.57 | 100 / 100 | 11.16 | 14.44 / 15.27 | 0.26 | 38.68 / 8.32 | Top-1 69.0%<br>Top-5 90.0% |
| [PP-LCNet ×1](examples/lcnet/README.md) | Image classification | 224×224 | 6.27 | 100 / 100 | 13.43 | 14.45 / 15.29 | 0.27 | 35.51 / 8.38 | Top-1 70.0%<br>Top-5 90.0% |
| [MobileNetV3 Small](examples/mobilenet_v3_small/README.md) | Image classification | 224×224 | 5.77 | 100 / 100 | 11.22 | 14.72 / 15.83 | 0.25 | 38.18 / 8.24 | Top-1 67.0%<br>Top-5 88.0% |
| [MobileOne-S0](examples/mobileone-s0/README.md) | Image classification | 224×224 | 4.26 | 100 / 100 | 11.41 | 14.78 / 15.93 | 0.26 | 37.81 / 8.26 | Top-1 71.0%<br>Top-5 90.0% |
| [SqueezeNet 1.1](examples/squeezenet1_1/README.md) | Image classification | 224×224 | 2.61 | 100 / 100 | 11.14 | 15.48 / 16.37 | 0.26 | 37.19 / 8.25 | Top-1 53.0%<br>Top-5 75.0% |
| [RepVGG-A0](examples/repvgg-a0/README.md) | Image classification | 224×224 | 16.08 | 100 / 100 | 11.23 | 16.18 / 17.11 | 0.26 | 36.14 / 8.42 | Top-1 71.0%<br>Top-5 93.0% |
| [FasterNet-T0](examples/fasternet-t0/README.md) | Image classification | 224×224 | 7.93 | 100 / 100 | 12.73 | 16.20 / 17.71 | 0.28 | 34.24 / 8.40 | Top-1 75.0%<br>Top-5 92.0% |
| [ShuffleNet V2 x1.0](examples/shufflenet-v2/README.md) | Image classification | 224×224 | 8.20 | 300 / 300 | 2.84 | 16.56 / 19.63 | 0.46 | 50.36 / — | Top-1 62.3%<br>Top-5 86.3% |
| [GERNet-S](examples/gernet-s/README.md) | Image classification | 224×224 | 15.95 | 100 / 100 | 11.36 | 16.83 / 17.95 | 0.28 | 35.13 / 8.28 | Top-1 77.0%<br>Top-5 93.0% |
| [MobileNet V2](examples/mobilenet-v2/README.md) | Image classification | 224×224 | 7.03 | 300 / 300 | 2.94 | 16.90 / 19.65 | 0.47 | 49.25 / — | Top-1 66.7%<br>Top-5 88.0% |
| [FBNet-C](examples/fbnetc/README.md) | Image classification | 224×224 | 11.14 | 100 / 100 | 11.30 | 18.67 / 19.95 | 0.28 | 33.05 / 8.26 | Top-1 73.0%<br>Top-5 95.0% |
| [HardCoRe-NAS-A](examples/hardcorenas-a/README.md) | Image classification | 224×224 | 11.14 | 100 / 100 | 11.17 | 18.83 / 20.39 | 0.27 | 33.04 / 8.18 | Top-1 75.0%<br>Top-5 95.0% |
| [repghostnet_100](examples/repghost/README.md) | Image classification | 224×224 | 8.79 | 100 / 100 | 13.47 | 20.53 / 22.56 | 0.27 | 29.18 / 8.05 | Top-1 73.0%<br>Top-5 95.0% |
| [EdgeNeXt-XXS](examples/edgenext-xxs/README.md) | Image classification | 224×224 | 4.25 | 100 / 100 | 13.26 | 21.36 / 23.89 | 0.25 | 28.67 / 7.83 | Top-1 65.0%<br>Top-5 92.0% |
| [GhostNet 1.0](examples/ghostnet/README.md) | Image classification | 224×224 | 11.30 | 100 / 100 | 13.42 | 21.46 / 23.02 | 0.26 | 28.46 / 7.80 | Top-1 73.0%<br>Top-5 92.0% |
| [RegNetX-400MF](examples/regnet_x_400mf/README.md) | Image classification | 224×224 | 12.16 | 100 / 100 | 11.49 | 21.70 / 25.66 | 0.29 | 29.87 / 7.22 | Top-1 73.0%<br>Top-5 96.0% |
| [ConvNeXt Atto](examples/convnext-atto/README.md) | Image classification | 224×224 | 7.90 | 100 / 100 | 13.26 | 22.49 / 24.22 | 0.28 | 27.76 / 7.75 | Top-1 71.0%<br>Top-5 94.0% |
| [EfficientNet-B0](examples/efficientnet_b0/README.md) | Image classification | 224×224 | 12.46 | 100 / 100 | 13.43 | 22.59 / 24.40 | 0.27 | 27.56 / 7.59 | Top-1 77.0%<br>Top-5 96.0% |
| [ResNet-18](examples/resnet18/README.md) | Image classification | 224×224 | 22.45 | 300 / 300 | 3.38 | 22.93 / 29.57 | 0.51 | 37.28 / — | Top-1 63.3%<br>Top-5 86.0% |
| [ReXNet-100](examples/rexnet/README.md) | Image classification | 224×224 | 11.61 | 100 / 100 | 13.23 | 23.72 / 25.44 | 0.28 | 26.86 / 7.76 | Top-1 73.0%<br>Top-5 93.0% |
| [RegNetY-400MF](examples/regnety-004/README.md) | Image classification | 224×224 | 11.41 | 100 / 100 | 13.00 | 24.67 / 27.05 | 0.28 | 26.35 / 7.68 | Top-1 73.0%<br>Top-5 96.0% |
| [levit_128s](examples/levit/README.md) | Image classification | 224×224 | 17.56 | 100 / 100 | 13.25 | 25.11 / 28.16 | 0.27 | 25.88 / 7.87 | Top-1 79.0%<br>Top-5 95.0% |
| [ese_vovnet19b_dw](examples/vovnet/README.md) | Image classification | 224×224 | 13.19 | 100 / 100 | 13.34 | 25.66 / 27.63 | 0.27 | 25.46 / 7.63 | Top-1 76.0%<br>Top-5 96.0% |
| [DLA-34](examples/dla34/README.md) | Image classification | 224×224 | 30.25 | 100 / 100 | 11.10 | 26.30 / 27.56 | 0.26 | 26.55 / 7.68 | Top-1 77.0%<br>Top-5 92.0% |
| [GoogLeNet](examples/googlenet/README.md) | Image classification | 224×224 | 13.06 | 100 / 100 | 12.42 | 28.48 / 31.86 | 0.32 | 24.26 / 6.40 | Top-1 65.0%<br>Top-5 87.0% |
| [PVTv2-B0](examples/pvtv2-b0/README.md) | Image classification | 224×224 | 8.76 | 100 / 100 | 13.18 | 31.55 / 34.45 | 0.28 | 22.22 / 7.27 | Top-1 67.0%<br>Top-5 89.0% |
| [HRNet-W18 Small V2](examples/hrnet-small/README.md) | Image classification | 224×224 | 31.11 | 100 / 100 | 11.27 | 32.01 / 34.78 | 0.28 | 22.96 / 7.22 | Top-1 72.0%<br>Top-5 93.0% |
| [MixNet-S](examples/mixnet-s/README.md) | Image classification | 224×224 | 17.59 | 100 / 100 | 13.53 | 34.78 / 37.37 | 0.27 | 20.58 / 7.07 | Top-1 78.0%<br>Top-5 94.0% |
| [fastvit_t8](examples/fastvit/README.md) | Image classification | 256×256 | 24.78 | 100 / 100 | 17.09 | 34.84 / 37.20 | 0.27 | 19.16 / 6.86 | Top-1 74.0%<br>Top-5 92.0% |
| [DenseNet-121](examples/densenet121/README.md) | Image classification | 224×224 | 16.40 | 100 / 100 | 11.13 | 42.20 / 45.13 | 0.33 | 18.63 / 6.64 | Top-1 70.0%<br>Top-5 93.0% |
| [DPN-68](examples/dpn68/README.md) | Image classification | 224×224 | 27.75 | 100 / 100 | 13.46 | 46.58 / 48.06 | 0.33 | 16.57 / 6.44 | Top-1 80.0%<br>Top-5 93.0% |
| [ResNeXt-50 32×4d](examples/resnext50_32x4d/README.md) | Image classification | 224×224 | 50.46 | 100 / 100 | 11.08 | 47.26 / 49.47 | 0.32 | 17.05 / 6.47 | Top-1 85.0%<br>Top-5 100.0% |
| [ConvNeXt Tiny](examples/convnext_tiny/README.md) | Image classification | 224×224 | 55.99 | 100 / 100 | 11.02 | 54.09 / 56.23 | 0.44 | 15.26 / 6.03 | Top-1 84.0%<br>Top-5 99.0% |
| [MobileViT-XS](examples/mobilevit-xs/README.md) | Image classification | 256×256 | 8.14 | 100 / 100 | 17.10 | 62.68 / 66.54 | 0.44 | 12.46 / 5.52 | Top-1 68.0%<br>Top-5 90.0% |
| [coat_lite_tiny](examples/coat-lite/README.md) | Image classification | 224×224 | 13.46 | 100 / 100 | 13.45 | 65.17 / 69.83 | 0.45 | 12.65 / 5.64 | Top-1 71.0%<br>Top-5 95.0% |
| [skresnet18](examples/skresnet/README.md) | Image classification | 224×224 | 23.96 | 100 / 100 | 13.41 | 85.23 / 87.22 | 0.48 | 10.09 / 5.10 | Top-1 68.0%<br>Top-5 91.0% |
| [resnest14d](examples/resnest/README.md) | Image classification | 224×224 | 21.21 | 100 / 100 | 11.30 | 103.04 / 111.51 | 0.47 | 8.71 / 4.68 | Top-1 77.0%<br>Top-5 94.0% |
| [EfficientFormer-L1](examples/efficientformer/README.md) | Image classification | 224×224 | 24.10 | 100 / 100 | 12.91 | 118.96 / 121.85 | 0.27 | 7.57 / 4.46 | Top-1 82.0%<br>Top-5 95.0% |
| [poolformer_s12](examples/poolformer/README.md) | Image classification | 224×224 | 29.31 | 100 / 100 | 13.19 | 147.22 / 155.04 | 0.29 | 6.22 / 3.99 | Top-1 72.0%<br>Top-5 96.0% |
| [ViT-B/32](examples/vit_b_32/README.md) | Image classification | 224×224 | 179.19 | 100 / 100 | 11.27 | 187.56 / 197.35 | 0.48 | 5.02 / 3.34 | Top-1 83.0%<br>Top-5 97.0% |
| [Swin Transformer Tiny](examples/swin_t/README.md) | Image classification | 224×224 | 77.66 | 100 / 100 | 12.94 | 362.25 / 391.70 | 0.46 | 2.66 / 2.10 | Top-1 77.0%<br>Top-5 99.0% |

</details>

<details>
<summary>Detection, segmentation, pose and tracking (35)</summary>

| Model | Task | Input | Weights (MiB) | Samples / calls | Pre (ms) | Inference mean / P95 (ms) | Post (ms) | FPS / with GC | Task metrics |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [Depthwise pyramid · voc-prompt](examples/voc-prompt-depthwise/README.md) | Box-prompted instance segmentation | 96×96 | 0.13 | 80 / 80 | 0.90 | 4.81 / 5.62 | 0.21 | 169.17 / 10.23 | Foreground IoU 37.3% |
| [Convolutional pyramid · voc-prompt](examples/voc-prompt-plain/README.md) | Box-prompted instance segmentation | 96×96 | 0.21 | 80 / 80 | 0.90 | 4.81 / 5.32 | 0.21 | 169.08 / 10.48 | Foreground IoU 52.3% |
| [Pretrained SqueezeNet pyramid · voc-prompt](examples/voc-prompt-fire/README.md) | Box-prompted instance segmentation | 96×96 | 0.41 | 80 / 80 | 0.94 | 6.41 / 7.16 | 0.22 | 132.08 / 10.02 | Foreground IoU 60.2% |
| [Depthwise pyramid · voc-dominant](examples/voc-dominant-depthwise/README.md) | Dominant-object box regression | 96×96 | 0.16 | 80 / 80 | 0.89 | 3.83 / 4.19 | 0.63 | 186.80 / 10.44 | Mean box IoU 34.0%<br>Fraction with IoU ≥ 0.5 20.0% |
| [Convolutional pyramid · voc-dominant](examples/voc-dominant-plain/README.md) | Dominant-object box regression | 96×96 | 0.25 | 80 / 80 | 0.90 | 3.90 / 4.49 | 0.64 | 183.69 / 10.43 | Mean box IoU 36.4%<br>Fraction with IoU ≥ 0.5 27.5% |
| [Pretrained SqueezeNet pyramid · voc-dominant](examples/voc-dominant-fire/README.md) | Dominant-object box regression | 96×96 | 0.42 | 80 / 80 | 0.93 | 4.63 / 5.12 | 0.63 | 161.45 / 10.20 | Mean box IoU 42.8%<br>Fraction with IoU ≥ 0.5 38.8% |
| [DexiNed](examples/edge_detection_dexined_2024sep/README.md) | Edge detection | 640×480 | 67.49 | 4 / 4 | 47.23 | 1105.91 / 1109.18 | 20.00 | 0.85 / 0.78 | Edge F1 ±2 px 43.0% |
| [RetinaFace MobileNet](examples/RetinaFace_mobile320/README.md) | Face detection | 320×320 | 1.15 | 16 / 30 | 14.43 | 25.56 / 26.78 | 3.30 | 23.10 / 7.28 | F1@IoU0.5 42.9% |
| [SCRFD-500M](examples/scrfd500m_640_fp16/README.md) | Face detection | 640×640 | 1.57 | 16 / 30 | 7.82 | 79.10 / 80.96 | 74.70 | 6.19 / 3.90 | F1@IoU0.5 48.5% |
| [YuNet](examples/face_detection_yunet_2023mar/README.md) | Face detection | 640×640 | 0.50 | 16 / 30 | 39.34 | 93.97 / 97.46 | 43.17 | 5.67 / 3.67 | F1@IoU0.5 33.3% |
| [MediaPipe FaceMesh](examples/face_landmark_fp16/README.md) | Facial landmarks | 192×192 | 1.39 | 25 / 30 | 0.80 | 10.49 / 10.95 | 0.27 | 86.50 / 9.22 | 7-point NME (interocular) 7.0% |
| [MediaPipe Hand Landmarks](examples/handpose_estimation_mediapipe_2023feb/README.md) | Hand landmarks | 224×224 | 2.27 | 15 / 30 | 2.72 | 15.80 / 16.85 | 0.21 | 53.40 / 8.79 | Three-gesture accuracy 33.3% |
| [BlazePose Landmarks](examples/pose_estimation_mediapipe_2023mar/README.md) | Human pose | 256×256 | 3.09 | 9 / 30 | 4.22 | 38.84 / 41.23 | 0.20 | 23.12 / 7.12 | PCK@0.10 ROI 48.7% |
| [PP-HumanSeg](examples/human_segmentation_pphumanseg_2023mar/README.md) | Human segmentation | 192×192 | 3.68 | 12 / 30 | 5.82 | 22.80 / 24.95 | 2.49 | 32.14 / 7.92 | Human foreground IoU 77.9% |
| [YOLOv5s-seg](examples/yolov5s-seg/README.md) | Instance segmentation | 640×640 | 15.82 | 12 / 30 | 47.68 | 230.20 / 241.89 | 147.26 | 2.35 / 1.89 | Human foreground IoU 77.1%<br>F1@IoU0.5 74.6% |
| [LPD-YuNet](examples/license_plate_detection_lpd_yunet_2023mar/README.md) | License plate corners | 320×240 | 39.01 | 5 / 30 | 9.65 | 37.03 / 40.44 | 4.68 | 19.47 / 6.70 | F1@IoU0.5 100.0% |
| [M-LSD Tiny](examples/mlsd/README.md) | Line detection | 320×320 | 1.49 | 16 / 30 | 6.81 | 49.10 / 51.75 | 3.77 | 16.75 / 6.65 | Endpoint F1 at 10 px 92.8% |
| [NanoDet-Plus-m](examples/object_detection_nanodet_2022nov/README.md) | Object detection | 416×416 | 5.79 | 4 / 30 | 35.52 | 59.50 / 62.94 | 20.56 | 8.65 / 4.74 | F1@IoU0.5 22.2% |
| [YOLOv10n](examples/yolov10n/README.md) | Object detection | 640×640 | 5.77 | 32 / 32 | 44.09 | 122.83 / 133.57 | 60.81 | 4.39 / 3.13 | F1@IoU0.5 63.0% |
| [YOLOX-s](examples/object_detection_yolox_2022nov/README.md) | Object detection | 640×640 | 18.65 | 4 / 30 | 44.04 | 164.14 / 171.57 | 22.15 | 4.34 / 3.05 | F1@IoU0.5 53.8% |
| [YOLOv5s-ReLU](examples/yolov5-relu/README.md) | Object detection | 640×640 | 14.21 | 32 / 32 | 52.27 | 172.03 / 189.11 | 92.40 | 3.16 / 2.41 | F1@IoU0.5 64.6% |
| [GGHL](examples/gghl-dota/README.md) | Oriented detection | 512×512 | 132.78 | 8 / 8 | 29.60 | 259.30 / 265.25 | 17.74 | 3.26 / 2.47 | F1@IoU0.5 8.3% |
| [BlazePalm](examples/palm_detection_mediapipe_2023feb/README.md) | Palm detection | 192×192 | 2.21 | 15 / 30 | 2.11 | 18.06 / 19.54 | 3.86 | 41.61 / 8.31 | Single-hand detection rate 100.0% |
| [BlazePose Detector](examples/person_detection_mediapipe_2023mar/README.md) | Person detection | 224×224 | 18.49 | 9 / 30 | 7.26 | 22.29 / 23.97 | 4.64 | 29.24 / 7.73 | Top hip center PCK@0.10 ROI 25.0% |
| [Convolutional pyramid · Pet](examples/pet-mask-plain/README.md) | Pet segmentation | 96×96 | 0.21 | 74 / 74 | 0.91 | 4.69 / 5.31 | 0.21 | 171.97 / 10.40 | Foreground IoU 73.4% |
| [Depthwise pyramid · Pet](examples/pet-mask-depthwise/README.md) | Pet segmentation | 96×96 | 0.13 | 74 / 74 | 0.92 | 4.81 / 5.35 | 0.21 | 168.37 / 10.17 | Foreground IoU 52.5% |
| [Pretrained SqueezeNet pyramid · Pet](examples/pet-mask-fire/README.md) | Pet segmentation | 96×96 | 0.41 | 74 / 74 | 0.88 | 6.19 / 6.88 | 0.21 | 137.15 / 10.24 | Foreground IoU 82.1% |
| [U²-NetP](examples/u2netp/README.md) | Salient object detection | 320×320 | 3.41 | 20 / 30 | 24.24 | 244.98 / 253.75 | 2.65 | 3.68 / 2.71 | VOC foreground IoU 57.4% |
| [LR-ASPP MobileNetV3](examples/lraspp_mobilenet_v3_large/README.md) | Semantic segmentation | 256×256 | 7.56 | 20 / 30 | 14.79 | 68.40 / 75.75 | 95.58 | 5.59 / 3.66 | mIoU (nonempty class union) 62.5% |
| [FCN ResNet-50](examples/fcn_resnet50/README.md) | Semantic segmentation | 256×256 | 63.31 | 20 / 30 | 14.81 | 261.47 / 272.43 | 97.41 | 2.68 / 2.12 | mIoU (nonempty class union) 66.1% |
| [DeepLabV3 MobileNetV3](examples/deeplabv3_mobilenet_v3_large/README.md) | Semantic segmentation | 256×256 | 32.35 | 20 / 30 | 14.24 | 333.40 / 346.20 | 97.57 | 2.25 / 1.85 | mIoU (nonempty class union) 55.5% |
| [PP-LiteSeg](examples/pp_liteseg_cityscapes/README.md) | Semantic segmentation | 512×512 | 16.32 | 10 / 30 | 47.57 | 446.94 / 465.40 | 229.95 | 1.38 / 1.20 | Mean cross-dataset class IoU 52.1% |
| [VitTrack](examples/object_tracking_vittrack_2023sep/README.md) | Single-object tracking | 128×128 + 256×256 | 0.93 | 18 / 30 | 15.64 | 21.32 / 22.84 | 0.63 | 26.61 / 7.43 | Mean box IoU 76.4% |
| [PP-OCRv3 EN Det](examples/text_detection_en_ppocrv3_2023may/README.md) | Text localization | 320×320 | 1.96 | 12 / 30 | 16.04 | 35.89 / 38.51 | 4.21 | 17.81 / 6.64 | F1@IoU0.5 91.7% |
| [PP-OCRv3 CN Det](examples/text_detection_cn_ppocrv3_2023may/README.md) | Text localization | 320×320 | 1.96 | 12 / 30 | 15.81 | 35.96 / 38.84 | 4.11 | 17.89 / 6.72 | F1@IoU0.5 91.7% |

</details>

<details>
<summary>Visual recognition and retrieval (48)</summary>

| Model | Task | Input | Weights (MiB) | Samples / calls | Pre (ms) | Inference mean / P95 (ms) | Post (ms) | FPS / with GC | Task metrics |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [Patch MLP-Mixer](examples/fashion-mixer/README.md) | Clothing classification | 28×28 | 0.08 | 120 / 120 | 0.29 | 1.97 / 2.56 | 0.17 | 409.68 / 10.58 | Top-1 76.7% |
| [Network in Network](examples/fashion-nin/README.md) | Clothing classification | 28×28 | 0.11 | 120 / 120 | 0.30 | 2.07 / 2.86 | 0.17 | 393.79 / 10.67 | Top-1 80.8% |
| [Highway MLP](examples/fashion-highway/README.md) | Clothing classification | 28×28 | 0.25 | 120 / 120 | 0.29 | 2.10 / 2.65 | 0.17 | 390.86 / 10.58 | Top-1 85.0% |
| [LeNet-5](examples/lenet-digits/README.md) | Digit recognition | 32×32 | 0.22 | 80 / 80 | 0.28 | 2.18 / 2.84 | 0.16 | 382.00 / 10.42 | Top-1 98.8% |
| [MiniFASNet V2](examples/liveness_minifasnet_fp16/README.md) | Face anti-spoofing | 80×80 | 0.92 | 3 / 30 | 0.62 | 4.04 / 4.81 | 0.06 | 212.07 / 9.86 | Liveness accuracy 100.0% |
| [MiniFASNet V1SE](examples/liveness_minifasnet_v1se_fp16/README.md) | Face anti-spoofing | 80×80 | 1.04 | 3 / 30 | 0.63 | 4.50 / 5.00 | 0.07 | 192.56 / 9.80 | Liveness accuracy 100.0% |
| [ArcFace MobileFaceNet](examples/arcface_mbf_fp16/README.md) | Face embeddings | 112×112 | 6.97 | 60 / 60 | 1.06 | 9.18 / 10.21 | 0.24 | 95.38 / 9.32 | Identity Rank-1 100.0% |
| [SFace](examples/face_recognition_sface_2021dec/README.md) | Face embeddings | 112×112 | 18.58 | 60 / 60 | 1.90 | 12.39 / 14.06 | 0.24 | 68.84 / 9.11 | Identity Rank-1 92.6% |
| [EDiffIQA Tiny](examples/ediffiqa_tiny_jun2024/README.md) | Face image quality | 112×112 | 3.55 | 24 / 30 | 2.24 | 7.69 / 8.43 | 0.05 | 100.23 / 9.53 | Sharp image scored higher 100.0% |
| [MobileFaceNet FER](examples/facial_expression_recognition_mobilefacenet_2022july/README.md) | Facial expression recognition | 112×112 | 2.49 | 60 / 60 | 2.24 | 7.52 / 8.37 | 0.07 | 101.77 / 9.44 | Top-1 61.7% |
| [Multiscale CNN · flowers](examples/flowers-branches/README.md) | Flower classification | 80×80 | 0.13 | 100 / 100 | 0.71 | 3.36 / 3.90 | 0.19 | 234.97 / 10.32 | Top-1 63.0% |
| [SqueezeNet ridge probe · flowers](examples/flowers-squeezeprobe/README.md) | Flower classification | 80×80 | 1.54 | 100 / 100 | 0.71 | 4.07 / 4.61 | 0.18 | 201.84 / 10.46 | Top-1 69.0% |
| [Gabor filter bank + ridge · flowers](examples/flowers-gabor/README.md) | Flower classification | 80×80 | 0.14 | 100 / 100 | 0.74 | 5.17 / 5.99 | 0.17 | 164.73 / 10.24 | Top-1 55.0% |
| [ResNet-18 ridge probe · flowers](examples/flowers-resprobe/README.md) | Flower classification | 80×80 | 21.46 | 100 / 100 | 0.72 | 8.12 / 9.36 | 0.17 | 110.92 / 10.02 | Top-1 66.0% |
| [Multiscale CNN · eurosat](examples/eurosat-branches/README.md) | Land-use classification | 80×80 | 0.13 | 120 / 120 | 0.72 | 3.31 / 3.84 | 0.19 | 237.23 / 10.24 | Top-1 27.5% |
| [SqueezeNet ridge probe · eurosat](examples/eurosat-squeezeprobe/README.md) | Land-use classification | 80×80 | 1.54 | 120 / 120 | 0.71 | 4.10 / 4.68 | 0.18 | 200.43 / 10.33 | Top-1 86.7% |
| [Gabor filter bank + ridge · eurosat](examples/eurosat-gabor/README.md) | Land-use classification | 80×80 | 0.14 | 120 / 120 | 0.70 | 5.36 / 6.03 | 0.18 | 160.34 / 10.21 | Top-1 66.7% |
| [ResNet-18 ridge probe · eurosat](examples/eurosat-resprobe/README.md) | Land-use classification | 80×80 | 21.46 | 120 / 120 | 0.72 | 8.21 / 9.39 | 0.17 | 109.82 / 10.14 | Top-1 85.0% |
| [LPRNet](examples/lprnet/README.md) | License plate recognition | 94×24 | 1.72 | 6 / 30 | 0.55 | 5.04 / 5.61 | 0.44 | 165.74 / 9.84 | Exact match 100.0% |
| [XFeat](examples/xfeat/README.md) | Local feature matching | 256×192 | 1.74 | 24 / 30 | 0.89 | 13.51 / 14.99 | 12.50 | 37.18 / 8.43 | Inlier rate at 3 px 73.1%<br>Mean matches/pair 150.67 |
| [SuperPoint](examples/superpoint/README.md) | Local feature matching | 256×192 | 2.58 | 24 / 30 | 0.88 | 23.56 / 25.43 | 29.39 | 18.58 / 6.92 | Inlier rate at 3 px 93.6%<br>Mean matches/pair 159.50 |
| [TFeat](examples/tfeat/README.md) | Local patch retrieval | 32×32 | 1.21 | 192 / 192 | 0.75 | 2.40 / 2.94 | 0.29 | 291.36 / 10.54 | Patch Rank-1 100.0% |
| [SOSNet](examples/sosnet/README.md) | Local patch retrieval | 32×32 | 2.60 | 192 / 192 | 0.75 | 2.59 / 3.12 | 0.28 | 276.90 / 10.25 | Patch Rank-1 100.0% |
| [HardNet](examples/hardnet/README.md) | Local patch retrieval | 32×32 | 2.60 | 192 / 192 | 0.76 | 2.60 / 3.08 | 0.29 | 274.05 / 10.36 | Patch Rank-1 100.0% |
| [Multiscale CNN · voc-tags](examples/voc-tags-branches/README.md) | Multi-label classification | 80×80 | 0.14 | 117 / 117 | 0.71 | 3.34 / 4.00 | 0.10 | 240.87 / 10.11 | Macro AP 27.2% |
| [SqueezeNet ridge probe · voc-tags](examples/voc-tags-squeezeprobe/README.md) | Multi-label classification | 80×80 | 1.55 | 117 / 117 | 0.70 | 4.02 / 4.59 | 0.09 | 207.59 / 10.44 | Macro AP 33.9% |
| [Gabor filter bank + ridge · voc-tags](examples/voc-tags-gabor/README.md) | Multi-label classification | 80×80 | 0.15 | 117 / 117 | 0.70 | 5.32 / 5.93 | 0.10 | 163.53 / 10.10 | Macro AP 27.4% |
| [ResNet-18 ridge probe · voc-tags](examples/voc-tags-resprobe/README.md) | Multi-label classification | 80×80 | 21.47 | 117 / 117 | 0.72 | 8.09 / 9.09 | 0.09 | 112.27 / 10.15 | Macro AP 41.0% |
| [Difference hash](examples/hash-difference/README.md) | Near-duplicate image retrieval | 32×32 | 0.04 | 180 / 180 | 0.41 | 1.92 / 2.45 | 0.39 | 367.63 / 10.42 | Identity Rank-1 96.7% |
| [Low-frequency DCT descriptor](examples/hash-spectrum/README.md) | Near-duplicate image retrieval | 32×32 | 1.03 | 180 / 180 | 0.41 | 2.24 / 2.79 | 0.26 | 343.80 / 10.21 | Identity Rank-1 100.0% |
| [DCT perceptual hash](examples/hash-dct/README.md) | Near-duplicate image retrieval | 32×32 | 1.04 | 180 / 180 | 0.41 | 2.25 / 2.74 | 0.39 | 327.22 / 10.61 | Identity Rank-1 96.7% |
| [Youtu ReID](examples/person_reid_youtu_2021nov/README.md) | Person re-identification | 128×256 | 51.28 | 40 / 40 | 5.99 | 29.01 / 30.46 | 0.24 | 28.38 / 7.65 | Identity Rank-1 85.0% |
| [MobileNetV3 Small ridge probe · Pet](examples/pet-breed-mobile/README.md) | Pet breed classification | 160×160 | 2.62 | 148 / 148 | 2.26 | 9.37 / 10.07 | 0.18 | 84.64 / 9.46 | Top-1 58.8% |
| [ShuffleNetV2 ridge probe · Pet](examples/pet-breed-shuffle/README.md) | Pet breed classification | 160×160 | 6.28 | 148 / 148 | 2.23 | 10.70 / 11.85 | 0.18 | 76.24 / 9.43 | Top-1 62.8% |
| [ResNet-18 ridge probe · Pet](examples/pet-breed-resnet/README.md) | Pet breed classification | 160×160 | 21.51 | 148 / 148 | 2.31 | 13.78 / 14.87 | 0.18 | 61.47 / 9.27 | Top-1 74.3% |
| [CRNN EN](examples/text_recognition_CRNN_EN_2021sep/README.md) | Text recognition | 100×32 | 19.51 | 12 / 30 | 0.52 | 18.50 / 21.36 | 0.40 | 51.48 / 8.79 | Exact match 33.3% |
| [CLIP ViT-B/32 text encoder](examples/clip-text/README.md) | Text-to-image retrieval | 20 tokens | 123.53 | 30 / 30 | 0.82 | 22.59 / 25.22 | 1.03 | 40.91 / 8.49 | Top-1 100.0%<br>Top-5 100.0% |
| [Multiscale CNN · gts](examples/gts-branches/README.md) | Traffic sign classification | 80×80 | 0.15 | 172 / 172 | 0.70 | 3.40 / 3.90 | 0.19 | 233.02 / 10.01 | Top-1 32.0% |
| [SqueezeNet ridge probe · gts](examples/gts-squeezeprobe/README.md) | Traffic sign classification | 80×80 | 1.58 | 172 / 172 | 0.71 | 4.08 / 4.72 | 0.18 | 201.16 / 10.37 | Top-1 39.5% |
| [Gabor filter bank + ridge · gts](examples/gts-gabor/README.md) | Traffic sign classification | 80×80 | 0.18 | 172 / 172 | 0.70 | 5.26 / 5.74 | 0.18 | 162.86 / 10.19 | Top-1 65.7% |
| [ResNet-18 ridge probe · gts](examples/gts-resprobe/README.md) | Traffic sign classification | 80×80 | 21.49 | 172 / 172 | 0.73 | 8.06 / 8.98 | 0.17 | 111.56 / 10.00 | Top-1 42.4% |
| [Convolutional autoencoder · bottle](examples/mvtec-bottle-ae/README.md) | Visual anomaly detection | 128×128 | 0.13 | 44 / 44 | 1.54 | 6.19 / 6.68 | 0.43 | 122.61 / 9.97 | Image AUROC 69.6% |
| [Convolutional autoencoder · grid](examples/mvtec-grid-ae/README.md) | Visual anomaly detection | 128×128 | 0.13 | 60 / 60 | 1.56 | 6.24 / 6.69 | 0.45 | 121.20 / 9.95 | Image AUROC 84.0% |
| [PaDiM (8 features) · grid](examples/mvtec-grid-padim/README.md) | Visual anomaly detection | 128×128 | 1.49 | 60 / 60 | 1.56 | 7.60 / 8.18 | 0.38 | 104.85 / 9.95 | Image AUROC 60.4% |
| [PaDiM (8 features) · bottle](examples/mvtec-bottle-padim/README.md) | Visual anomaly detection | 128×128 | 1.49 | 44 / 44 | 1.55 | 7.77 / 8.33 | 0.39 | 102.96 / 9.93 | Image AUROC 93.1% |
| [Student–teacher CNN · grid](examples/mvtec-grid-student/README.md) | Visual anomaly detection | 128×128 | 1.47 | 60 / 60 | 1.54 | 7.78 / 8.16 | 0.39 | 103.09 / 10.04 | Image AUROC 61.8% |
| [Student–teacher CNN · bottle](examples/mvtec-bottle-student/README.md) | Visual anomaly detection | 128×128 | 1.47 | 44 / 44 | 1.53 | 7.93 / 8.45 | 0.39 | 101.50 / 9.97 | Image AUROC 54.1% |
| [CLIP ViT-B/32 image encoder](examples/clip-image/README.md) | Zero-shot image classification | 224×224 | 178.50 | 100 / 100 | 12.80 | 183.80 / 192.15 | 0.99 | 5.06 / 3.33 | Top-1 100.0%<br>Top-5 100.0% |

</details>

<details>
<summary>Image restoration, image quality and geometry (34)</summary>

| Model | Task | Input | Weights (MiB) | Samples / calls | Pre (ms) | Inference mean / P95 (ms) | Post (ms) | FPS / with GC | Task metrics |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [Bilinear RGGB](examples/bayer-linear/README.md) | Bayer demosaicing | 96×96 → 4×48×48 | 0.04 | 20 / 30 | 0.67 | 3.05 / 3.69 | 1.28 | 199.80 / 10.56 | PSNR dB 26.60 |
| [Residual demosaicing CNN](examples/bayer-residual/README.md) | Bayer demosaicing | 96×96 → 4×48×48 | 0.07 | 20 / 30 | 0.66 | 3.22 / 3.59 | 1.27 | 194.10 / 10.35 | PSNR dB 29.97 |
| [Block DCT energy probe · focus-level](examples/focus-level-dct/README.md) | Defocus severity classification | 64×64 | 0.13 | 96 / 96 | 0.33 | 2.27 / 2.76 | 0.17 | 360.99 / 10.75 | Top-1 85.4% |
| [Three-stage texture CNN · focus-level](examples/focus-level-stackcnn/README.md) | Defocus severity classification | 64×64 | 0.08 | 96 / 96 | 0.34 | 2.33 / 2.76 | 0.18 | 350.92 / 10.56 | Top-1 64.6% |
| [Directional derivative energy probe · focus-level](examples/focus-level-derivative/README.md) | Defocus severity classification | 64×64 | 0.08 | 96 / 96 | 0.33 | 2.61 / 3.16 | 0.17 | 321.33 / 10.74 | Top-1 88.5% |
| [Dense Lucas–Kanade (7×7)](examples/flow-lk/README.md) | Dense optical flow | 160×112 × 2 | 0.08 | 8 / 30 | 1.01 | 13.26 / 14.30 | 0.10 | 69.58 / 9.50 | EPE px 0.77<br>Within 1 px 80.7% |
| [Horn–Schunck (12 iterations)](examples/flow-hs/README.md) | Dense optical flow | 160×112 × 2 | 0.52 | 8 / 30 | 1.03 | 26.59 / 29.41 | 0.11 | 36.06 / 8.34 | EPE px 0.88<br>Within 1 px 75.9% |
| [2-D block matching (49 offsets)](examples/flow-sad/README.md) | Dense optical flow | 160×112 × 2 | 0.09 | 8 / 30 | 0.99 | 29.33 / 31.04 | 22.87 | 18.80 / 6.94 | EPE px 0.66<br>Within 1 px 85.4% |
| [Gray World](examples/wb-gray/README.md) | Illuminant estimation | 96×96 | 0.04 | 48 / 48 | 0.93 | 3.72 / 4.26 | 0.09 | 211.06 / 10.69 | Angular error (°) 3.74 |
| [Shades of Gray (p=4)](examples/wb-shades/README.md) | Illuminant estimation | 96×96 | 0.06 | 48 / 48 | 0.88 | 3.88 / 4.27 | 0.09 | 206.22 / 10.67 | Angular error (°) 2.85 |
| [First-order Gray Edge](examples/wb-edge/README.md) | Illuminant estimation | 96×96 | 0.07 | 48 / 48 | 0.90 | 4.36 / 4.81 | 0.09 | 186.91 / 10.37 | Angular error (°) 2.95 |
| [FFDNet Color σ25](examples/ffdnet25/README.md) | Image denoising | 128×128 | 1.97 | 10 / 30 | 1.52 | 15.55 / 16.39 | 1.89 | 52.75 / 9.03 | PSNR dB 28.44 |
| [DnCNN σ25](examples/dncnn25/README.md) | Image denoising | 128×128 | 1.18 | 10 / 30 | 0.47 | 26.80 / 28.01 | 0.64 | 35.83 / 8.25 | PSNR dB 26.43 |
| [LaMa](examples/inpainting_lama_2025jan/README.md) | Inpainting | 512×512 + mask | 126.24 | 8 / 8 | 23.73 | 9277.85 / 9321.12 | 9.18 | 0.11 / 0.11 | Masked PSNR (dB) 14.41 |
| [Dilated residual CNN](examples/jpeg-dilated/README.md) | JPEG artifact removal | 96×96 | 0.08 | 20 / 30 | 0.89 | 4.94 / 5.54 | 1.27 | 141.01 / 10.16 | PSNR dB 26.03 |
| [AR-CNN](examples/jpeg-arcnn/README.md) | JPEG artifact removal | 96×96 | 0.27 | 20 / 30 | 0.91 | 9.51 / 10.06 | 1.28 | 85.45 / 9.63 | PSNR dB 20.73 |
| [Block DCT energy probe · jpeg-level](examples/jpeg-level-dct/README.md) | JPEG quality classification | 64×64 | 0.13 | 96 / 96 | 0.33 | 2.20 / 2.62 | 0.17 | 370.88 / 10.90 | Top-1 80.2% |
| [Three-stage texture CNN · jpeg-level](examples/jpeg-level-stackcnn/README.md) | JPEG quality classification | 64×64 | 0.08 | 96 / 96 | 0.33 | 2.30 / 2.85 | 0.18 | 356.76 / 10.55 | Top-1 29.2% |
| [Directional derivative energy probe · jpeg-level](examples/jpeg-level-derivative/README.md) | JPEG quality classification | 64×64 | 0.08 | 96 / 96 | 0.33 | 2.64 / 3.14 | 0.17 | 318.03 / 10.55 | Top-1 59.4% |
| [Zero-DCE](examples/zero-dce/README.md) | Low-light enhancement | 192×192 | 0.32 | 10 / 30 | 3.42 | 40.63 / 45.12 | 4.47 | 20.61 / 6.92 | PSNR dB 25.21 |
| [MiDaS v2.1 Small](examples/depth_midas_v21_small_256_fp16/README.md) | Monocular relative depth | 256×256 | 32.17 | 1 / 30 | 3.93 | 53.61 / 55.35 | 0.65 | 17.18 / 6.30 | Disparity Pearson r 0.881 |
| [Block DCT energy probe · blur-angle](examples/blur-angle-dct/README.md) | Motion blur direction | 64×64 | 0.13 | 96 / 96 | 0.32 | 2.22 / 2.53 | 0.17 | 369.39 / 10.88 | Top-1 75.0% |
| [Three-stage texture CNN · blur-angle](examples/blur-angle-stackcnn/README.md) | Motion blur direction | 64×64 | 0.08 | 96 / 96 | 0.33 | 2.37 / 2.94 | 0.18 | 346.21 / 10.48 | Top-1 78.1% |
| [Directional derivative energy probe · blur-angle](examples/blur-angle-derivative/README.md) | Motion blur direction | 64×64 | 0.08 | 96 / 96 | 0.32 | 2.58 / 2.99 | 0.17 | 326.48 / 10.69 | Top-1 97.9% |
| [NAFNet](examples/deblurring_nafnet_2025may/README.md) | Motion deblurring | 384×384 | 149.34 | 8 / 8 | 18.16 | 1795.73 / 1810.85 | 19.71 | 0.55 / 0.52 | PSNR dB 25.96 |
| [Block DCT energy probe · noise-level](examples/noise-level-dct/README.md) | Noise level classification | 64×64 | 0.13 | 96 / 96 | 0.33 | 2.24 / 2.74 | 0.17 | 366.09 / 10.85 | Top-1 84.4% |
| [Three-stage texture CNN · noise-level](examples/noise-level-stackcnn/README.md) | Noise level classification | 64×64 | 0.08 | 96 / 96 | 0.33 | 2.32 / 2.91 | 0.17 | 354.42 / 10.45 | Top-1 76.0% |
| [Directional derivative energy probe · noise-level](examples/noise-level-derivative/README.md) | Noise level classification | 64×64 | 0.08 | 96 / 96 | 0.33 | 2.62 / 3.15 | 0.17 | 320.79 / 10.63 | Top-1 78.1% |
| [SAD block matching](examples/stereo-sad/README.md) | Stereo disparity | 160×96 × 2 | 0.07 | 18 / 30 | 0.95 | 18.81 / 20.24 | 13.47 | 30.09 / 7.89 | Disparity MAE (px) 1.37<br>Within 1 px 78.8% |
| [Gradient block matching](examples/stereo-gradient/README.md) | Stereo disparity | 160×96 × 2 | 0.07 | 18 / 30 | 0.97 | 19.03 / 20.16 | 13.07 | 30.24 / 7.99 | Disparity MAE (px) 1.31<br>Within 1 px 83.9% |
| [Zero-mean SAD](examples/stereo-zsad/README.md) | Stereo disparity | 160×96 × 2 | 0.08 | 18 / 30 | 0.99 | 20.54 / 21.72 | 13.18 | 28.81 / 7.93 | Disparity MAE (px) 1.22<br>Within 1 px 83.8% |
| [Normalized cross-correlation](examples/stereo-ncc/README.md) | Stereo disparity | 160×96 × 2 | 0.14 | 18 / 30 | 0.93 | 36.06 / 42.88 | 14.48 | 19.43 / 6.93 | Disparity MAE (px) 1.46<br>Within 1 px 79.6% |
| [Mosaic TransformerNet](examples/mosaic/README.md) | Style transfer | 224×224 | 4.94 | 10 / 30 | 3.78 | 118.77 / 125.46 | 8.82 | 7.61 / 4.39 | MAE vs. CPU (0–255) 7.96 |
| [ESPCN ×3](examples/espcn-x3/README.md) | Super-resolution | 224×224 | 0.18 | 8 / 30 | 1.26 | 31.37 / 32.27 | 2.11 | 28.78 / 7.68 | Y-PSNR dB 28.04 |

</details>

<details>
<summary>Audio, speech and text (23)</summary>

| Model | Task | Input | Weights (MiB) | Samples / calls | Pre (ms) | Inference mean / P95 (ms) | Post (ms) | FPS / with GC | Task metrics |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [Quadrature tone bank](examples/dtmf-filter/README.md) | DTMF recognition | 1024 pts (128 ms) | 0.15 | 96 / 96 | 0.05 | 1.97 / 2.35 | 0.17 | 457.02 / 10.99 | Top-1 100.0% |
| [Spectral-bin MLP](examples/dtmf-mlp/README.md) | DTMF recognition | 1024 pts (128 ms) | 0.14 | 96 / 96 | 0.05 | 1.99 / 2.32 | 0.18 | 452.22 / 10.86 | Top-1 100.0% |
| [Normalized autocorrelation](examples/pitch-acf/README.md) | Pitch estimation | 1024 pts (128 ms) | 0.33 | 96 / 96 | 0.05 | 3.28 / 3.87 | 0.41 | 267.70 / 10.56 | Pitch MAE (cents) 247.47<br>Within 50 cents 81.2% |
| [YIN difference function](examples/pitch-yin/README.md) | Pitch estimation | 1024 pts (128 ms) | 0.32 | 96 / 96 | 0.05 | 3.30 / 3.89 | 0.43 | 264.58 / 10.62 | Pitch MAE (cents) 247.84<br>Within 50 cents 81.2% |
| [Averaged linear perceptron · trec](examples/trec-perceptron/README.md) | Question classification | 512 features | 0.04 | 157 / 157 | 0.50 | 1.78 / 2.27 | 0.17 | 409.22 / 10.53 | Top-1 77.1% |
| [N-gram ridge classifier · trec](examples/trec-ridge/README.md) | Question classification | 512 features | 0.04 | 157 / 157 | 0.50 | 1.78 / 2.21 | 0.17 | 407.51 / 10.75 | Top-1 73.2% |
| [Bernoulli Naive Bayes · trec](examples/trec-bnb/README.md) | Question classification | 512 features | 0.05 | 157 / 157 | 0.50 | 1.83 / 2.37 | 0.17 | 400.90 / 10.79 | Top-1 67.5% |
| [Bernoulli Naive Bayes · sent](examples/sent-bnb/README.md) | Review sentiment | 512 features | 0.04 | 60 / 60 | 0.72 | 1.77 / 2.08 | 0.17 | 375.97 / 11.16 | Top-1 71.7% |
| [Averaged linear perceptron · sent](examples/sent-perceptron/README.md) | Review sentiment | 512 features | 0.04 | 60 / 60 | 0.72 | 1.78 / 2.33 | 0.17 | 375.82 / 10.80 | Top-1 73.3% |
| [N-gram ridge classifier · sent](examples/sent-ridge/README.md) | Review sentiment | 512 features | 0.04 | 60 / 60 | 0.73 | 1.85 / 2.48 | 0.17 | 364.50 / 10.97 | Top-1 73.3% |
| [Hashed n-gram MLP](examples/sms-mlp/README.md) | SMS spam detection | 512 features | 0.08 | 128 / 128 | 1.05 | 1.79 / 2.16 | 0.10 | 340.27 / 10.21 | Accuracy 93.0%<br>Spam F1 92.4% |
| [Linear SVM](examples/sms-svm/README.md) | SMS spam detection | 512 features | 0.04 | 128 / 128 | 1.05 | 1.79 / 2.23 | 0.10 | 340.03 / 10.28 | Accuracy 93.0%<br>Spam F1 92.6% |
| [Multinomial Naive Bayes](examples/sms-nb/README.md) | SMS spam detection | 512 features | 0.04 | 128 / 128 | 1.06 | 1.82 / 2.33 | 0.10 | 335.34 / 10.36 | Accuracy 89.8%<br>Spam F1 89.1% |
| [YAMNet](examples/yamnet_3s/README.md) | Sound event classification | 48000 (3 s) | 9.30 | 16 / 30 | 0.25 | 23.93 / 27.20 | 0.29 | 40.87 / 8.16 | Top-1 18.8% |
| [Residual speaker CNN](examples/speaker-rescnn/README.md) | Speaker retrieval | 1 s → 40×98 | 0.12 | 96 / 96 | 16.03 | 2.30 / 2.78 | 0.25 | 53.82 / 8.96 | Identity Rank-1 83.3% |
| [Compact x-vector TDNN](examples/speaker-tdnn/README.md) | Speaker retrieval | 1 s → 40×98 | 0.14 | 96 / 96 | 15.84 | 2.42 / 2.97 | 0.24 | 54.05 / 9.07 | Identity Rank-1 80.6% |
| [TC-ResNet](examples/tc-kws/README.md) | Speech commands | 1 s → 40×98 | 0.12 | 96 / 96 | 15.75 | 2.35 / 2.73 | 0.18 | 54.70 / 9.09 | Top-1 80.2% |
| [DS-CNN](examples/ds-kws/README.md) | Speech commands | 1 s → 40×98 | 0.11 | 96 / 96 | 15.86 | 2.66 / 3.32 | 0.16 | 53.52 / 9.11 | Top-1 86.5% |
| [Log-Mel shrinkage LDA](examples/presence-lda/README.md) | Speech presence | 1 s → 40×98 | 0.05 | 128 / 128 | 15.90 | 2.31 / 2.94 | 0.17 | 54.39 / 8.99 | Top-1 81.2% |
| [Spectrogram CNN](examples/presence-cnn/README.md) | Speech presence | 1 s → 40×98 | 0.12 | 128 / 128 | 15.89 | 2.43 / 2.91 | 0.19 | 53.99 / 8.85 | Top-1 83.6% |
| [NeMo Citrinet-512](examples/citrinet-6s/README.md) | Speech recognition | 80×600 (6 s) | 73.57 | 9 / 9 | 148.71 | 86.50 / 92.22 | 0.99 | 4.23 / 2.98 | WER 15.0% |
| [NeMo Conformer Small CTC](examples/nemo-ctc-6s/README.md) | Speech recognition | 80×600 (6 s) | 35.04 | 9 / 9 | 152.05 | 166.20 / 173.71 | 1.11 | 3.13 / 2.39 | WER 11.2% |
| [Wav2Vec2 Base 960h · 6s](examples/wav2vec2-6s/README.md) | Speech recognition | 96000 (6 s) | 186.93 | 9 / 9 | 2.51 | 502.38 / 514.21 | 1.03 | 1.98 / 1.64 | WER 4.7% |

</details>

<details>
<summary>Sensors, time series, spectra and tabular data (71)</summary>

| Model | Task | Input | Weights (MiB) | Samples / calls | Pre (ms) | Inference mean / P95 (ms) | Post (ms) | FPS / with GC | Task metrics |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [Cosine basis ridge · Plane](examples/plane-cosine/README.md) | Aircraft outlines | 1×144 | 0.07 | 60 / 60 | 0.18 | 1.90 / 2.40 | 0.16 | 444.62 / 10.51 | Top-1 98.3% |
| [Random convolution max/mean ridge · Plane](examples/plane-randomconv/README.md) | Aircraft outlines | 1×144 | 0.10 | 60 / 60 | 0.19 | 1.91 / 2.44 | 0.16 | 442.89 / 10.52 | Top-1 100.0% |
| [Subsequence distance transform · Plane](examples/plane-shapelets/README.md) | Aircraft outlines | 1×144 | 0.17 | 60 / 60 | 0.20 | 2.55 / 3.23 | 0.16 | 343.25 / 10.46 | Top-1 100.0% |
| [PLS regression · airfoil](examples/airfoil-pls/README.md) | Airfoil noise regression | 5 features | 0.03 | 96 / 96 | 0.14 | 1.70 / 2.07 | 0.09 | 520.70 / 10.84 | MAE 4.11 |
| [Extreme Learning Machine · airfoil](examples/airfoil-elm/README.md) | Airfoil noise regression | 5 features | 0.05 | 96 / 96 | 0.14 | 1.83 / 2.50 | 0.09 | 485.90 / 10.58 | MAE 3.15 |
| [Random Fourier ridge · airfoil](examples/airfoil-rff/README.md) | Airfoil noise regression | 5 features | 0.04 | 96 / 96 | 0.13 | 1.85 / 2.62 | 0.09 | 482.53 / 10.62 | MAE 2.98 |
| [Multiscale Inception CNN · ElectricDevices](examples/sig-electricdevices/README.md) | Appliance identification | 1 ch × 96 pts | 0.12 | 60 / 60 | 0.18 | 1.92 / 2.24 | 0.16 | 443.44 / 10.51 | Top-1 55.0% |
| [Cosine basis ridge · ArrowHead](examples/arrowhead-cosine/README.md) | Arrowhead outlines | 1×251 | 0.09 | 60 / 60 | 0.19 | 1.85 / 2.19 | 0.18 | 451.71 / 10.88 | Top-1 65.0% |
| [Random convolution max/mean ridge · ArrowHead](examples/arrowhead-randomconv/README.md) | Arrowhead outlines | 1×251 | 0.10 | 60 / 60 | 0.20 | 2.02 / 2.70 | 0.17 | 418.04 / 10.61 | Top-1 81.7% |
| [Subsequence distance transform · ArrowHead](examples/arrowhead-shapelets/README.md) | Arrowhead outlines | 1×251 | 0.13 | 60 / 60 | 0.20 | 2.42 / 2.89 | 0.17 | 358.33 / 10.93 | Top-1 78.3% |
| [Cosine basis ridge · Beef](examples/beef-cosine/README.md) | Beef spectra | 1×470 | 0.13 | 30 / 30 | 0.19 | 1.90 / 2.29 | 0.17 | 441.55 / 10.72 | Top-1 83.3% |
| [Random convolution max/mean ridge · Beef](examples/beef-randomconv/README.md) | Beef spectra | 1×470 | 0.12 | 30 / 30 | 0.20 | 2.08 / 2.61 | 0.17 | 409.27 / 10.69 | Top-1 83.3% |
| [Subsequence distance transform · Beef](examples/beef-shapelets/README.md) | Beef spectra | 1×470 | 0.15 | 30 / 30 | 0.20 | 3.08 / 3.67 | 0.19 | 287.17 / 10.55 | Top-1 73.3% |
| [PLS regression · energy](examples/energy-pls/README.md) | Building heating load | 8 features | 0.03 | 96 / 96 | 0.14 | 1.68 / 2.04 | 0.09 | 523.69 / 10.19 | MAE 1.93 |
| [Random Fourier ridge · energy](examples/energy-rff/README.md) | Building heating load | 8 features | 0.04 | 96 / 96 | 0.14 | 1.82 / 2.21 | 0.09 | 488.66 / 10.07 | MAE 1.44 |
| [Extreme Learning Machine · energy](examples/energy-elm/README.md) | Building heating load | 8 features | 0.05 | 96 / 96 | 0.15 | 1.86 / 2.35 | 0.09 | 476.85 / 10.13 | MAE 1.44 |
| [Cosine basis ridge · Car](examples/car-cosine/README.md) | Car sensor sequences | 1×577 | 0.15 | 60 / 60 | 0.19 | 1.87 / 2.28 | 0.18 | 445.93 / 10.68 | Top-1 63.3% |
| [Random convolution max/mean ridge · Car](examples/car-randomconv/README.md) | Car sensor sequences | 1×577 | 0.12 | 60 / 60 | 0.20 | 2.11 / 2.64 | 0.17 | 403.20 / 10.53 | Top-1 76.7% |
| [Subsequence distance transform · Car](examples/car-shapelets/README.md) | Car sensor sequences | 1×577 | 0.13 | 60 / 60 | 0.21 | 2.90 / 3.56 | 0.19 | 303.56 / 10.55 | Top-1 73.3% |
| [Shrinkage LDA · Coffee](examples/sig-coffee/README.md) | Coffee spectra | 1 ch × 286 pts | 0.04 | 28 / 30 | 0.19 | 1.82 / 2.36 | 0.16 | 460.31 / 10.07 | Top-1 100.0% |
| [Two-layer MLP · ItalyPowerDemand](examples/sig-italypowerdemand/README.md) | Electricity demand by season | 1 ch × 24 pts | 0.04 | 60 / 60 | 0.21 | 1.89 / 2.35 | 0.15 | 443.96 / 9.76 | Top-1 96.7% |
| [Temporal FCN · FordA](examples/sig-forda/README.md) | Engine fault classification | 1 ch × 500 pts | 0.07 | 60 / 60 | 0.20 | 2.00 / 2.28 | 0.19 | 418.66 / 10.52 | Top-1 88.3% |
| [Two-layer MLP · FaceAll](examples/sig-faceall/README.md) | Face outline sequences | 1 ch × 131 pts | 0.06 | 60 / 60 | 0.18 | 1.84 / 2.25 | 0.16 | 459.73 / 9.84 | Top-1 83.3% |
| [Cosine basis ridge · Fish](examples/fish-cosine/README.md) | Fish outlines | 1×463 | 0.13 | 60 / 60 | 0.20 | 1.88 / 2.25 | 0.16 | 446.88 / 10.64 | Top-1 70.0% |
| [Random convolution max/mean ridge · Fish](examples/fish-randomconv/README.md) | Fish outlines | 1×463 | 0.12 | 60 / 60 | 0.19 | 2.04 / 2.48 | 0.16 | 416.67 / 10.70 | Top-1 95.0% |
| [Subsequence distance transform · Fish](examples/fish-shapelets/README.md) | Fish outlines | 1×463 | 0.17 | 60 / 60 | 0.19 | 3.13 / 3.61 | 0.18 | 285.51 / 10.58 | Top-1 93.3% |
| [Ridge classifier · Strawberry](examples/sig-strawberry/README.md) | Fruit puree spectra | 1 ch × 235 pts | 0.04 | 60 / 60 | 0.19 | 1.88 / 2.44 | 0.16 | 449.25 / 10.03 | Top-1 88.3% |
| [Cosine basis ridge · Ham](examples/ham-cosine/README.md) | Ham protein spectra | 1×431 | 0.12 | 60 / 60 | 0.21 | 1.85 / 2.19 | 0.17 | 450.74 / 10.58 | Top-1 63.3% |
| [Random convolution max/mean ridge · Ham](examples/ham-randomconv/README.md) | Ham protein spectra | 1×431 | 0.12 | 60 / 60 | 0.20 | 2.01 / 2.41 | 0.16 | 421.45 / 10.43 | Top-1 71.7% |
| [Subsequence distance transform · Ham](examples/ham-shapelets/README.md) | Ham protein spectra | 1×431 | 0.12 | 60 / 60 | 0.19 | 2.45 / 3.04 | 0.17 | 355.59 / 10.47 | Top-1 70.0% |
| [Cosine basis ridge · GunPoint](examples/gunpoint-cosine/README.md) | Hand motion trajectories | 1×150 | 0.07 | 60 / 60 | 0.18 | 1.83 / 2.20 | 0.18 | 456.16 / 10.87 | Top-1 90.0% |
| [Random convolution max/mean ridge · GunPoint](examples/gunpoint-randomconv/README.md) | Hand motion trajectories | 1×150 | 0.10 | 60 / 60 | 0.19 | 1.93 / 2.27 | 0.16 | 437.70 / 10.74 | Top-1 100.0% |
| [Subsequence distance transform · GunPoint](examples/gunpoint-shapelets/README.md) | Hand motion trajectories | 1×150 | 0.12 | 60 / 60 | 0.21 | 2.32 / 2.76 | 0.17 | 369.88 / 10.81 | Top-1 96.7% |
| [Multiscale Inception CNN · Handwriting](examples/sig-handwriting/README.md) | Handwriting trajectories | 3 ch × 152 pts | 0.12 | 60 / 60 | 0.22 | 2.06 / 2.54 | 0.17 | 408.00 / 10.33 | Top-1 6.7% |
| [Residual temporal CNN · ECG5000](examples/sig-ecg5000/README.md) | Heartbeat classification | 1 ch × 140 pts | 0.09 | 60 / 60 | 0.19 | 1.93 / 2.55 | 0.18 | 435.87 / 10.49 | Top-1 45.0% |
| [Temporal FCN · InsectWingbeatSound](examples/sig-insectwingbeatsound/README.md) | Insect wingbeats | 1 ch × 256 pts | 0.07 | 60 / 60 | 0.18 | 1.90 / 2.32 | 0.17 | 443.90 / 9.91 | Top-1 18.3% |
| [Random convolution max/mean ridge · SwedishLeaf](examples/swedishleaf-randomconv/README.md) | Leaf outline classification | 1×128 | 0.10 | 60 / 60 | 0.19 | 1.93 / 2.39 | 0.17 | 438.02 / 10.91 | Top-1 95.0% |
| [Cosine basis ridge · SwedishLeaf](examples/swedishleaf-cosine/README.md) | Leaf outline classification | 1×128 | 0.07 | 60 / 60 | 0.19 | 1.94 / 2.71 | 0.19 | 430.42 / 10.75 | Top-1 73.3% |
| [Subsequence distance transform · SwedishLeaf](examples/swedishleaf-shapelets/README.md) | Leaf outline classification | 1×128 | 0.27 | 60 / 60 | 0.19 | 2.61 / 3.18 | 0.17 | 336.76 / 10.75 | Top-1 90.0% |
| [RBF prototype network · Meat](examples/sig-meat/README.md) | Meat spectra | 1 ch × 448 pts | 0.06 | 60 / 60 | 0.20 | 2.00 / 2.39 | 0.16 | 423.24 / 9.98 | Top-1 93.3% |
| [Cosine basis ridge · OliveOil](examples/oliveoil-cosine/README.md) | Olive oil origin | 1×570 | 0.15 | 30 / 30 | 0.20 | 1.90 / 2.37 | 0.18 | 438.91 / 10.82 | Top-1 96.7% |
| [Random convolution max/mean ridge · OliveOil](examples/oliveoil-randomconv/README.md) | Olive oil origin | 1×570 | 0.12 | 30 / 30 | 0.19 | 2.04 / 2.48 | 0.18 | 413.63 / 10.74 | Top-1 76.7% |
| [Subsequence distance transform · OliveOil](examples/oliveoil-shapelets/README.md) | Olive oil origin | 1×570 | 0.13 | 30 / 30 | 0.21 | 2.97 / 3.41 | 0.18 | 296.81 / 10.78 | Top-1 43.3% |
| [Diagonal QDA · gridstab](examples/gridstab-qda/README.md) | Power grid stability | 12 features | 0.03 | 48 / 48 | 0.16 | 1.72 / 2.05 | 0.17 | 488.61 / 10.72 | Top-1 81.2% |
| [Random ReLU features + ridge · gridstab](examples/gridstab-randomnet/README.md) | Power grid stability | 12 features | 0.04 | 48 / 48 | 0.14 | 1.72 / 2.16 | 0.16 | 495.04 / 10.69 | Top-1 79.2% |
| [Nearest subclass prototype · gridstab](examples/gridstab-lvq/README.md) | Power grid stability | 12 features | 0.03 | 48 / 48 | 0.13 | 1.81 / 2.30 | 0.16 | 475.11 / 10.74 | Top-1 77.1% |
| [Random ReLU features + ridge · raisins](examples/raisins-randomnet/README.md) | Raisin variety | 7 features | 0.04 | 48 / 48 | 0.14 | 1.68 / 1.99 | 0.17 | 503.03 / 10.74 | Top-1 79.2% |
| [Diagonal QDA · raisins](examples/raisins-qda/README.md) | Raisin variety | 7 features | 0.03 | 48 / 48 | 0.14 | 1.71 / 2.07 | 0.16 | 499.06 / 11.07 | Top-1 75.0% |
| [Nearest subclass prototype · raisins](examples/raisins-lvq/README.md) | Raisin variety | 7 features | 0.03 | 48 / 48 | 0.15 | 1.88 / 2.38 | 0.18 | 451.43 / 10.74 | Top-1 81.2% |
| [Cosine basis ridge · RefrigerationDevices](examples/refrigerationdevices-cosine/README.md) | Refrigeration load | 1×720 | 0.18 | 60 / 60 | 0.20 | 1.90 / 2.18 | 0.17 | 442.17 / 10.74 | Top-1 35.0% |
| [Random convolution max/mean ridge · RefrigerationDevices](examples/refrigerationdevices-randomconv/README.md) | Refrigeration load | 1×720 | 0.12 | 60 / 60 | 0.20 | 2.13 / 2.46 | 0.17 | 400.03 / 10.68 | Top-1 45.0% |
| [Subsequence distance transform · RefrigerationDevices](examples/refrigerationdevices-shapelets/README.md) | Refrigeration load | 1×720 | 0.13 | 60 / 60 | 0.21 | 2.90 / 3.39 | 0.17 | 305.07 / 10.72 | Top-1 53.3% |
| [Cosine basis ridge · ScreenType](examples/screentype-cosine/README.md) | Screen type | 1×720 | 0.18 | 60 / 60 | 0.20 | 1.90 / 2.16 | 0.19 | 437.71 / 10.99 | Top-1 31.7% |
| [Random convolution max/mean ridge · ScreenType](examples/screentype-randomconv/README.md) | Screen type | 1×720 | 0.12 | 60 / 60 | 0.20 | 2.10 / 2.44 | 0.17 | 405.66 / 10.87 | Top-1 30.0% |
| [Subsequence distance transform · ScreenType](examples/screentype-shapelets/README.md) | Screen type | 1×720 | 0.13 | 60 / 60 | 0.20 | 2.95 / 3.38 | 0.18 | 300.37 / 10.87 | Top-1 33.3% |
| [Depthwise temporal CNN · Earthquakes](examples/sig-earthquakes/README.md) | Seismic signals | 1 ch × 512 pts | 0.07 | 60 / 60 | 0.19 | 2.04 / 2.51 | 0.19 | 413.29 / 9.97 | Top-1 50.0% |
| [Residual temporal CNN · Epilepsy](examples/sig-epilepsy/README.md) | Simulated gesture recognition | 3 ch × 206 pts | 0.09 | 60 / 60 | 0.22 | 1.93 / 2.22 | 0.16 | 433.47 / 10.42 | Top-1 96.7% |
| [Cosine basis ridge · Trace](examples/trace-cosine/README.md) | Simulated process transients | 1×275 | 0.10 | 60 / 60 | 0.20 | 1.90 / 2.33 | 0.19 | 437.30 / 10.85 | Top-1 93.3% |
| [Random convolution max/mean ridge · Trace](examples/trace-randomconv/README.md) | Simulated process transients | 1×275 | 0.10 | 60 / 60 | 0.21 | 2.07 / 2.67 | 0.17 | 408.09 / 10.52 | Top-1 100.0% |
| [Subsequence distance transform · Trace](examples/trace-shapelets/README.md) | Simulated process transients | 1×275 | 0.13 | 60 / 60 | 0.19 | 2.52 / 3.03 | 0.18 | 346.24 / 10.80 | Top-1 100.0% |
| [Dilated temporal CNN · Wafer](examples/sig-wafer/README.md) | Wafer process anomalies | 1 ch × 152 pts | 0.07 | 60 / 60 | 0.19 | 1.94 / 2.32 | 0.16 | 436.42 / 10.28 | Top-1 100.0% |
| [Random ReLU features + ridge · seeds](examples/seeds-randomnet/README.md) | Wheat variety | 7 features | 0.04 | 59 / 59 | 0.13 | 1.70 / 2.05 | 0.16 | 501.04 / 11.09 | Top-1 93.2% |
| [Diagonal QDA · seeds](examples/seeds-qda/README.md) | Wheat variety | 7 features | 0.03 | 59 / 59 | 0.14 | 1.72 / 1.99 | 0.17 | 492.34 / 11.25 | Top-1 93.2% |
| [Nearest subclass prototype · seeds](examples/seeds-lvq/README.md) | Wheat variety | 7 features | 0.03 | 59 / 59 | 0.13 | 1.77 / 2.12 | 0.17 | 481.76 / 11.06 | Top-1 93.2% |
| [Random ReLU features + ridge · wifi](examples/wifi-randomnet/README.md) | Wi-Fi room localization | 7 features | 0.04 | 96 / 96 | 0.15 | 1.72 / 2.13 | 0.17 | 491.69 / 10.80 | Top-1 99.0% |
| [Diagonal QDA · wifi](examples/wifi-qda/README.md) | Wi-Fi room localization | 7 features | 0.03 | 96 / 96 | 0.14 | 1.72 / 2.13 | 0.17 | 492.16 / 11.20 | Top-1 99.0% |
| [Nearest subclass prototype · wifi](examples/wifi-lvq/README.md) | Wi-Fi room localization | 7 features | 0.03 | 96 / 96 | 0.14 | 1.81 / 2.21 | 0.17 | 473.71 / 10.99 | Top-1 96.9% |
| [PLS regression · wine](examples/wine-pls/README.md) | Wine quality regression | 11 features | 0.03 | 96 / 96 | 0.14 | 1.71 / 2.22 | 0.09 | 517.39 / 10.48 | MAE 0.60 |
| [Extreme Learning Machine · wine](examples/wine-elm/README.md) | Wine quality regression | 11 features | 0.06 | 96 / 96 | 0.14 | 1.82 / 2.40 | 0.09 | 487.68 / 10.38 | MAE 0.59 |
| [Random Fourier ridge · wine](examples/wine-rff/README.md) | Wine quality regression | 11 features | 0.04 | 96 / 96 | 0.14 | 1.86 / 2.50 | 0.09 | 478.77 / 10.24 | MAE 0.62 |
| [Depthwise temporal CNN · BasicMotions](examples/sig-basicmotions/README.md) | Wrist activity | 6 ch × 100 pts | 0.06 | 40 / 40 | 0.22 | 1.89 / 2.18 | 0.18 | 435.64 / 10.43 | Top-1 80.0% |

</details>

### VLM scene description latency

The preview app measured these two VLMs separately. Times below are medians. [VLM timing records](benchmark_results/app/vlm-timing.json)

| Model | Captions | App latency |
|---|---:|---|
| SmolVLM-256M-Instruct | 12 | Vision encoder 1.589 s, language decoder 1.683 s |
| InternVL3-1B | 5 | First text 12.769 s, complete caption 16.434 s |

---

## Quick Start

```bash
# Clone the repo (with Git LFS for model weights)
git lfs install
git clone https://github.com/xr686/reCamera-pro-benchmark.git
cd reCamera-pro-benchmark

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
