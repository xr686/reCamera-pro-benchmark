# Native C++ Benchmark Harness (RV1126B)

`bench_native` is a single aarch64 binary that benchmarks every YOLO model in this repo on
the reCamera Pro. It replaces the previous Python `rknn-toolkit-lite2` inference scripts and
uses the RKNN Runtime C API directly.

## Build

```bash
export RECAMERA_SDK_ROOT=/path/to/recamera-pro-sdk-v1.0.10   # optional, has a default
./build.sh          # runs fetch_libs.sh on first use, then cross-compiles
```

`fetch_libs.sh` fills `libs/` (gitignored) with:

| Artifact | Source |
|----------|--------|
| `librknnrt.so` | `$RECAMERA_RKNNRT` → `recamera-rknn-dev` skill asset → the device |
| `rknn_api.h` | `$RECAMERA_SDK_ROOT/project/app/recamera_ipc/common/rc_infer/include` |
| RGA headers | `$RECAMERA_SDK_ROOT/media/rga/rga/{include,im2d_api}` |
| `librga.so` | `$RECAMERA_RGA` → the device (`/oem/usr/lib/librga.so`) |

`librknnrt.so` must match the Seeed-qualified RKNN 2.3.2 runtime
(SHA-256 `d31fc19c85b8…`); the script aborts on a mismatch. The binary links the vendor
libraries by absolute path and carries `RPATH=/oem/usr/lib`, so no host path leaks into the
runtime search path.

## Run

```bash
./bench_native --model models/yolov8n_640x640_W8A8.rknn --image bench.rgb
./bench_native --model models/x.rknn --dump-io          # inspect tensor attributes
```

| Option | Meaning |
|--------|---------|
| `--model <file.rknn>` | Model to benchmark (required) |
| `--image <raw.rgb>` | Raw RGB888 input frame |
| `--src WxH` | Geometry of that frame (default `1280x720`) |
| `--runs N` / `--warmup N` | Timed / discarded iterations (default 100 / 20) |
| `--conf F` / `--nms F` | Detection thresholds (default 0.25 / 0.45) |
| `--task T` | Force `det`/`pose`/`seg`/`obb`/`cls`/`depth` (default: auto-detect) |
| `--dump-io` | Print input/output tensor attributes and exit |
| `--trace-post` | Per-stage post-processing timing (depth) |

From the repo root, `scripts/bench_native.sh` builds, stages the binary/frame/models and
runs the whole matrix, writing `benchmark_results/native/{*.log,results.md,results.csv}`.

## What makes it fast

1. **Pinned clocks** — before the first frame the harness sets the NPU devfreq governor to
   `userspace` at its top OPP (950 MHz) and the CPU/VENC/VPU governors to `performance`
   (1608 MHz). Leaving the NPU on `dvfs` alone costs 25–35 %.
2. **Zero-copy input** — one `rknn_create_mem` buffer is handed to `rknn_set_io_mem` and RGA
   letterboxes the source frame straight into it. The buffer is declared `UINT8`/`NHWC` with
   `pass_through = 0` so the runtime maps it onto the model's own int8 (or fp16) input;
   declaring it as the native int8 tensor instead would reinterpret every pixel below 128 and
   roughly double NPU time.
3. **Zero-copy int8 outputs** — outputs are bound as dense NCHW int8 (`n_elems` bytes, not
   `size_with_stride`), so the runtime performs the NHWC→NCHW reorder and no float32 copy of
   the head tensors ever exists. The Python runtime allocates a fresh float32 NumPy array per
   output per call (~5 ms at 640×640).
4. **int8-domain filtering** — quantisation scales are positive, so ordering is preserved:
   the per-cell class maximum is computed on raw int8 with a branchless sweep, and only the
   cells above the confidence threshold are dequantised (argmax, sigmoid, DFL softmax,
   dist2rbox). Post-processing costs 0.1–0.7 ms instead of several milliseconds.
5. **Cache-friendly heavy post** — segmentation assembles `sigmoid(coef · proto)` per
   detection with the channel loop outermost (the int8 planes are walked sequentially, and
   `sigmoid(x) > 0.5` reduces to `x > 0`); depth resolves `exp()` through a 256-entry int8
   lookup table and lets RGA do the upscale.
6. **Serial, honest timing** — preprocess, `rknn_run` and postprocess are timed separately
   and summed, matching the "end-to-end latency" definition used by the tables in the README.

## Head auto-detection

The decoder needs no per-model configuration. Outputs are grouped by spatial size into
strides, then classified by tensor name (`dfl`/`bbox`, `cls`, `conf`/`objectness`,
`keypoint`, `mask_coeff`, `angle`) with a channel-count fallback for anonymous names
(`318`, `onnx::ReduceSum_476`, `/model.22/Concat_1_output_0`). Concatenated tensors are
split by offset. Layouts found in this repo:

| Model | Outputs per stride | Notes |
|-------|--------------------|-------|
| YOLO26n det | `bbox[4]`, `cls[80]` (sigmoid) | direct box regression, no DFL |
| YOLO11n/s det | `dfl[64]`, `cls[80]` (sigmoid), `conf[1]` | score = `conf × max cls` |
| YOLOv8n/s det | `dfl[64]`, `cls[80]` (sigmoid), `conf[1]` | as above |
| YOLO26n/11n pose | `bbox[4]`/`dfl[64]`, `keypoint[51]`, `objectness[1]` | objectness is a logit |
| YOLOv8n pose | `[65]` = `dfl[64]` + class logit, `keypoint[51]` | concatenated head |
| YOLO26n/11n seg | `bbox`, `cls[80]`, `mask_coeff[32]` + `proto[32,160,160]` | |
| YOLOv8n seg | `dfl[64]`, `cls[80]`, `conf[1]`, `mask_coeff[32]` + proto | |
| YOLO26n/11n obb | `bbox`, `cls[15]` (sigmoid), `angle[1]` | `dist2rbox`, rotated NMS |
| YOLOv8n obb | `[79]` = `dfl[64]` + `cls[15]`, `angle[1]` | concatenated head |
| YOLO26n/11n cls | `[1,1000]` | YOLO26n int8 in/fp16 out, YOLO11n fp16 in |
| YOLO26n depth | `[1,1,160,160]` | metric decode on the CPU |

Whether a tensor already carries a sigmoid is derived from its quantisation range
(`lo ≥ -0.02 && hi ≤ 1.06`) rather than hard-coded, which is what separates the sigmoided
`cls_*_sigmoid` heads from the logit-domain `objectness`/`angle` heads.

## Notes

- Boxes are reported in source-frame coordinates: the letterbox is undone after decoding.
- OBB models are DOTA-trained, so `bus.jpg` yields 0 detections at conf 0.25 (the harness
  prints the best raw class score so this is not mistaken for a decode failure).
- Depth post-processing includes the metric decode and an RGA upscale of the normalised map
  to the source resolution; `--trace-post` splits the two.
- The board exposes no power telemetry (only SARADC and the IMU), so the harness reports
  timing only.
