#!/usr/bin/env bash
# Native (C++) YOLO benchmark runner for the reCamera Pro (RV1126B).
#
# Cross-compiles native/bench_native, stages it plus a raw RGB test frame on the
# device, runs every YOLO model with the clocks pinned to their top OPPs and
# writes per-model logs plus a markdown summary.
#
#   ./scripts/bench_native.sh                     # full matrix, 100 runs
#   RUNS=30 WARMUP=10 ./scripts/bench_native.sh   # quick pass
#   ./scripts/bench_native.sh --only det          # one task family
#   ./scripts/bench_native.sh --skip-build --skip-push
#
# Environment: DEVICE_IP, DEVICE_USER, DEVICE_PASS, REMOTE_DIR, RUNS, WARMUP.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_IP="${DEVICE_IP:-192.168.42.1}"
DEVICE_USER="${DEVICE_USER:-root}"
DEVICE_PASS="${DEVICE_PASS:-recamera.1}"
REMOTE_DIR="${REMOTE_DIR:-/userdata/benchmark}"
RUNS="${RUNS:-100}"
WARMUP="${WARMUP:-20}"
SRC_W=1280
SRC_H=720
OUT_DIR="$ROOT/benchmark_results/native"

ONLY=""
DO_BUILD=1
DO_PUSH=1
DO_RUN=1
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --skip-build) DO_BUILD=0; shift ;;
    --skip-push) DO_PUSH=0; shift ;;
    --report-only) DO_BUILD=0; DO_PUSH=0; DO_RUN=0; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
rc() { sshpass -p "$DEVICE_PASS" ssh "${SSH_OPTS[@]}" "$DEVICE_USER@$DEVICE_IP" "$@"; }
rcp() { sshpass -p "$DEVICE_PASS" scp "${SSH_OPTS[@]}" -r "$1" "$DEVICE_USER@$DEVICE_IP:$2"; }

# label | task | model path (relative to the repo root)
MODELS=(
  "YOLO26n|det|models/yolo/yolo26_det/yolo26n_640x640_W8A8.rknn"
  "YOLO11n|det|models/yolo/yolo11_det/yolo11n_640x640_W8A8.rknn"
  "YOLO11s|det|models/yolo/yolo11_det/yolo11s_640x640_W8A8.rknn"
  "YOLOv8n|det|models/yolo/yolov8_det/yolov8n_640x640_W8A8.rknn"
  "YOLOv8s|det|models/yolo/yolov8_det/yolov8s_640x640_W8A8.rknn"
  "YOLO26n-pose|pose|models/yolo/yolo26_pose/yolo26n-pose_640x640_W8A8.rknn"
  "YOLO11n-pose|pose|models/yolo/yolo11_pose/yolo11n-pose_640x640_W8A8.rknn"
  "YOLOv8n-pose|pose|models/yolo/yolov8_pose/yolov8n-pose_640x640_W8A8.rknn"
  "YOLO26n-seg|seg|models/yolo/yolo26_seg/yolo26n-seg_640x640_W8A8.rknn"
  "YOLO11n-seg|seg|models/yolo/yolo11_seg/yolo11n-seg_640x640_W8A8.rknn"
  "YOLOv8n-seg|seg|models/yolo/yolov8_seg/yolov8n-seg_640x640_W8A8.rknn"
  "YOLO26n-obb|obb|models/yolo/yolo26_obb/yolo26n-obb_640x640_W8A8.rknn"
  "YOLO11n-obb|obb|models/yolo/yolo11_obb/yolo11n-obb_640x640_W8A8.rknn"
  "YOLOv8n-obb|obb|models/yolo/yolov8_obb/yolov8n-obb_640x640_W8A8.rknn"
  "YOLO26n-cls|cls|models/yolo/yolo26_cls/yolo26n-cls_224x224_W8A8.rknn"
  "YOLO11n-cls|cls|models/yolo/yolo11_cls/yolo11n-cls_224x224_W8A8.rknn"
  "YOLO26n-depth|depth|models/yolo/yolo26_depth/yolo26n-depth_640x640_W8A8.rknn"
)

mkdir -p "$OUT_DIR"

if [ "$DO_BUILD" = "1" ]; then
  echo "== building bench_native =="
  (cd "$ROOT/native" && ./build.sh)
fi

if [ "$DO_PUSH" = "1" ]; then
  echo "== staging $REMOTE_DIR on $DEVICE_IP =="
  rc "mkdir -p $REMOTE_DIR/models"
  rcp "$ROOT/native/build/bench_native" "$REMOTE_DIR/"
  # deterministic test frame: the repo's own asset, decoded to raw RGB888
  RGB="$OUT_DIR/bench_${SRC_W}x${SRC_H}.rgb"
  ffmpeg -loglevel error -y -i "$ROOT/assets/bus.jpg" \
    -vf "scale=${SRC_W}:${SRC_H}" -pix_fmt rgb24 -f rawvideo "$RGB"
  rcp "$RGB" "$REMOTE_DIR/bench.rgb"
  # a second NPU client would distort every measurement
  rc "pkill -x yolo_rt 2>/dev/null; true"
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r label task path <<< "$entry"
    [ -n "$ONLY" ] && [ "$task" != "$ONLY" ] && [ "$label" != "$ONLY" ] && continue
    rcp "$ROOT/$path" "$REMOTE_DIR/models/"
  done
fi

if [ "$DO_RUN" = "1" ]; then
echo "== benchmarking (runs=$RUNS warmup=$WARMUP) =="
for entry in "${MODELS[@]}"; do
  IFS='|' read -r label task path <<< "$entry"
  [ -n "$ONLY" ] && [ "$task" != "$ONLY" ] && [ "$label" != "$ONLY" ] && continue
  model="$(basename "$path")"
  log="$OUT_DIR/${label}.log"
  echo "-- $label ($task)"
  rc "cd $REMOTE_DIR && LD_LIBRARY_PATH=/oem/usr/lib ./bench_native \
      --model models/$model --image $REMOTE_DIR/bench.rgb --src ${SRC_W}x${SRC_H} \
      --runs $RUNS --warmup $WARMUP" > "$log" 2>&1 || { echo "   FAILED (see $log)"; continue; }
  grep -E "^(total|fps|npu|preprocess|postprocess|detections|top5|depth) " "$log" | sed 's/^/   /'
done
fi

{
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r label task path <<< "$entry"
    echo "$task|$label"
  done
} > "$OUT_DIR/order.txt"

echo "== writing $OUT_DIR/results.md =="
OUT_DIR="$OUT_DIR" RUNS="$RUNS" WARMUP="$WARMUP" SRC_W="$SRC_W" SRC_H="$SRC_H" python3 - <<'PYEOF'
import os, re, glob

out_dir = os.environ["OUT_DIR"]
runs, warmup = os.environ["RUNS"], os.environ["WARMUP"]
rows = {}
for log in sorted(glob.glob(os.path.join(out_dir, "*.log"))):
    label = os.path.splitext(os.path.basename(log))[0]
    text = open(log, errors="ignore").read()
    def grab(pattern, default="?"):
        m = re.search(pattern, text, re.M)
        return m.group(1).strip() if m else default
    rows[label] = {
        "task": grab(r"^task\s*:\s*(\S+)"),
        "input": grab(r"^input\s*:\s*(\S+)"),
        "pre": grab(r"^preprocess : avg\s*([\d.]+)"),
        "npu": grab(r"^npu\s*: avg\s*([\d.]+)"),
        "post": grab(r"^postprocess: avg\s*([\d.]+)"),
        "total": grab(r"^total\s*: avg\s*([\d.]+)"),
        "min": grab(r"^total\s*: avg\s*[\d.]+\s*ms\s*min\s*([\d.]+)"),
        "max": grab(r"^total\s*: avg\s*[\d.]+\s*ms\s*min\s*[\d.]+\s*max\s*([\d.]+)"),
        "fps": grab(r"^fps\s*:\s*([\d.]+)"),
        "detail": grab(r"^(?:detections|top5|depth)\s*:\s*(.+)$"),
    }

order = ["det", "pose", "seg", "obb", "cls", "depth"]
# keep the runner's model order inside each task family
ordered = []
for line in open(os.path.join(out_dir, "order.txt")):
    task, label = line.strip().split("|")
    if label in rows and (task, label) not in ordered:
        ordered.append((task, label))
for label in rows:
    if (rows[label]["task"], label) not in ordered:
        ordered.append((rows[label]["task"], label))
titles = {"det": "Object Detection", "pose": "Pose Estimation", "seg": "Instance Segmentation",
          "obb": "Oriented Bounding Box", "cls": "Classification", "depth": "Depth Estimation"}

lines = ["# Native YOLO Benchmark Results (C++ / RV1126B)", "",
         f"Device: Seeed reCamera Pro (Rockchip RV1126B, Cortex-A7 x4 @ 1608 MHz, NPU @ 950 MHz).",
         f"Harness: `native/bench_native.cpp`, `--runs {runs} --warmup {warmup}`.",
         f"Input: `assets/bus.jpg` decoded to a {os.environ['SRC_W']}x{os.environ['SRC_H']} raw RGB888 frame, letterboxed to the model input by RGA.",
         "Latency = preprocess (RGA) + NPU inference + postprocess (decode + NMS), measured serially; FPS = 1000 / latency.", ""]
lines += ["| Task | Model | Input | Pre (ms) | NPU (ms) | Post (ms) | Total (ms) | Min | Max | FPS |",
          "|------|-------|-------|---------:|---------:|----------:|-----------:|----:|----:|----:|"]
csv = ["task,model,input,pre_ms,npu_ms,post_ms,total_ms,min_ms,max_ms,fps"]
for task, label in ordered:
    r = rows[label]
    lines.append(f"| {titles[task]} | {label} | {r['input']} | {r['pre']} | {r['npu']} | "
                 f"{r['post']} | **{r['total']}** | {r['min']} | {r['max']} | **{r['fps']}** |")
    csv.append(f"{task},{label},{r['input']},{r['pre']},{r['npu']},{r['post']},"
               f"{r['total']},{r['min']},{r['max']},{r['fps']}")
lines += ["", "## Per-model output sanity", ""]
for task, label in ordered:
    lines.append(f"- **{label}** ({task}): {rows[label]['detail']}")
open(os.path.join(out_dir, "results.md"), "w").write("\n".join(lines) + "\n")
open(os.path.join(out_dir, "results.csv"), "w").write("\n".join(csv) + "\n")
print("\n".join(lines))
PYEOF
