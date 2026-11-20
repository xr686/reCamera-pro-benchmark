import hashlib
import json
import os
import platform
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from kit.app import App
from kit.config import app_dir_of, appdata_root
from kit.resources import ExternalNpuLease

MODELS = {"resnet18": "ResNet-18", "shufflenet-v2": "ShuffleNet V2 x1.0", "mobilenet-v2": "MobileNet V2"}
WARMUP = 20
LOAD = Path("/proc/rknpu/load")
FREQ = Path("/sys/class/devfreq/22000000.npu")


def preprocess(bgr):
    h, w = bgr.shape[:2]
    resized = cv2.resize(bgr, (256 * w // min(h, w), 256 * h // min(h, w)),
                         interpolation=cv2.INTER_LINEAR)
    y, x = (round((d - 224) / 2) for d in resized.shape[:2])
    return cv2.cvtColor(resized[y:y + 224, x:x + 224], cv2.COLOR_BGR2RGB)[None, ...]


def top_five(outputs):
    if outputs is None or len(outputs) != 1:
        raise ValueError("Expected one RKNN output tensor")
    logits = outputs[0].reshape(-1)
    if logits.size != 1000 or not np.isfinite(logits).all():
        raise ValueError("Expected 1000 finite ImageNet logits")
    return np.argsort(-logits, kind="stable")[:5].tolist()


def load_samples(directory):
    payload = (directory / "index.json").read_bytes()
    manifest = json.loads(payload)
    records = manifest["samples"]
    names = set()
    for sample in records:
        name, label = sample["file"], sample["label"]
        image = (directory / name).resolve()
        if not image.is_relative_to(directory.resolve()) or name in names or not 0 <= label < 1000:
            raise ValueError(f"Invalid sample {name}")
        names.add(name)
        if hashlib.sha256(image.read_bytes()).hexdigest() != sample["sha256"]:
            raise ValueError(f"Image digest mismatch for {name}")
    if not records:
        raise ValueError("Empty evaluation set")
    return manifest["dataset"], hashlib.sha256(payload).hexdigest(), records


def owners():
    """RV1126B exposes its NPU as a misc device. Capture its open device handles."""
    clients = {}
    for pid_dir in Path("/proc").glob("[0-9]*"):
        try:
            for fd in (pid_dir / "fd").iterdir():
                try:
                    link = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if link == "/dev/rknpu":
                    clients[pid_dir.name] = (pid_dir / "comm").read_text().strip()
                    break
        except (FileNotFoundError, ProcessLookupError):
            continue
    return clients


def average_stats(samples, key):
    values = [row[key] for row in samples]
    return dict(mean=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)))


def atomic_write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data)
    tmp.replace(path)


def markdown(report):
    if not report["models"]:
        return f"# RV1126B 模型基准实测\n\n设备时间（UTC） {report['utc']}。\n"
    lines = ["# RV1126B 模型基准实测", "",
             f"设备时间（UTC） {report['utc']}，数据集 {report['dataset']}，每个模型 {report['count']} 张，预热 {WARMUP} 次。",
             "", "| 模型 | 输入 RGB | 前处理 ms | 同步推理 ms | 推理 P95 ms | 后处理 ms | 串行 FPS | 含读取解码 FPS | Top-1 | Top-5 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for result in report["models"]:
        stats = result["ms"]
        lines.append(f"| {result['name']} | 1×224×224×3 | {stats['pre']['mean']:.3f} | "
                     f"{stats['infer']['mean']:.3f} | {stats['infer']['p95']:.3f} | {stats['post']['mean']:.3f} | "
                     f"{result['fps']:.2f} | {result['decoded_fps']:.2f} | "
                     f"{result['top1']:.2%} | {result['top5']:.2%} |")
    lines += ["", "RKNN FP16，batch 1。OpenCV 单线程，短边缩放到 256、中心裁剪、BGR 转 RGB。"
              "模型内置均值和标准差归一化。后处理将 1000 个输出分数排序，返回前五类。",
              "", "各阶段耗时为算术平均值。串行 FPS 为 1000 / 平均（前处理 + 同步推理 + 后处理）毫秒。"
              "含读取解码 FPS 另计 cv2.imread；两列均不含采集、绘制与发布。"
              "准确率使用验证图片的原始 ImageNet 类别编号。完整样本、时延及独占检查见 report.json。"]
    return "\n".join(lines) + "\n"


class NpuBenchmark(App):
    id = "npu-benchmark"
    name = "NPU Benchmark"
    owns_loop = True
    needs_frames = False
    needs_model = False

    def prepare_runtime(self):
        self.data_dir = Path(appdata_root()) / self.id
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir = self.data_dir / "fixtures"
        self.selected = {x.strip() for x in self.config.get("model_ids", "").split(",") if x.strip()}
        if not self.selected and self.config.get("benchmark_scope", "new_families") == "new_families":
            from architecture_models import FAMILIES
            manifest = self.data_dir / "architectures" / "suite.json"
            if manifest.is_file():
                self.selected = set(json.loads(manifest.read_text())["models"]) & FAMILIES.keys()
            if not self.selected:
                raise ValueError("Install the architecture resource pack or select another benchmark scope")
        if not self.selected or self.selected.intersection(MODELS):
            self.dataset, self.index_sha, self.records = load_samples(self.sample_dir)
        else:
            self.dataset, self.index_sha, self.records = "Per-model fixtures", None, []
        if os.environ.get("RECAMERA_NPU_MODE") != "legacy-direct":
            raise RuntimeError("Exclusive App Center NPU allocation required")
        self.lease = ExternalNpuLease()
        if self.lease.backend != "broker":
            raise RuntimeError("Firmware NPU broker required")
        self.lease.acquire(timeout=10)
        self.lease.ready()
        cv2.setNumThreads(1)

    def check_exclusive(self):
        clients = owners()
        # rkipc owns the firmware broker. Its built-in model is drained on acquire.
        if not self.lease.alive() or any(pid != str(os.getpid()) and name != "rkipc"
                                         for pid, name in clients.items()):
            raise RuntimeError(f"Exclusive NPU ownership lost: {clients}")
        return {"owners": clients, "freq_hz": int((FREQ / "cur_freq").read_text()),
                "temperature_mC": int(Path("/sys/class/thermal/thermal_zone0/temp").read_text())}

    def measure(self, runtime, sample):
        start = time.perf_counter_ns()
        image = cv2.imread(str(self.sample_dir / sample["file"]), cv2.IMREAD_COLOR)
        decoded = time.perf_counter_ns()
        if image is None:
            raise ValueError(f"Cannot decode {sample['file']}")
        tensor = preprocess(image)
        prepared = time.perf_counter_ns()
        outputs = runtime.inference(inputs=[tensor], data_format=["nhwc"])
        inferred = time.perf_counter_ns()
        predictions = top_five(outputs)
        end = time.perf_counter_ns()
        return dict(file=sample["file"], label=sample["label"], top5=predictions,
                    read_ms=(decoded-start)/1e6, pre=(prepared-decoded)/1e6,
                    infer=(inferred-prepared)/1e6, post=(end-inferred)/1e6,
                    pipeline=(end-decoded)/1e6, decoded_pipeline=(end-start)/1e6)

    def benchmark_model(self, slug, name):
        from rknnlite.api import RKNNLite

        path = Path(app_dir_of(self)) / "models" / f"{slug}.rknn"
        runtime = RKNNLite()
        samples, checks = [], []
        try:
            if runtime.load_rknn(str(path)) or runtime.init_runtime():
                raise RuntimeError(f"Failed to load {name}")
            for i in range(WARMUP + len(self.records)):
                if self._stop_flag:
                    return None
                if i % 25 == 0 or i == WARMUP:
                    checks.append(dict(iteration=i, **self.check_exclusive()))
                row = self.measure(runtime, self.records[(i - WARMUP) % len(self.records)])
                if i >= WARMUP:
                    samples.append(row)
            checks.append(dict(iteration=WARMUP + len(samples), **self.check_exclusive()))
        finally:
            runtime.release()
        stats = {key: average_stats(samples, key) for key in
                 ("pre", "infer", "post", "pipeline", "read_ms", "decoded_pipeline")}
        return dict(name=name, id=slug, ms=stats, fps=1000/stats["pipeline"]["mean"],
                    decoded_fps=1000/stats["decoded_pipeline"]["mean"],
                    top1=sum(s["label"] == s["top5"][0] for s in samples)/len(samples),
                    top5=sum(s["label"] in s["top5"] for s in samples)/len(samples),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), checks=checks, samples=samples)

    def run(self):
        self.check_exclusive()
        idle = []
        for _ in range(20):
            time.sleep(1)
            idle.append(LOAD.read_text().strip())
            if self._stop_flag:
                return
            if len(idle) >= 3 and all(re.findall(r"(\d+)%", value) == ["0"] for value in idle[-3:]):
                break
        else:
            raise RuntimeError(f"NPU was busy before model initialization: {idle}")
        report = dict(utc=datetime.now(timezone.utc).isoformat(), dataset=self.dataset,
                      count=len(self.records), index_sha256=self.index_sha, pid=os.getpid(),
                      warmup=WARMUP, idle=idle, kernel=platform.release(), opencv=cv2.__version__,
                      npu_driver=Path("/proc/rknpu/version").read_text().strip(),
                      governor=(FREQ / "governor").read_text().strip(), models=[])
        from task_models import MODELS as task_models
        selected = self.selected
        unknown = selected - set(MODELS) - {m.file for m in task_models}
        if unknown:
            raise ValueError(f"Unknown benchmark model IDs: {sorted(unknown)}")
        for slug, name in MODELS.items():
            if selected and slug not in selected:
                continue
            result = self.benchmark_model(slug, name)
            if result is None:
                return
            report["models"].append(result)
            self.emit(results=[dict(task="图像分类", quality={"Top-1": result["top1"], "Top-5": result["top5"]},
                                    **{key: result[key] for key in ("name", "fps", "top1", "top5")})])
        document = markdown(report)
        import task_suite
        suites = []
        requested = selected - set(MODELS)
        if not selected or requested:
            for folder in ("multitask", "multitask-extra", "multitask-extended", "multitask-field", "multitask-sequence", "architectures"):
                path = self.data_dir / folder
                if (path / "suite.json").is_file():
                    ids = set(json.loads((path / "suite.json").read_text())["models"])
                    suites.append((path, ids))
            installed = set().union(*(ids for _, ids in suites))
            if requested - installed:
                raise ValueError(f"Missing benchmark resources: {sorted(requested - installed)}")
        tasks = []
        def checkpoint(results):
            atomic_write(self.data_dir / "multitask-progress.json", json.dumps(
                {"utc": report["utc"], "pid": report["pid"], "models": tasks + results}, ensure_ascii=False))
        for path, ids in suites:
            chosen = (requested & ids) if selected else ids
            if chosen:
                tasks.extend(task_suite.run_tasks(self, path, checkpoint, chosen))
        if tasks:
            report["tasks"] = tasks
            document += task_suite.markdown(tasks)
        atomic_write(self.data_dir / "report.md", document)
        atomic_write(self.data_dir / "report.json", json.dumps(report, ensure_ascii=False))
        print(f"Benchmark complete: {self.data_dir}/report.md", flush=True)

    def finish(self):
        try:
            if getattr(self, "lease", None):
                self.lease.release()
        finally:
            super().finish()
