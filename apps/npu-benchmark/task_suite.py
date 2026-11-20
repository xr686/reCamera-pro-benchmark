import hashlib
import json
import time
import gc
from cv2 import error as cv_error

import numpy as np

from task_decode import decode
from task_inputs import prepare, read_sample, fingerprint
from task_models import MODELS
from task_quality import quality, summarize


def check_digest(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path.name}")


def compare(out, reference):
    scores = []
    observed = fingerprint(out)
    if len(observed) != len(reference):
        raise ValueError("ONNX and RKNN output counts differ")
    for actual, want in zip(observed, reference):
        if np.prod(actual["shape"]) != np.prod(want["shape"]):
            raise ValueError(
                f"Output shape {actual['shape']} versus ONNX {want['shape']}"
            )
        a, b = (np.array(x["values"]) for x in (actual, want))
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(np.dot(a, b) / denom) if denom > 0 else float(np.array_equal(a, b))
        scores.append(
            {
                "cosine": cos,
                "relative_l2": float(
                    np.linalg.norm(a - b) / max(1e-8, np.linalg.norm(b))
                ),
            }
        )
    return scores


def one_model(app, base, model, metadata, records, audits):
    from rknnlite.api import RKNNLite

    path = base / "compiled" / (model.file + ".rknn")
    check_digest(path, metadata["sha256"])
    from clip_benchmark import gallery

    reference_gallery = (
        gallery(base / "fixtures", model) if model.decoder == "clip-rank" else None
    )
    runtime = RKNNLite()
    times, checks, rows, cleanup = [], [], [], []
    heavy = model.decoder in ("speech", "subword", "gghl", "edge", "restore", "inpaint")
    warmup = 2 if heavy else 5
    count = max(len(records), 4 if heavy else 30)
    try:
        if runtime.load_rknn(str(path)) or runtime.init_runtime():
            raise ValueError("RKNN initialization failed")
        for i in range(-warmup, count):
            if app._stop_flag:
                raise InterruptedError("App stopped")
            if i in (-warmup, 0, count - 1) or i % 10 == 0:
                check = dict(iteration=i, **app.check_exclusive())
                checks.append(check)
                audits.append(check)
            rec = records[i % len(records)]
            data = read_sample(base / "fixtures", rec)
            t0 = time.perf_counter_ns()
            inputs = prepare(model, data, metadata)
            # For image networks the vendor runtime consumes interleaved HWC.
            # We keep ONNX tensor contracts for the independent CPU reference.
            if model.layout == "nchw":
                inputs = [
                    np.ascontiguousarray(t.transpose(0, 2, 3, 1)) if t.ndim == 4 else t
                    for t in inputs
                ]
            t1 = time.perf_counter_ns()
            out = runtime.inference(
                inputs=inputs,
                data_format=["nhwc" if t.ndim == 4 else "nchw" for t in inputs],
            )
            t2 = time.perf_counter_ns()
            if out is None:
                raise ValueError("RKNN inference returned no outputs")
            result = decode(model, out, metadata.get("tokens"), reference_gallery)
            t3 = time.perf_counter_ns()
            if i >= 0:
                times.append([(t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6])
            if 0 <= i < len(records):
                if any(not np.isfinite(t).all() for t in out):
                    raise ValueError(
                        f"Non-finite network output(s): {[j for j, t in enumerate(out) if not np.isfinite(t).all()]}"
                    )
                info = {
                    "file": rec["file"],
                    "quality": quality(base / "fixtures", model, rec, result),
                }
                golden = metadata.get("reference", {}).get(rec["file"])
                if golden:
                    info["onnx"] = compare(out, golden)
                rows.append(info)
            # RKNNLite 2.3.2 can leave native buffers in ctypes reference cycles.
            # Drop tensor views before collecting, and measure this cost independently.
            start_cleanup = time.perf_counter_ns()
            del inputs, result, out
            gc.collect()
            if i >= 0:
                cleanup.append((time.perf_counter_ns() - start_cleanup) / 1e6)
    finally:
        runtime.release()
        gc.collect()
    means = np.asarray(times).mean(0)
    scores = summarize([r["quality"] for r in rows])
    for r in rows:
        r["quality"].pop("embedding", None)
    cos = [t["cosine"] for r in rows for t in r.get("onnx", [])]
    return dict(
        status="ok",
        frames=count,
        unique=len(records),
        warmup=warmup,
        ms=means.tolist(),
        p95_infer_ms=float(np.percentile(np.array(times)[:, 1], 95)),
        fps=1000 / float(means.sum()),
        cleanup_ms=cleanup,
        fps_with_cleanup=1000 / float(means.sum() + np.mean(cleanup)),
        metrics=scores,
        onnx_cosine=float(np.mean(cos)) if cos else None,
        checks=checks,
        rows=rows,
        times=times,
        sha256=metadata["sha256"],
    )


def run_tasks(app, base, checkpoint, selected):
    suite = json.loads((base / "suite.json").read_text())
    root = (base / "fixtures").resolve()

    def verified_file(name):
        p = (root / name).resolve()
        if not p.is_relative_to(root) or name not in suite["files"]:
            raise ValueError(f"Unregistered fixture {name}")
        return p

    for name, sha in suite["files"].items():
        check_digest(verified_file(name), sha)
    groups = json.loads(verified_file("groups.json").read_text())
    for records in groups.values():
        for rec in records:
            for key in (
                "file",
                "template",
                "target",
                "mask",
                "disparity",
                "edges",
                "restore",
                "hole",
                "citymask",
                "objectmask",
            ):
                if key in rec:
                    verified_file(rec[key])
    results = []
    for model in MODELS:
        if selected and model.file not in selected:
            continue
        if not selected and model.file not in suite["models"]:
            continue
        app.check_exclusive()
        if app._stop_flag:
            raise InterruptedError("App stopped")
        item = dict(
            name=model.name,
            task=model.task,
            id=model.file,
            size=list(model.size),
            corpus=model.corpus,
        )
        audits = []
        try:
            item.update(
                one_model(
                    app,
                    base,
                    model,
                    suite["models"][model.file],
                    groups[model.corpus],
                    audits,
                )
            )
        except (ValueError, KeyError, IndexError, StopIteration, cv_error) as exc:
            if app._stop_flag:
                raise InterruptedError("App stopped by supervisor") from exc
            item.update(
                status="error", error=f"{type(exc).__name__}: {exc}", checks=audits
            )
        gc.collect()
        results.append(item)
        checkpoint(results)
        print(
            f"Task benchmark {model.name}: {item['status']} {item.get('error', '')}",
            flush=True,
        )
        if item["status"] == "ok":
            app.emit(
                results=[
                    {
                        "name": model.name,
                        "task": model.task,
                        "fps": item["fps"],
                        "quality": item["metrics"],
                    }
                ]
            )
    return results


def markdown(results):
    lines = [
        "",
        "## 跨任务基准",
        "",
        "FPS 表示单路输入单元/秒。语音输入为一段波形。每个样本的预测质量只计一次。",
        "",
        "| 任务 | 模型 | 输入 | 前处理 ms | 同步推理 ms | 后处理 ms | FPS | 任务质量 |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        size = "×".join(map(str, r["size"]))
        if r["status"] != "ok":
            lines.append(
                f"| {r['task']} | {r['name']} | {size} | — | — | — | — | {r['error']} |"
            )
            continue
        metric = (
            "，".join(f"{k} {v:.4f}" for k, v in r["metrics"].items()) or "无任务真值"
        )
        pre, infer, post = r["ms"]
        lines.append(
            f"| {r['task']} | {r['name']} | {size} | {pre:.2f} | {infer:.2f} | {post:.2f} | {r['fps']:.2f} | {metric} |"
        )
    return "\n".join(lines) + "\n"
