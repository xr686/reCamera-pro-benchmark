"""Render a firmware-specific CSV and Markdown comparison using only finished board runs."""

import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_models import MODELS
from extra_tasks import VISION, SEG
from render_multitask import PRE, POST, shape

EXTRA_PRE = {
    "cropnet": "权重对应短边缩放、中心裁剪、RGB 标准化",
    "gray01": "缩放、灰度 /255",
    "logmel": "16 kHz 单声道 → 40×98 log-Mel",
    "signal": "训练集均值和标准差归一化",
    "saliency": "RGB 归一化、按通道标准化",
    "line4": "面积插值、BGR 加常数通道",
    "clipimage": "短边 224、中心裁剪、RGB CLIP 标准化",
    "cliptext": "英语 BPE、补齐至 20 token",
}
EXTRA_POST = {
    "extra-class": "logits 排序",
    "voc21": "21 类像素 argmax",
    "saliency": "显著图归一化和二值化",
    "photometric": "输出像素裁剪、转灰度/BGR",
    "stylize": "RGB 裁剪、转 BGR",
    "features": "网格解码、局部极大值、采样并归一化描述子",
    "lines": "中心峰筛选、回归端点、长度过滤",
    "clip-rank": "512 维向量归一化，与对端图库做余弦检索",
}
FAMILY = {
    "clip-rank": "图文检索",
    "features": "视觉几何",
    "lines": "视觉几何",
    "voc21": "语义分割和显著目标",
    "saliency": "语义分割和显著目标",
    "photometric": "图像增强和生成",
    "stylize": "图像增强和生成",
}


def metrics(d):
    def format_one(k, v):
        if "dB" in k or "MAE" in k or "匹配数" in k or "Pearson" in k:
            return f"{k} {v:.3f}"
        return f"{k} {v:.2%}"

    return "，".join(format_one(k, v) for k, v in d.items())


def main(args):
    work = args.work
    extra = json.loads((work / "suite.json").read_text())
    previous = json.loads((args.original / "suite.json").read_text())
    by = {m.file: m for m in MODELS}
    addon = set(extra["models"])
    expected = (
        set(previous["models"]) | addon | {"resnet18", "mobilenet-v2", "shufflenet-v2"}
    )
    ref = json.loads((work / "onnx-quality.json").read_text())
    result = {}
    audits = []
    allrows = {}
    for path in args.reports:
        d = json.loads(path.read_text())
        audit = json.loads(path.with_name(path.stem + "-audit.json").read_text())
        assert all(s["running"] == ["npu-benchmark"] for s in audit["samples"][:-1])
        assert len(d["idle"]) >= 3 and all(
            s.rsplit(":", 1)[-1].strip() == "0%" for s in d["idle"][-3:]
        )
        for r in d["models"] + d.get("tasks", []):
            if r.get("status", "ok") != "ok":
                continue
            assert r["checks"] and all(
                all(
                    pid == str(d["pid"]) or proc == "rkipc"
                    for pid, proc in c["owners"].items()
                )
                for c in r["checks"]
            )
            k = r["id"]
            if k in addon and r["sha256"] != extra["models"][k]["sha256"]:
                # The original SuperPoint descriptor head was superseded and rerun.
                continue
            audits.extend(r["checks"])
            allrows[k] = r
            if k in by:
                m = by[k]
                mbytes = (
                    (
                        (work if k in addon else args.original)
                        / "compiled"
                        / (k + ".rknn")
                    )
                    .stat()
                    .st_size
                )
                assert (
                    r["sha256"]
                    == (extra if k in addon else previous)["models"][k]["sha256"]
                )
                size = shape(m)
                if m.pixels == "signal":
                    size = f"{m.size[0]} 通道 × {m.size[1]} 点"
                if m.pixels == "logmel":
                    size = "1 s → 40×98"
                if m.pixels == "cliptext":
                    size = "20 token"
                data = dict(
                    pre_ms=r["ms"][0],
                    infer_ms=r["ms"][1],
                    p95_ms=r["p95_infer_ms"],
                    post_ms=r["ms"][2],
                    fps=r["fps"],
                    with_gc=r["fps_with_cleanup"],
                    unique=r["unique"],
                    calls=r["frames"],
                    quality=r["metrics"],
                    precision="FP16",
                    size=size,
                    task=m.task,
                    mib=mbytes / 2**20,
                    prep=EXTRA_PRE.get(m.pixels, PRE.get(m.pixels)),
                    post=EXTRA_POST.get(m.decoder, POST.get(m.decoder)),
                    cos=r.get("onnx_cosine"),
                )
                if k not in addon:
                    section = "原有视觉、音频模型复测"
                elif m.pixels == "signal":
                    section = "传感器、时序和光谱"
                elif k in VISION:
                    section = "图像分类"
                elif k in SEG:
                    section = "语义分割和显著目标"
                else:
                    section = FAMILY.get(m.decoder, "语音指令和数字识别")
            else:
                n = len(r["samples"])
                s = r["ms"]
                data = dict(
                    pre_ms=s["pre"]["mean"],
                    infer_ms=s["infer"]["mean"],
                    p95_ms=s["infer"]["p95"],
                    post_ms=s["post"]["mean"],
                    fps=r["fps"],
                    with_gc=None,
                    unique=n,
                    calls=n,
                    quality={"Top-1": r["top1"], "Top-5": r["top5"]},
                    precision="FP16",
                    size="224×224",
                    task="图像分类",
                    mib=(args.app / "models" / (k + ".rknn")).stat().st_size / 2**20,
                    prep="短边缩放、中心裁剪、RGB",
                    post="1000 类排序",
                    cos=None,
                )
                section = "原有视觉、音频模型复测"
            result[k] = dict(
                id=k,
                name=r["name"],
                group=section,
                **data,
                sha256=r["sha256"],
                run=path.name,
            )
    missing = expected - set(result)
    rows = list(result.values())
    new = [r for r in rows if r["id"] in addon]
    groups = (
        "图像分类",
        "语义分割和显著目标",
        "视觉几何",
        "图像增强和生成",
        "语音指令和数字识别",
        "图文检索",
        "传感器、时序和光谱",
        "原有视觉、音频模型复测",
    )
    lines = [
        "# reCamera Pro V1.3.2 模型基准",
        "",
        "在 RV1126B 真机上，本轮验证了 **%d 个模型配置、%d 个任务标签**，其中 %d 个为新增，%d 个为既有配置在这版固件上的复测。"
        % (len(rows), len({r["task"] for r in rows}), len(new), len(rows) - len(new)),
        "",
        "计数按实际权重和输入契约进行。CLIP 的图像塔和文本塔分别计时。14 个时序和光谱配置使用 9 种建模方法，部分方法在多个数据集上独立训练。各数据集保留各自的任务名。",
        "",
        "选型对照了参考仓库的 YOLOv8/11/26、PP-OCRv4 和 Zipformer 清单。此次新增模型均在该清单之外。参考仓库的 YOLO 使用锁频、RGA 和原生 C++，本报告采用固件应用的 Python SDK 和自动调频，跨仓库比较时应保留这些条件。[参考仓库固定版本](https://github.com/xr686/reCamera-pro-benchmark/tree/11929a7582eb58bb025727c818e0eb50c4318eae)",
        "",
        "下表所有阶段耗时单位均为 ms，推理列为 RKNNLite 从提交输入到取回输出的同步耗时。三段吞吐量按 1000 /（平均前处理 + 同步推理 + 后处理）计算。“含回收”再计入各帧显式释放缓冲区的时间。图像用帧/s，语音、文本和时序用输入样本/s。权重均为 RV1126B FP16 RKNN，batch=1，程序逐个加载。",
        "",
    ]
    if missing:
        lines.extend(["**未获有效成绩的配置**　" + ", ".join(sorted(missing)), ""])
    for g in groups:
        subset = [r for r in rows if r["group"] == g]
        if not subset:
            continue
        lines.extend(
            [
                "## " + g,
                "",
                "| 任务 / 模型 | 输入 | 权重 MiB | 样本/计时次数 | 前处理 | 同步推理 | 推理 P95 | 后处理 | 三段吞吐 | 含回收 | 样本质量 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for r in subset:
            gc = "自动" if r["with_gc"] is None else f"{r['with_gc']:.2f}"
            lines.append(
                f"| {r['task']} / {r['name']} | {r['size']} | {r['mib']:.2f} | {r['unique']}/{r['calls']} | {r['pre_ms']:.2f} | {r['infer_ms']:.2f} | {r['p95_ms']:.2f} | {r['post_ms']:.2f} | {r['fps']:.2f} | {gc} | {metrics(r['quality'])} |"
            )
        lines.append("")
    lines.extend(["## 端到端图文检索", ""])
    if {"clip-text", "clip-image"} <= set(allrows):
        fx = json.loads((work / "fixtures/groups.json").read_text())

        def vec(k, g):
            labels = {r["file"]: r["label"] for r in fx[g]}
            rr = allrows[k]["rows"]
            return np.array([r["quality"]["clip_projection"] for r in rr]), np.array(
                [labels[r["file"]] for r in rr]
            )

        im, yi = vec("clip-image", "clip-images")
        tx, yt = vec("clip-text", "clip-texts")
        cats = np.unique(yt)
        centers = np.array([tx[yt == c].mean(0) for c in cats])
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        lines.extend(
            [
                f"使用本轮两侧板端实际输出的投影向量重算余弦检索。每类 3 条提示的平均单位向量作为图像分类原型，100 张图的 Top-1 为 {np.mean(cats[(im @ centers.T).argmax(1)] == yi):.2%}。以 30 条提示各自检索完整的 100 张图，首选图像属于正确类别的比例为 {np.mean(yi[(tx @ im.T).argmax(1)] == yt):.2%}。",
                "",
            ]
        )
    lines.extend(
        [
            "## 数值对照和复测 ID",
            "",
            "所有新增配置均在主机用同输入、同标注运行原始 ONNX。cos 是在至多 3 个输入上，每个输出张量均匀抽取最多 512 个位置后得到的平均余弦相似度。它用于定位转换偏差。CSV 收录每个配置的 ID、权重 SHA-256、处理方式、汇总时延和质量，原始 JSON 保留逐次时延。",
            "",
            "| 新增配置 ID | 前处理 | 后处理 | 端侧 / ONNX cos | CPU ONNX 样本质量 |",
            "|---|---|---|---:|---|",
        ]
    )
    for r in new:
        c = r["cos"]
        lines.append(
            f"| `{r['id']}` | {r['prep']} | {r['post']} | {c:.5f} | {metrics(ref[r['id']]['metrics'])} |"
        )
    lines.extend(
        [
            "",
            (args.app / "tools/extra-methods.md").read_text().strip(),
            "",
            "## 独占、复测与限制",
            "",
        ]
    )
    freq = [c["freq_hz"] / 1e6 for c in audits]
    temp = [c["temperature_mC"] / 1000 for c in audits]
    mhz = f"{min(freq):g}" if min(freq) == max(freq) else f"{min(freq):g}–{max(freq):g}"
    lines.append(
        f"来源为 {len(args.reports)} 份完整板端记录。每轮先申请固件的独占 NPU 租约，连续 3 次读到 0% 后才开始。应用同时检查打开 /dev/rknpu 的进程。有效模型共计 {len(audits)} 次检查，所有者只包含对应测试进程及固件 broker rkipc。检查时频率 {mhz} MHz，温度 {min(temp):.1f}–{max(temp):.1f} °C。AppMgr 每约 3 秒进行独立采样，报告落盘之后可出现内置应用接管。"
    )
    lines.extend(
        [
            "",
            "基准应用启动前暂停原有的 motion-detect，每组计划结束后通过 AppMgr 恢复。固件为 V1.3.2，Linux 6.1.157，NPU 驱动 0.9.8。本轮使用的应用版本和分组见原始证据。",
            "",
            "同一小样本的多次计时不增加质量分母。真实使用时的目标大小、光照、说话人和传感器分布均会影响精度。每个模型的预热和测量次数都在原始 JSON 中，常规模型预热 5 次并至少计时 30 次，既有大模型和基线分类器遵循各自原测量协议。",
            "",
        ]
    )
    args.output.write_text("\n".join(lines))
    fields = list(rows[0]) + ["cpu_metrics"]
    with args.csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                dict(
                    r,
                    quality=json.dumps(r["quality"], ensure_ascii=False),
                    cpu_metrics=json.dumps(
                        ref.get(r["id"], {}).get("metrics"), ensure_ascii=False
                    ),
                )
            )
    print("valid", len(rows), "new", len(new), "missing", sorted(missing))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "original", "app", "output", "csv"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("reports", nargs="+", type=Path)
    main(p.parse_args())
