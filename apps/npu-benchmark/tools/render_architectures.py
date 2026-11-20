"""Generate the per-model benchmark and a deduplicated ImageNet backbone index."""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architecture_models import FAMILIES

# The first ID is the retained representative. Historical task heads do not
# become entries in this same-task index.
HISTORICAL = {
    "ResNet": ("resnet18", "resnext50_32x4d", "skresnet", "resnest"),
    "MobileNet": ("mobilenet_v3_small", "mobilenet-v2", "mobilenetv4-small"),
    "ShuffleNet": ("shufflenet_v2_x0_5", "shufflenet-v2"),
    "GhostNet": ("ghostnet", "repghost"),
    "RegNet": ("regnet_x_400mf", "regnety-004"),
    "ConvNeXt": ("convnext_tiny", "convnext-atto"),
    "SqueezeNet": ("squeezenet1_1",),
    "MNASNet": ("mnasnet0_5",),
    "MobileOne": ("mobileone-s0",),
    "RepVGG": ("repvgg-a0",),
    "EdgeNeXt": ("edgenext-xxs",),
    "DenseNet": ("densenet121",),
    "EfficientNet": ("efficientnet_b0",),
    "Inception": ("googlenet",),
    "Swin Transformer": ("swin_t",),
    "ViT": ("vit_b_32",),
    "MixNet": ("mixnet-s",),
    "ReXNet": ("rexnet",),
    "DLA": ("dla34",),
    "HRNet": ("hrnet-small",),
    "PP-LCNet": ("lcnet",),
    "DPN": ("dpn68",),
    "EfficientFormer": ("efficientformer",),
    "MobileViT": ("mobilevit-xs",),
    "PoolFormer": ("poolformer",),
    "VoVNet": ("vovnet",),
    "FastViT": ("fastvit",),
    "LeViT": ("levit",),
    "CoaT": ("coat-lite",),
    "FasterNet": ("fasternet-t0",),
    "PVT": ("pvtv2-b0",),
    "GERNet": ("gernet-s",),
    "FBNet": ("fbnetc",),
    "HardCoRe-NAS": ("hardcorenas-a",),
}


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(work, previous, output, plans):
    release = work / "release"
    suite = load(work / "suite.json")
    gold = load(work / "onnx-quality.json")
    records = load(work / "fixtures/groups.json")["imagenette-extra"]
    files = [r["file"] for r in records]
    assert len(set(files)) == len(files) == 100
    with previous.open(encoding="utf-8-sig") as f:
        old = {r["id"]: r for r in csv.DictReader(f) if r["task"] == "图像分类"}
    assert sorted(k for ids in HISTORICAL.values() for k in ids) == sorted(old)
    assert not set(HISTORICAL).intersection(v[0] for v in FAMILIES.values())
    cache = {}
    results = work / "results"
    for plan in plans:
        ledger = load(results / (plan.name + ".audit.json"))
        assert (
            ledger["firmware"] == "V1.3.2"
            and ledger["before"]["running"] == ledger["after"]["running"]
        )
        for tag, ids in load(plan).items():
            if tag in ledger["failures"]:
                continue
            raw = load(results / (tag + ".json"))
            audit = load(results / (tag + "-audit.json"))
            assert len(audit["samples"]) >= 2 and all(
                s["running"] == ["npu-benchmark"] for s in audit["samples"][:-1]
            )
            assert len(raw["idle"]) >= 3 and all(
                re.findall(r"(\d+)%", v) == ["0"] for v in raw["idle"][-3:]
            )
            assert set(ids) == {r["id"] for r in raw["tasks"]}
            for m in raw["tasks"]:
                key = m["id"]
                if m["status"] != "ok" or m["sha256"] != suite["models"][key]["sha256"]:
                    continue
                assert (
                    m["frames"]
                    == m["unique"]
                    == len(m["times"])
                    == len(m["cleanup_ms"])
                    == 100
                    and m["warmup"] == 5
                )
                assert [r["file"] for r in m["rows"]] == files
                assert all(
                    math.isclose(statistics.mean(t[i] for t in m["times"]), v)
                    for i, v in enumerate(m["ms"])
                )
                assert math.isclose(m["fps"], 1000 / sum(m["ms"]))
                assert math.isclose(
                    m["fps_with_cleanup"],
                    1000 / (sum(m["ms"]) + statistics.mean(m["cleanup_ms"])),
                )
                assert m["checks"] and all(
                    str(raw["pid"]) in c["owners"]
                    and all(
                        p == str(raw["pid"]) or n == "rkipc"
                        for p, n in c["owners"].items()
                    )
                    for c in m["checks"]
                )
                for metric in ("Top-1", "Top-5"):
                    assert math.isclose(
                        m["metrics"][metric],
                        statistics.mean(r["quality"][metric][0] for r in m["rows"]),
                    )
                cache[key] = (m, tag)
    assert set(cache) == set(suite["models"]), sorted(
        set(suite["models"]) - cache.keys()
    )
    rows = []
    for key, (model, checkpoint, side, design) in FAMILIES.items():
        if key not in cache:
            continue
        m, tag = cache[key]
        path = work / "compiled" / (key + ".rknn")
        assert sha(path) == m["sha256"] and math.isfinite(m["onnx_cosine"])
        pre, infer, post = m["ms"]
        tr = suite["models"][key]["transform"]
        rows.append(
            dict(
                model=model,
                representative=checkpoint,
                id=key,
                design=design,
                input=f"{side}×{side}",
                precision="FP16",
                mib=path.stat().st_size / 2**20,
                samples=100,
                warmup=5,
                pre_ms=pre,
                infer_ms=infer,
                infer_p95_ms=m["p95_infer_ms"],
                post_ms=post,
                pipeline_fps=m["fps"],
                with_cleanup_fps=m["fps_with_cleanup"],
                top1=m["metrics"]["Top-1"],
                top5=m["metrics"]["Top-5"],
                onnx_top1=gold[key]["Top-1"],
                onnx_top5=gold[key]["Top-5"],
                onnx_cosine=m["onnx_cosine"],
                resize_short=tr["resize"][0],
                mean=json.dumps(tr["mean"]),
                std=json.dumps(tr["std"]),
                source=tr["url"],
                sha256=m["sha256"],
                run=tag + ".json",
            )
        )
    rows.sort(key=lambda r: r["infer_ms"])
    write_csv(release / "RV1126B-architectures.csv", rows)
    index = []
    for model, ids in HISTORICAL.items():
        r = old[ids[0]]
        index.append(
            dict(
                model=model,
                representative=r["name"],
                id=r["id"],
                sample_count=r["unique"],
                precision=r["precision"],
                input=r["size"],
                infer_ms=r["infer_ms"],
                top1=json.loads(r["quality"])["Top-1"],
                series=r["series"],
                merged_ids=",".join(ids[1:]),
            )
        )
    for r in rows:
        index.append(
            dict(
                model=r["model"],
                representative=r["representative"],
                id=r["id"],
                sample_count=r["samples"],
                precision=r["precision"],
                input=r["input"],
                infer_ms=r["infer_ms"],
                top1=r["top1"],
                series="architectures",
                merged_ids="",
            )
        )
    write_csv(release / "RV1126B-ImageNet-model-index.csv", index)
    checks = [c for m, _ in cache.values() for c in m["checks"]]
    doc = [
        "# RV1126B 模型架构基准",
        "",
        f"reCamera Pro，固件 V1.3.2。本轮 {len(rows)} 个新增视觉分类模型，每项对应一个预训练权重。历史 ImageNet 分类记录共 {len(old)} 个配置，按下述口径归为 {len(HISTORICAL)} 个代表模型，合计 {len(index)} 个代表模型。历史多任务总表的 254 是权重和任务配置数。",
        "",
        "## 相同测试集上的板端结果",
        "",
        "100 张 Imagenette 验证图，每类 10 张，按原始 ImageNet 1000 类排序。全部采用 batch 1、RV1126B FP16 RKNN，每模型预热 5 次，再依次测量相同的 100 张图片。按同步推理均值升序排列。",
        "",
        "| 代表模型 | 输入 | RKNN MiB | 前处理 ms | 推理 ms | 推理 P95 ms | 后处理 ms | 三段 FPS | 含回收 FPS | Top-1 | Top-5 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        doc.append(
            "| {model} | {input} | {mib:.1f} | {pre_ms:.2f} | {infer_ms:.2f} | {infer_p95_ms:.2f} | {post_ms:.2f} | {pipeline_fps:.2f} | {with_cleanup_fps:.2f} | {top1:.0%} | {top5:.0%} |".format(
                **r
            )
        )
    doc += [
        "",
        "## 架构与主机对照",
        "",
        "| 代表模型 | 权重 / 实现 | 主要结构 | ONNX Top-1 / Top-5 | 板端与 ONNX 抽样余弦 |",
        "|---|---|---|---:|---:|",
    ]
    for r in rows:
        doc.append(
            f"| {r['model']} | [{r['representative']}]({r['source']}) | {r['design']} | {r['onnx_top1']:.0%} / {r['onnx_top5']:.0%} | {r['onnx_cosine']:.5f} |"
        )
    doc += [
        "",
        "## 去重与测量口径",
        "",
        "架构以主干的网络拓扑和核心计算模块分组。同一模型变更深度、宽度、分辨率、下游任务、分类头或权重只占一行。历史表中保守地将 ResNeXt、SK-ResNet、ResNeSt 归入 ResNet，将不同代的 MobileNet 归入 MobileNet。应用默认仅运行已安装的新增模型代表，历史数据可按模型 ID 查询。历史索引按已测 ImageNet 分类骨干建立，保留原始样本量和轮次。",
        "",
        "前处理包括按权重设置缩放短边、中心裁剪、BGR 转 RGB、均值方差归一化及输入内存布局转换。短边和归一化参数逐模型保存在 CSV。推理包含输入提交至同步取回输出，后处理对 1000 个类别分数排序取前五。三段 FPS 按三段均值之和取倒数；含回收 FPS 另计 Python 应用每帧释放输出及垃圾回收。采集、读盘、图像解码和标注计分均在计时外。准确率仅适用于本次 100 图子集。",
        "",
        f"有效结果来自 {len(set(tag for _, tag in cache.values()))} 份真机分组记录。每组通过 AppMgr 取得 NPU 独占租约，连续三次观察到零负载后预热。控制器轮询运行应用，共记录 {len(checks)} 次模型内设备持有者检查，只观察到基准进程及固件 broker rkipc。记录到的 NPU 频率为 {min(c['freq_hz'] for c in checks) / 1e6:g}–{max(c['freq_hz'] for c in checks) / 1e6:g} MHz，温度为 {min(c['temperature_mC'] for c in checks) / 1000:.1f}–{max(c['temperature_mC'] for c in checks) / 1000:.1f} °C。原始 ONNX 在相同的全部输入上评分，余弦为三个输入的抽样张量对照。",
        "",
    ]
    notes = work / "conversion-notes.md"
    if notes.is_file():
        doc += [notes.read_text().strip(), ""]
    output.write_text("\n".join(doc))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "previous", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("plans", nargs="+", type=Path)
    render(**vars(p.parse_args()))
