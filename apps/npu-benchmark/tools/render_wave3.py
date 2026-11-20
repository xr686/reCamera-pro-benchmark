"""Validate completed board measurements and assemble the cumulative V1.3.2 results."""

import argparse, csv, json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extended_models import contracts
from task_models import Model
from field_models import contracts as field_contracts
from sequence_models import contracts as sequence_contracts

PRE = {
    "signal": "按训练集通道统计量标准化，转换为 FP32 张量",
    "cropnet": "按预训练配置缩放和中心裁剪，RGB 标准化",
    "stereo-pair": "双灰度帧堆叠、归一化",
    "std-patch": "灰度块按各自均值和方差标准化",
    "hashed-text": "单词与双词组计数，CRC32 映射 512 维",
    "tabular": "按训练集逐列均值和标准差归一化",
    "bayer4": "RGGB 四相位拆分，除以 255",
    "rgb01": "缩放、BGR 转 RGB、除以 255",
    "gray01": "灰度化并除以 255",
    "short-wave": "8 kHz，1024 个单声道采样点转 FP32",
    "logmel": "16 kHz 单声道，40×98 log-Mel",
}
POST = {
    "prompt-mask": "按输出 logit 的零阈值生成前景掩膜",
    "dominant-box": "截断到单位区间并按 xyxy 排序",
    "extra-class": "类别分数排序",
    "disparity-map": "32 个视差代价逐像素 argmin",
    "patch-embedding": "128 维描述子 L2 归一化",
    "binary-hash": "逐位符号阈值、映射 ±1 和 L2 归一化",
    "multi-label": "输出全部 20 个类别分数",
    "flow-cost": "对 49 个位移代价逐像素取最小值",
    "flow-vector": "转为水平和垂直位移图",
    "pitch": "选取延迟峰并作抛物线插值，换算 Hz",
    "illumination": "取 RGB 光源三分量",
    "spam": "两类分数 argmax",
    "regression": "读取物理量标量",
    "yolov5-relu": "锚框解码，按类别 NMS",
    "yolov10n": "DFL 边距解码，one-to-one 分数筛选",
    "photometric": "裁剪到 8 位像素、还原通道序",
    "anomaly-score": "异常图最高 1% 网格的均值",
    "embedding": "说话人向量 L2 归一化",
}


def pretty(d):
    def show(k, v):
        if "MAE" in k or "dB" in k or "EPE" in k or "°" in k:
            return f"{k} {v:.3f}"
        return f"{k} {v:.2%}"

    return "，".join(show(k, v) for k, v in d.items())


def render(args):
    work = args.work
    release = work / "release"
    files = work / "results"
    catalog = field_contracts(Model) if args.field else contracts(Model)
    if args.sequence:
        catalog = sequence_contracts(Model)
    models = {m.file: m for m in catalog}
    suite = json.loads((work / "suite.json").read_text())
    targets = suite["models"]
    cpu = json.loads((work / "onnx-quality.json").read_text())
    with args.previous.open(encoding="utf-8-sig") as f:
        prior = list(csv.DictReader(f))
    assert set(targets) == set(models) and not (set(targets) & {r["id"] for r in prior})
    seen = {}
    audits = []
    temperatures = []
    freq = []
    used_reports = set()
    skipped = []
    finalstates = {}
    for plan_path in args.plans:
        plan = json.loads(plan_path.read_text())
        ledger = json.loads((files / (plan_path.name + ".audit.json")).read_text())
        assert ledger["firmware"] == "V1.3.2"
        finalstates[plan_path.name] = ledger["after"]["running"]
        assert set(ledger["before"]["running"]) - {"builtin"} == set(
            ledger["after"]["running"]
        ) - {"builtin"}
        for tag, ids in plan.items():
            if tag in ledger["failures"]:
                skipped.append(tag)
                continue
            report = json.loads((files / (tag + ".json")).read_text())
            audit = json.loads((files / (tag + "-audit.json")).read_text())["samples"]
            assert len(audit) >= 2 and all(
                r["running"] == ["npu-benchmark"] for r in audit[:-1]
            ), tag
            assert len(report["idle"]) >= 3 and all(
                z.split(":")[-1].strip() == "0%" for z in report["idle"][-3:]
            ), tag
            tested = report["tasks"]
            assert set(ids) == {r["id"] for r in tested}
            for r in tested:
                k = r["id"]
                if r["status"] != "ok" or r.get("sha256") != targets[k]["sha256"]:
                    continue
                assert r["checks"] and all(
                    set(c["owners"])
                    and all(
                        pid == str(report["pid"]) or name == "rkipc"
                        for pid, name in c["owners"].items()
                    )
                    for c in r["checks"]
                )
                m = models[k]
                pre, infer, post = r["ms"]
                assert r["frames"] >= r["unique"] > 0 and math.isclose(
                    1000 / sum(r["ms"]), r["fps"]
                )
                size = "×".join(map(str, m.size))
                if m.pixels == "logmel":
                    size = "1 s → 40×98"
                if m.pixels == "hashed-text":
                    size = "512 维"
                if m.pixels == "tabular":
                    size = f"{m.size[0]} 维"
                if m.pixels == "stereo-pair":
                    size += " × 2"
                if m.pixels == "bayer4":
                    size = "96×96 → 4×48×48"
                if m.pixels == "short-wave":
                    size = "1024 点（128 ms）"
                row = dict(
                    id=k,
                    name=m.name,
                    group="本轮扩展",
                    pre_ms=pre,
                    infer_ms=infer,
                    p95_ms=r["p95_infer_ms"],
                    post_ms=post,
                    fps=r["fps"],
                    with_gc=r["fps_with_cleanup"],
                    unique=r["unique"],
                    calls=r["frames"],
                    quality=json.dumps(r["metrics"], ensure_ascii=False),
                    precision="FP16",
                    size=size,
                    task=m.task,
                    mib=(work / "compiled" / (k + ".rknn")).stat().st_size / 2**20,
                    prep=PRE[m.pixels],
                    post=(
                        "哈希或描述子 L2 归一化"
                        if m.corpus == "copies" and m.decoder == "embedding"
                        else POST[m.decoder]
                    ),
                    cos=r["onnx_cosine"],
                    sha256=r["sha256"],
                    run=tag + ".json",
                    cpu_metrics=json.dumps(cpu[k], ensure_ascii=False),
                )
                seen[k] = (row, r["checks"], tag)
    missing = set(models) - seen.keys()
    if missing:
        raise ValueError(f"Missing valid results: {sorted(missing)}")
    rows = [seen[k][0] for k in models]
    for _, checks, tag in seen.values():
        audits.extend(checks)
        temperatures.extend(c["temperature_mC"] / 1000 for c in checks)
        freq.extend(c["freq_hz"] / 1e6 for c in checks)
        used_reports.add(tag)
    full = prior + rows
    fields = list(dict.fromkeys([*prior[0], "series"]))
    csv_name = "RV1126B-field-combined.csv" if args.field else "RV1126B-combined.csv"
    if args.sequence:
        csv_name = "RV1126B-sequence-combined.csv"
    with (release / csv_name).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for band, items in [
            ("previous-85", prior),
            ("sequence" if args.sequence else ("field" if args.field else "extended"), rows),
        ]:
            for r in items:
                writer.writerow({**r, "series": r.get("series", band)})
    tasks = lambda rs: {r["task"] for r in rs}
    doc = [
        "# RV1126B V1.3.2 第五批模型实测" if args.sequence else
        "# RV1126B V1.3.2 第四批模型实测"
        if args.field
        else "# RV1126B V1.3.2 扩展基准",
        "",
        f"设备为 reCamera Pro，RV1126B，固件 V1.3.2。本轮实测 **{len(rows)} 个新增配置**，涵盖 {len(tasks(rows))} 个任务标签，其中 {len(tasks(rows) - tasks(prior))} 个标签是首次测量。加上同固件上一轮的 {len(prior)} 个，累计 **{len(full)} 个配置、{len(tasks(full))} 个任务标签**。",
        "",
        "累计表保留此前 186 项同固件记录。新配置按实际部署的权重计数。同任务配置使用相同的测试样本，CSV 的 series、run 和 sha256 列可追溯每项结果。[前 186 项报告](FIELD-1.3.2.md) · [参考仓库清单](https://github.com/xr686/reCamera-pro-benchmark)" if args.sequence else
        "累计表保留前 131 项的原始板端记录。本轮每组同任务模型使用相同测试输入。分别拟合的权重、固定算子图各按一个配置计数。光流、颜色估计及感知哈希包含经典算法。CSV 的 series 和 run 列指向对应轮次和原始结果。[前 131 项报告](EXTENDED-1.3.2.md) · [参考仓库模型清单](https://github.com/xr686/reCamera-pro-benchmark/tree/11929a7582eb58bb025727c818e0eb50c4318eae)"
        if args.field
        else "模型配置按实际运行的权重及输入契约计数。同一算法在两个工业类别或多个回归数据集上分别训练并计数。双目组中的四项是用 RKNN 算子实现的经典匹配算法。原有 85 项保留其原始测量时间和数据范围。可按 CSV 中的 series 和 run 追溯记录。[前 85 项的逐模型数据](FIRMWARE-1.3.2.md) · [参考仓库模型清单](https://github.com/xr686/reCamera-pro-benchmark/tree/11929a7582eb58bb025727c818e0eb50c4318eae)",
        "",
        "所有新增配置均为 batch 1、RV1126B FP16 RKNN。同步推理从提交输入到取得输出，包含运行时搬运。三段吞吐量为 1000 除以前处理、同步推理和后处理的平均毫秒数之和。含回收吞吐量还计入逐次释放输出缓冲区的时间。摄像头采集、文件读取、标注评分与写盘在计时之外。图像对模型按对，其余按单图、图像块、音频段、文本或表格记录计。",
        "",
    ]
    for task in dict.fromkeys(r["task"] for r in rows):
        doc.extend(
            [
                "## " + task,
                "",
                "| 模型 | 输入 | MiB | 样本/计时 | 前处理 ms | 推理 ms | 推理 P95 ms | 后处理 ms | 三段/s | 含回收/s | 实测质量 |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for r in rows:
            if r["task"] == task:
                doc.append(
                    f"| {r['name']} | {r['size']} | {r['mib']:.2f} | {r['unique']}/{r['calls']} | {r['pre_ms']:.2f} | {r['infer_ms']:.2f} | {r['p95_ms']:.2f} | {r['post_ms']:.2f} | {r['fps']:.2f} | {r['with_gc']:.2f} | {pretty(json.loads(r['quality']))} |"
                )
        doc.append("")
    doc.extend(
        [
            "## 模型 ID、处理方式和转换对照",
            "",
            "cos 是同一输入上板端与主机 ONNX 输出的抽样余弦相似度。每个网络最多比对 3 个输入，每个张量最多 512 个元素。任务质量采用表中全部独立输入。回归网络的标量 cos 几乎只反映符号，应直接比较 MAE。",
            "",
            "| 模型 ID | 前处理 | 后处理 | cos | 主机 ONNX 样本质量 |",
            "|---|---|---|---:|---|",
        ]
    )
    for r in rows:
        doc.append(
            f"| `{r['id']}` | {r['prep']} | {r['post']} | {r['cos']:.5f} | {pretty(json.loads(r['cpu_metrics']))} |"
        )
    clock_mhz = (
        f"{min(freq):g}" if min(freq) == max(freq) else f"{min(freq):g}–{max(freq):g}"
    )
    doc.extend(
        [
            "",
            (
                Path(__file__).with_name(
                    "wave5-methods.md" if args.sequence else ("wave4-methods.md" if args.field else "wave3-methods.md")
                )
            )
            .read_text()
            .strip(),
            "",
            "## 真机独占和结果审计",
            "",
            f"最终结果来自 {len(used_reports)} 份有效真机分组记录。每组都取得独占 NPU 租约，连续 3 次零负载后开始，常规模型预热 5 次，每项至少计时 30 次。{len(audits)} 次设备文件持有者检查仅观察到测试进程与固件 broker rkipc。检查时 NPU 频率为 {clock_mhz} MHz，温度 {min(temperatures):.1f}–{max(temperatures):.1f} °C。AppMgr 同时每约 3 秒检查正在运行的应用。",
            "",
        ]
    )
    if skipped:
        doc.extend(
            [
                "控制器排除的轮次为 "
                + ", ".join(skipped)
                + "。这些轮次保存在原始证据的计划审计中，最终表格仅引用有效的独占复测。",
                "",
            ]
        )
    doc.extend(
        [
            f"各模型的计时、原始任务质量、模型摘要和源报告文件名收录在 {csv_name}。原始报告包含逐输入质量和每次独占检查。",
            "",
        ]
    )
    args.output.write_text("\n".join(doc))
    print(len(rows), len(full), len(tasks(full)), len(audits), finalstates)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "previous", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--field",
        action="store_true",
        help="Render the fourth pack and retain earlier CSV series",
    )
    p.add_argument("--sequence", action="store_true", help="Render the fifth pack")
    p.add_argument("plans", nargs="+", type=Path)
    render(p.parse_args())
