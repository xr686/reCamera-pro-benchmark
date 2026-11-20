"""Render comparable device readings from complete App Center runs in argument order."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_models import MODELS

PRE = {
    "nanodet": "RGB，按通道标准化",
    "rgb": "RGB 0–255",
    "rgb01": "RGB / 255",
    "rgb11": "RGB 映射到 ±1",
    "bgr11": "BGR 映射到 ±1",
    "retina": "RGB 减均值",
    "bgr": "BGR 0–255",
    "bgr8": "BGR uint8",
    "rgb8": "RGB uint8",
    "imagenet": "RGB，ImageNet 标准化",
    "db": "BGR 按通道标准化",
    "track": "模板与搜索区 BGR 逐通道标准化",
    "gray11": "转灰度，映射到 ±1",
    "y01": "取亮度 Y / 255",
    "pcm": "16 kHz 单声道波形",
    "wave_norm": "16 kHz 波形，逐段标准化、补零",
    "fbank": "16 kHz 波形提取 80 维 log-Mel，逐段标准化、补零",
    "edge": "BGR 减均值",
    "inpaint": "BGR / 255，二值遮罩",
}
POST = {
    "nanodet": "距离分布解码、分类 NMS",
    "yolox": "网格与步长解码、分类 NMS",
    "gghl": "四角回归、多边形 NMS",
    "instances": "框解码、NMS、原型合成、逐框裁剪掩膜",
    "semantic": "逐像素 argmax",
    "depth": "提取单通道相对深度",
    "palm": "先验框解码、NMS",
    "person": "分数 sigmoid、NMS、髋中心回归",
    "pose": "回归点坐标归一化",
    "hand": "21 点坐标和在场分数",
    "track": "响应图乘余弦窗、中心偏移与尺度回归",
    "embedding": "特征向量 L2 归一化",
    "yunet": "多尺度得分融合、框回归、NMS",
    "retina": "先验框回归、NMS",
    "scrfd": "距离回归、NMS",
    "class": "类别 argmax",
    "scalar": "提取质量标量",
    "mesh": "提取 468 个关键点的平面坐标",
    "platebox": "四角偏移回归、得分融合、NMS",
    "plate": "CTC 贪心解码、去重去空白",
    "word": "英文 CTC 贪心解码",
    "textboxes": "DB 概率图阈值、轮廓转四边形",
    "sr": "亮度图裁到 [0,1]",
    "sound": "按时间平均 521 类分数、取首选类别",
    "edge": "末级边缘图 sigmoid 和归一化",
    "inpaint": "像素转 HWC 并裁到 0–255",
    "restore": "转 BGR、缩放并裁到 0–255",
    "speech": "字符级 CTC 贪心解码",
    "subword": "子词 CTC 解码、合并单词",
}
SECTIONS = {
    "分类、检测、分割及深度": "目标检测 旋转目标检测 实例分割 人像分割 场景语义分割 单目相对深度",
    "人体、手势、跟踪与检索": "人体检测 人体姿态 手掌检测 手部关键点 单目标跟踪 行人重识别",
    "人脸": "人脸检测 人脸特征 表情识别 人脸质量 人脸关键点 人脸防伪",
    "文字与车牌": "车牌四角检测 车牌字符识别 文字定位 文字识别",
    "图像增强与音频": "图像超分辨率 边缘检测 图像修复 运动去模糊 声音事件分类 语音转文字",
}


def shape(model):
    if model.decoder == "track":
        return "128×128 + 256×256"
    if model.decoder == "inpaint":
        return "512×512 + 遮罩"
    if model.pixels == "fbank":
        return "80×600（6 s）"
    if len(model.size) == 1:
        return f"{model.size[0]}（{model.size[0] // 16000} s）"
    return "×".join(map(str, model.size))


def score(scores):
    items = []
    for key, value in scores.items():
        numeric = "dB" in key or "Pearson" in key
        items.append(f"{key} {value:.3f}" if numeric else f"{key} {value:.2%}")
    return "；".join(items) if items else "未评估"


def render(args):
    by_id = {m.file: m for m in MODELS}
    data = json.loads((args.assets / "fixtures/groups.json").read_text())
    result = {}
    classifiers = {}
    runs = []
    for report in args.report:
        d = json.loads(report.read_text())
        result.update((r["id"], r) for r in d.get("tasks", []))
        classifiers.update((r["id"], r) for r in d["models"])
        runs.append(d)
    ok = [r for r in result.values() if r["status"] == "ok"]
    n = len(ok) + len(classifiers)
    categories = {r["task"] for r in ok} | ({"图像分类"} if classifiers else set())
    lines = [
        "# RV1126B 多任务模型真机基准",
        "",
        f"2026-11-19，在 reCamera Pro（RV1126B，2 GB，固件 V1.3.1）上取得 **{n} 个模型配置、{len(categories)} 项任务**的有效结果，包含 {len(ok)} 个扩展配置及 {len(classifiers)} 个分类器。PP-OCRv3 的中英文检测权重分别计数，MiniFASNet 的两个变体也分别计数。",
        "",
        "对照仓库所列的图像分类、普通框检测、旋转框检测、实例分割、姿态、深度、文字定位、文字识别和语音识别均有对应实测。此处额外覆盖手势、人脸、行人检索、跟踪、图像增强及环境声等。参考仓库使用了 YOLOv8/11/26、PP-OCRv4 和 Zipformer，本次选取的网络或版本均不在该清单中。[参考仓库固定版本](https://github.com/xr686/reCamera-pro-benchmark/tree/11929a7582eb58bb025727c818e0eb50c4318eae)",
        "",
        "## 测量口径",
        "",
        "下表尺寸为宽×高，batch 为 1。权重为 RV1126B FP16 RKNN，包括从已量化源权重折叠后转 FP16 的网络。每项按实际应用执行一次前处理、一次同步 RKNN 调用和一次后处理。同步推理包含运行时输入输出搬运和等待，无法从该数字单独读出纯 NPU 核心耗时。耗时均值及 P95 的单位都是 ms。吞吐量 = 1000 / 三段平均耗时之和。图片单位为帧/秒，声音单位为段/秒。文件读取、图片解码、预先生成 ROI、标注比对、绘图、模型初始化和报告写盘均在计时之外。扩展模型还逐帧释放输出并回收 Python/ctypes 缓冲区。下方配置表单列“含回收吞吐”以显示这一运行成本。",
        "",
        "样本/次数分别是独立标注输入数和计时调用数。分类预热 20 次，扩展常规模型预热 5 次、至少计时 30 次。GGHL、DexiNed、图像修复、去模糊及 ASR 预热 2 次、至少计时 4 次。质量对每个独立输入计一次，同一图片多次跑分只影响时延统计。小样本 P95 对单个慢帧敏感。",
        "",
    ]
    header = [
        "| 任务 / 模型 | 输入 | RKNN MiB | 样本/次数 | 前处理 | 同步推理 | 推理 P95 | 后处理 | 帧或段/s | 任务质量 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, (section, tasks) in enumerate(SECTIONS.items()):
        lines.extend(["## " + section, "", *header])
        if i == 0:
            for r in classifiers.values():
                s = r["ms"]
                size = (
                    args.app / "models" / (r["id"] + ".rknn")
                ).stat().st_size / 2**20
                lines.append(
                    f"| 图像分类 / {r['name']} | 224×224 | {size:.2f} | {len(r['samples'])}/{len(r['samples'])} | {s['pre']['mean']:.2f} | {s['infer']['mean']:.2f} | {s['infer']['p95']:.2f} | {s['post']['mean']:.2f} | {r['fps']:.2f} | Top-1 {r['top1']:.2%}，Top-5 {r['top5']:.2%} |"
                )
        for r in ok:
            if r["task"] not in tasks.split():
                continue
            m = by_id[r["id"]]
            size = (
                args.assets / "compiled" / (m.file + ".rknn")
            ).stat().st_size / 2**20
            pre, infer, post = r["ms"]
            lines.append(
                f"| {m.task} / {m.name} | {shape(m)} | {size:.2f} | {r['unique']}/{r['frames']} | {pre:.2f} | {infer:.2f} | {r['p95_infer_ms']:.2f} | {post:.2f} | {r['fps']:.2f} | {score(r['metrics'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 音频实时率",
            "",
            "RTF = 全部计时段的前处理、推理和后处理耗时之和 / 这些段对应的有效音频时长之和，越小越快。ASR 输入补齐到固定 6 秒。另列按固定窗计算的 RTF 便于复核。",
            "",
            "| 模型 | 有效音频秒（独立样本） | 有效音频 RTF | 固定窗 RTF |",
            "|---|---:|---:|---:|",
        ]
    )
    for r in ok:
        m = by_id[r["id"]]
        if m.pixels not in ("pcm", "fbank", "wave_norm"):
            continue
        secs = [min(m.size[0] / 16000, s["seconds"]) for s in data[m.corpus]]
        total = sum(secs[i % len(secs)] for i in range(r["frames"]))
        elapsed = np.sum(r["times"]) / 1000
        lines.append(
            f"| {m.name} | {sum(secs):.2f} | {elapsed / total:.4f} | {elapsed / (r['frames'] * m.size[0] / 16000):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 模型 ID、前后处理与转换核对",
            "",
            "分类模型采用短边缩放、224 中心裁剪、RGB uint8、图内归一化和 1000 类排序。其三轮重复性及原始权重核对见[分类专项报告](BENCHMARK.md)。以下各网络默认将图像缩放到表中尺寸，姿态、车牌识别、表情和特征提取的输入由标注或测试图片提供。业务上组合检测、裁剪和识别所需的时间应另外计算。",
            "",
            "cos 为主机 CPU ONNX 与板端 RKNN 在最多 3 个相同输入上，各张量均匀抽取最多 512 个位置后的余弦相似度均值。它衡量转换时的数值一致性。仅有预编译 RKNN 的模型以横线表示。",
            "",
            "| 配置 ID | 前处理 | 后处理 | cos | 含回收吞吐（帧或段/s） |",
            "|---|---|---|---:|---:|",
        ]
    )
    for r in classifiers.values():
        lines.append(
            f"| `{r['id']}` | 短边缩放、中心裁剪、RGB | 排序取前五类 | 见分类报告 | 默认自动回收 |"
        )
    for r in ok:
        m = by_id[r["id"]]
        cos = r.get("onnx_cosine")
        value = f"{cos:.5f}" if cos is not None else "—"
        rate = f"{r['fps_with_cleanup']:.2f}" if "fps_with_cleanup" in r else "—"
        lines.append(
            f"| `{m.file}` | {PRE[m.pixels]} | {POST[m.decoder]} | {value} | {rate} |"
        )
    lines.extend(["", args.notes.read_text().strip(), "", "## 设备状态和原始证据", ""])
    allchecks = [
        c
        for d in runs
        for r in d["models"] + d.get("tasks", [])
        for c in r.get("checks", [])
    ]
    freq = [c["freq_hz"] / 1e6 for c in allchecks]
    temp = [c["temperature_mC"] / 1000 for c in allchecks]
    lines.append(
        f"Linux {runs[0]['kernel']}，驱动 {runs[0]['npu_driver']}，板端 RKNN Runtime 2.3.2，OpenCV {runs[0]['opencv']}，调频策略 {runs[0]['governor']}。各来源轮次共记录 {len(allchecks)} 次模型内租约及句柄检查。检查时的频率为 {min(freq):g}–{max(freq):g} MHz，thermal_zone0 为 {min(temp):.1f}–{max(temp):.1f} °C。"
    )
    lines.extend(
        [
            "",
            "| 板端记录 | 开始时间 UTC | 计时前空闲读数 | 进程 PID |",
            "|---|---|---|---:|",
        ]
    )
    for p, d in zip(args.report, runs):
        lines.append(
            f"| {p.name} | {d['utc']} | {', '.join(d['idle'][-3:])} | {d['pid']} |"
        )
    lines.extend(
        [
            "",
            "审计包含 AppMgr 的运行应用采样、停止和恢复结果。基准持有固件独占租约后开始测量，检查 /dev/rknpu 只允许本进程及固件 broker 进程 rkipc。每次只加载一个模型。原始 JSON 保存逐次时延、预测或质量计数、硬件读数以及模型 SHA-256。测试应用和配套资源的安装说明见[应用说明](README.md)。",
            "",
        ]
    )
    args.out.write_text("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("app", "assets", "notes", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("report", nargs="+", type=Path)
    render(p.parse_args())
