"""Render model-specific instructions from the same contracts used by the packager."""

from collections import defaultdict
import json
from pathlib import Path

from build import HERE, index


def render():
    tasks = defaultdict(list)
    sources = index("classification-sources.csv")
    hashes = index("model-hashes.csv")
    for key, row in index("measurements.csv").items():
        folder = HERE / key
        launcher = folder / "demo.sh"
        if not launcher.is_symlink():
            launcher.symlink_to("../demo.sh")
        spec = json.loads((folder / "model.json").read_text())
        weight = sources.get(key, {}).get("rknn_in_repo")
        app_id = "demo-" + key.lower().replace("_", "-")
        kind = spec["input_kind"]
        inputs = {"image":"--input photo.jpg", "array":"--input sample.npy", "text":"--input message.txt",
                  "tokens":"--input token-ids.npy", "mask":"--input photo.jpg --second mask.png",
                  "pair":"--input template.jpg --second search.jpg"}[kind]
        help_text = {
            "image":"传入 PNG 或 JPEG 图片。", "text":"传入 UTF-8 文本文件。",
            "tokens":"传入 20 个整数 token ID 的 NPY 文件，包含起始、结束与填充位置。使用基准的 CLIP 词表。",
            "array":f"传入形状为 `{spec['sample_shape']}`、类型为 `{spec['sample_dtype']}` 的 NumPy 数组，以 NPY 格式保存。",
            "mask":"第一个文件为原图，第二个为灰度洞区掩膜，非零值标记待修复区域。",
            "pair":"第一个文件为目标模板，第二个为搜索区域。"
        }[kind]
        fields = row["post"]
        links = [f"[测量记录](../../benchmark_results/app/runs/{row['series']}/{row['run']})"]
        if spec.get("source"):
            links.append(f"[模型来源]({spec['source']})")
        title = row["name"]
        lines = [f"# {title}", "", f"模型 ID 为 `{key}`，任务为{row['task']}。", "",
                 "| 输入 | 实测格式 | 前处理 | 预测输出 |", "|---|---|---|---|",
                 f"| {row['size']} | {row['precision']} | {row['prep']} | {fields} |", "",
                 "## 构建应用", "", "在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。", ""]
        if weight:
            lines += ["下载已在 RV1126B 实测的权重后直接打包。", "", "```bash",
                      f'git lfs pull --include="{weight}"', f"examples/{key}/demo.sh", "```", ""]
        if spec.get("export"):
            params = spec["export"].get("requires",[])
            examples = {"imagenette":"imagenette2-160","voc":"VOCdevkit/VOC2012","pet_archive":"oxford-iiit-pet.tgz",
                        "commands":"mini_speech_commands.zip","esc_index":"esc50.csv","previous_fixtures":"fixtures"}
            flags = "".join(f" --{p.replace('_','-')} /path/to/{examples[p]}" for p in params)
            lines += ["### 从源模型构建", "", "导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。", "",
                      "```bash", f'python3 examples/export.py {key} work/{key}{flags}',
                      f'examples/{key}/demo.sh --onnx work/{key}/ready/{key}.onnx', "```", ""]
            steps = [x[0] for x in spec["export"]["steps"] if x[0] != "-m"]
            if steps:
                lines += ["本模型调用 " + " → ".join(f"[{n}](../../apps/npu-benchmark/tools/{n})" for n in steps) + "。", ""]
            if "conversion_note" in spec:
                lines += [spec["conversion_note"], ""]
        else:
            app = spec["supplier_app"]
            lines += [f"实测权重随 SDK 的 `{app}` 应用提供。准备含 `models/{key}.rknn` 的该应用安装包，然后运行：",
                      "", "```bash", f'examples/{key}/demo.sh --from-pack /path/to/{app}-arm64.tar.gz', "```", "",
                      f"[SDK 应用源码](https://github.com/Seeed-Studio/recamera-pro-ext-api/tree/main/apps/{app})", ""]
        lines += ["### 使用已有 RKNN", "", "```bash", f"examples/{key}/demo.sh --weights /path/to/{key}.rknn",
                  "```", "", f"实测 RKNN SHA-256 为 `{hashes[key]['rknn_sha256']}`。自行重编译的兼容权重可附加 `--rebuilt`。", "",
                  "## 运行示例", "", f"{help_text} 在上面的打包命令末尾加上 `{inputs}` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。", "",
                  f"在设备应用中心安装 `dist/{app_id}-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。", "",
                  "```bash", f'ssh "$DEVICE" cat /userdata/local/appdata/{app_id}/prediction.json', "```", "",
                  "较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。", "",
                  " · ".join(links), ""]
        (folder / "README.md").write_text("\n".join(lines))
        tasks[row["task"]].append((title,key))
    doc = ["# 模型示例索引", "", f"共 {sum(map(len,tasks.values()))} 个模型配置。点击模型名查看输入格式、获取权重、编译和安装命令。[运行指南](README.md)", ""]
    for task, members in sorted(tasks.items()):
        doc.extend([f"## {task}", ""])
        doc.extend(f"- [{name}]({key}/README.md) · `{key}`" for name,key in members)
        doc.append("")
    (HERE / "MODELS.md").write_text("\n".join(doc))


if __name__ == "__main__":
    render()
