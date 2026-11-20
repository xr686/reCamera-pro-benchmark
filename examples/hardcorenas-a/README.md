# HardCoRe-NAS-A

模型 ID 为 `hardcorenas-a`，任务为图像分类。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 224×224 | FP16 | 按预训练配置缩放和中心裁剪，RGB 标准化 | 类别分数排序 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

下载已在 RV1126B 实测的权重后直接打包。

```bash
git lfs pull --include="models/app-architectures/hardcorenas-a.rknn"
examples/hardcorenas-a/demo.sh
```

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py hardcorenas-a work/hardcorenas-a
examples/hardcorenas-a/demo.sh --onnx work/hardcorenas-a/ready/hardcorenas-a.onnx
```

本模型调用 [build_sequence_backbones.py](../../apps/npu-benchmark/tools/build_sequence_backbones.py)。

### 使用已有 RKNN

```bash
examples/hardcorenas-a/demo.sh --weights /path/to/hardcorenas-a.rknn
```

实测 RKNN SHA-256 为 `003e8c3035de6cd5404015c806713a590050158a2f06ea6e98943861d577aa58`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-hardcorenas-a-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-hardcorenas-a/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/sequence/class-b.json) · [模型来源](https://huggingface.co/timm/hardcorenas_a.miil_green_in1k)
