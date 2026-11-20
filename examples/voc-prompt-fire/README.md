# Pretrained SqueezeNet pyramid · voc-prompt

模型 ID 为 `voc-prompt-fire`，任务为提示框内单实例分割。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 96×96 | FP16 | 缩放、BGR 转 RGB、除以 255 | 按输出 logit 的零阈值生成前景掩膜 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py voc-prompt-fire work/voc-prompt-fire --voc /path/to/VOCdevkit/VOC2012
examples/voc-prompt-fire/demo.sh --onnx work/voc-prompt-fire/ready/voc-prompt-fire.onnx
```

本模型调用 [train_sequence_objects.py](../../apps/npu-benchmark/tools/train_sequence_objects.py) → [lower_pyramid_resize.py](../../apps/npu-benchmark/tools/lower_pyramid_resize.py)。

### 使用已有 RKNN

```bash
examples/voc-prompt-fire/demo.sh --weights /path/to/voc-prompt-fire.rknn
```

实测 RKNN SHA-256 为 `c2c13d7974b517cf577678f6416638063d413eba27d5ab7acb3dcc04ae8d1632`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-voc-prompt-fire-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-voc-prompt-fire/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/sequence/fixed-masks.json)
