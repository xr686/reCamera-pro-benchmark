# PP-HumanSeg

模型 ID 为 `human_segmentation_pphumanseg_2023mar`，任务为人像分割。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 192×192 | FP16 | RGB 映射到 ±1 | 逐像素 argmax |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py human_segmentation_pphumanseg_2023mar work/human_segmentation_pphumanseg_2023mar
examples/human_segmentation_pphumanseg_2023mar/demo.sh --onnx work/human_segmentation_pphumanseg_2023mar/ready/human_segmentation_pphumanseg_2023mar.onnx
```

### 使用已有 RKNN

```bash
examples/human_segmentation_pphumanseg_2023mar/demo.sh --weights /path/to/human_segmentation_pphumanseg_2023mar.rknn
```

实测 RKNN SHA-256 为 `f4315a920c91cb5a4b36adb1919ccfcb4ffc99e977f0552f888cd0430aa4c2b8`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-human-segmentation-pphumanseg-2023mar-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-human-segmentation-pphumanseg-2023mar/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-1.json) · [模型来源](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx)
