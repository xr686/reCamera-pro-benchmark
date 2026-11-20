# LaMa

模型 ID 为 `inpainting_lama_2025jan`，任务为图像修复。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 512×512 + 遮罩 | FP16 | BGR / 255，二值遮罩 | 像素转 HWC 并裁到 0–255 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py inpainting_lama_2025jan work/inpainting_lama_2025jan
examples/inpainting_lama_2025jan/demo.sh --onnx work/inpainting_lama_2025jan/ready/inpainting_lama_2025jan.onnx
```

本模型调用 [fold_weights.py](../../apps/npu-benchmark/tools/fold_weights.py)。

### 使用已有 RKNN

```bash
examples/inpainting_lama_2025jan/demo.sh --weights /path/to/inpainting_lama_2025jan.rknn
```

实测 RKNN SHA-256 为 `462cb1715f636d1c509a6b6841b649f6bcf7fa5ce8731b329afc7b9e9c0d3355`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

第一个文件为原图，第二个为灰度洞区掩膜，非零值标记待修复区域。 在上面的打包命令末尾加上 `--input photo.jpg --second mask.png` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-inpainting-lama-2025jan-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-inpainting-lama-2025jan/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-3.json) · [模型来源](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/inpainting_lama/inpainting_lama_2025jan.onnx)
