# BlazePalm

模型 ID 为 `palm_detection_mediapipe_2023feb`，任务为手掌检测。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 192×192 | FP16 | RGB / 255 | 先验框解码、NMS |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py palm_detection_mediapipe_2023feb work/palm_detection_mediapipe_2023feb
examples/palm_detection_mediapipe_2023feb/demo.sh --onnx work/palm_detection_mediapipe_2023feb/ready/palm_detection_mediapipe_2023feb.onnx
```

本模型调用 [mediapipe_inputs.py](../../apps/npu-benchmark/tools/mediapipe_inputs.py)。

### 使用已有 RKNN

```bash
examples/palm_detection_mediapipe_2023feb/demo.sh --weights /path/to/palm_detection_mediapipe_2023feb.rknn
```

实测 RKNN SHA-256 为 `05df76ed8d5803214f14f09e9edfd4cfda5d713a14b9b36ab90cce8c2bf23a8f`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-palm-detection-mediapipe-2023feb-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-palm-detection-mediapipe-2023feb/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-1.json) · [模型来源](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx)
