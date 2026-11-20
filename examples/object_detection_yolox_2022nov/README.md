# YOLOX-s

模型 ID 为 `object_detection_yolox_2022nov`，任务为目标检测。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 640×640 | FP16 | RGB 0–255 | 网格与步长解码、分类 NMS |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py object_detection_yolox_2022nov work/object_detection_yolox_2022nov
examples/object_detection_yolox_2022nov/demo.sh --onnx work/object_detection_yolox_2022nov/ready/object_detection_yolox_2022nov.onnx
```

### 使用已有 RKNN

```bash
examples/object_detection_yolox_2022nov/demo.sh --weights /path/to/object_detection_yolox_2022nov.rknn
```

实测 RKNN SHA-256 为 `8f3fa58bc9bfdba45729df14d05c146432417c98f18642fac9ddf3447e65d0ee`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-object-detection-yolox-2022nov-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-object-detection-yolox-2022nov/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-1.json) · [模型来源](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_detection_yolox/object_detection_yolox_2022nov.onnx)
