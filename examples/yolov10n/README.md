# YOLOv10n

模型 ID 为 `yolov10n`，任务为目标检测。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 640×640 | FP16 | 缩放、BGR 转 RGB、除以 255 | DFL 边距解码，one-to-one 分数筛选 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py yolov10n work/yolov10n
examples/yolov10n/demo.sh --onnx work/yolov10n/ready/yolov10n.onnx
```

### 使用已有 RKNN

```bash
examples/yolov10n/demo.sh --weights /path/to/yolov10n.rknn
```

实测 RKNN SHA-256 为 `0f82103d4321a344bf1ee2f27b842d246bf87b5acf2664834514857d581b67cb`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-yolov10n-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-yolov10n/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/extended/detectors.json) · [模型来源](https://github.com/airockchip/rknn_model_zoo/tree/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yolov10)
