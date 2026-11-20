# YOLOv5s-seg

模型 ID 为 `yolov5s-seg`，任务为实例分割。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 640×640 | FP16 | RGB / 255 | 框解码、NMS、原型合成、逐框裁剪掩膜 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py yolov5s-seg work/yolov5s-seg
examples/yolov5s-seg/demo.sh --onnx work/yolov5s-seg/ready/yolov5s-seg.onnx
```

### 使用已有 RKNN

```bash
examples/yolov5s-seg/demo.sh --weights /path/to/yolov5s-seg.rknn
```

实测 RKNN SHA-256 为 `f068e9b47832440ee2a431f45dfc1264bc8efcc416f3e011e7dd70c78b4f68de`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-yolov5s-seg-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-yolov5s-seg/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-1.json) · [模型来源](https://github.com/airockchip/rknn_model_zoo/tree/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yolov5_seg)
