# RetinaFace MobileNet

模型 ID 为 `RetinaFace_mobile320`，任务为人脸检测。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 320×320 | FP16 | RGB 减均值 | 先验框回归、NMS |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py RetinaFace_mobile320 work/RetinaFace_mobile320
examples/RetinaFace_mobile320/demo.sh --onnx work/RetinaFace_mobile320/ready/RetinaFace_mobile320.onnx
```

### 使用已有 RKNN

```bash
examples/RetinaFace_mobile320/demo.sh --weights /path/to/RetinaFace_mobile320.rknn
```

实测 RKNN SHA-256 为 `21b2cf913f63679cf97fb9362295a08df9913311ba247fd4420aca49247e6a2a`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-retinaface-mobile320-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-retinaface-mobile320/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-2.json) · [模型来源](https://github.com/airockchip/rknn_model_zoo/tree/bad6c7334531becaf90a561988519b7bec34d0ab/examples/RetinaFace)
