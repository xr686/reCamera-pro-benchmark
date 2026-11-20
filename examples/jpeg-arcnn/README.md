# AR-CNN

模型 ID 为 `jpeg-arcnn`，任务为JPEG 去压缩伪影。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 96×96 | FP16 | 缩放、BGR 转 RGB、除以 255 | 裁剪到 8 位像素、还原通道序 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py jpeg-arcnn work/jpeg-arcnn --imagenette /path/to/imagenette2-160
examples/jpeg-arcnn/demo.sh --onnx work/jpeg-arcnn/ready/jpeg-arcnn.onnx
```

本模型调用 [train_wave3_restore.py](../../apps/npu-benchmark/tools/train_wave3_restore.py)。

### 使用已有 RKNN

```bash
examples/jpeg-arcnn/demo.sh --weights /path/to/jpeg-arcnn.rknn
```

实测 RKNN SHA-256 为 `207bf50e61af83746f1e795eb90cb0949cdff36fcbb698aa95034d161b1f88d8`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-jpeg-arcnn-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-jpeg-arcnn/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/extended/restore-clean.json)
