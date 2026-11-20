# Multiscale Inception CNN · ElectricDevices

模型 ID 为 `sig-electricdevices`，任务为用电设备识别。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 1 通道 × 96 点 | FP16 | 训练集均值和标准差归一化 | logits 排序 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py sig-electricdevices work/sig-electricdevices
examples/sig-electricdevices/demo.sh --onnx work/sig-electricdevices/ready/sig-electricdevices.onnx
```

本模型调用 [train_signal_suite.py](../../apps/npu-benchmark/tools/train_signal_suite.py)。

### 使用已有 RKNN

```bash
examples/sig-electricdevices/demo.sh --weights /path/to/sig-electricdevices.rknn
```

实测 RKNN SHA-256 为 `d68807a5d539366ee3282be353adc5e5bff99b36fb8f1b744665ebba8ef44850`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入形状为 `[1, 96]`、类型为 `float32` 的 NumPy 数组，以 NPY 格式保存。 在上面的打包命令末尾加上 `--input sample.npy` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-sig-electricdevices-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-sig-electricdevices/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/signals.json)
