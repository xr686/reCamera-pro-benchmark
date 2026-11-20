# Cosine basis ridge · GunPoint

模型 ID 为 `gunpoint-cosine`，任务为手部动作轨迹分类。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 1×150 | FP16 | 按训练集通道统计量标准化，转换为 FP32 张量 | 类别分数排序 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py gunpoint-cosine work/gunpoint-cosine
examples/gunpoint-cosine/demo.sh --onnx work/gunpoint-cosine/ready/gunpoint-cosine.onnx
```

本模型调用 [train_sequence_models.py](../../apps/npu-benchmark/tools/train_sequence_models.py)。

### 使用已有 RKNN

```bash
examples/gunpoint-cosine/demo.sh --weights /path/to/gunpoint-cosine.rknn
```

实测 RKNN SHA-256 为 `ab053c99c5f39ae7d85069ebd05966803b1e8424f30351d0480b864c46c60db0`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入形状为 `[1, 150]`、类型为 `float32` 的 NumPy 数组，以 NPY 格式保存。 在上面的打包命令末尾加上 `--input sample.npy` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-gunpoint-cosine-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-gunpoint-cosine/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/sequence/seq-1.json)
