# YAMNet

模型 ID 为 `yamnet_3s`，任务为声音事件分类。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 48000（3 s） | FP16 | 16 kHz 单声道波形 | 按时间平均 521 类分数、取首选类别 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py yamnet_3s work/yamnet_3s
examples/yamnet_3s/demo.sh --onnx work/yamnet_3s/ready/yamnet_3s.onnx
```

### 使用已有 RKNN

```bash
examples/yamnet_3s/demo.sh --weights /path/to/yamnet_3s.rknn
```

实测 RKNN SHA-256 为 `97abedba18030df00260f7d6f0f8f90e6a3caa024586ceeb566272bf1bc57ef2`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入形状为 `[48000]`、类型为 `float32` 的 NumPy 数组，以 NPY 格式保存。 在上面的打包命令末尾加上 `--input sample.npy` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-yamnet-3s-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-yamnet-3s/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-3.json) · [模型来源](https://github.com/airockchip/rknn_model_zoo/tree/bad6c7334531becaf90a561988519b7bec34d0ab/examples/yamnet)
