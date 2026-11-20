# N-gram ridge classifier · sent

模型 ID 为 `sent-ridge`，任务为英文短评情感分类。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 512 维 | FP16 | 单词与双词组计数，CRC32 映射 512 维 | 类别分数排序 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py sent-ridge work/sent-ridge
examples/sent-ridge/demo.sh --onnx work/sent-ridge/ready/sent-ridge.onnx
```

本模型调用 [train_field_vectors.py](../../apps/npu-benchmark/tools/train_field_vectors.py)。

### 使用已有 RKNN

```bash
examples/sent-ridge/demo.sh --weights /path/to/sent-ridge.rknn
```

实测 RKNN SHA-256 为 `d187c187d7719754a99e6d9119a355e56db1ac956168b984f883bd5b4a26f50c`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 UTF-8 文本文件。 在上面的打包命令末尾加上 `--input message.txt` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-sent-ridge-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-sent-ridge/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/field/text.json)
