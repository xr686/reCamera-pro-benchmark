# Linear SVM

模型 ID 为 `sms-svm`，任务为短信垃圾信息识别。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 512 维 | FP16 | 单词与双词组计数，CRC32 映射 512 维 | 两类分数 argmax |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py sms-svm work/sms-svm
examples/sms-svm/demo.sh --onnx work/sms-svm/ready/sms-svm.onnx
```

本模型调用 [train_wave3_supervised.py](../../apps/npu-benchmark/tools/train_wave3_supervised.py)。

### 使用已有 RKNN

```bash
examples/sms-svm/demo.sh --weights /path/to/sms-svm.rknn
```

实测 RKNN SHA-256 为 `9204d000593d477494beb36f345805d4feecf2e4c67012a6c1e2a04e5a0a17d3`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 UTF-8 文本文件。 在上面的打包命令末尾加上 `--input message.txt` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-sms-svm-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-sms-svm/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/extended/fashion-text.json)
