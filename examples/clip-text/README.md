# CLIP ViT-B/32 文本塔

模型 ID 为 `clip-text`，任务为以文检图。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 20 token | FP16 | 英语 BPE、补齐至 20 token | 512 维向量归一化，与对端图库做余弦检索 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py clip-text work/clip-text
examples/clip-text/demo.sh --onnx work/clip-text/ready/clip-text.onnx
```

### 使用已有 RKNN

```bash
examples/clip-text/demo.sh --weights /path/to/clip-text.rknn
```

实测 RKNN SHA-256 为 `01a2eedfcbb1f8a8108f252d442fea48bbfb6900d9ff0d8d760ee8ebe40a9d8a`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 20 个整数 token ID 的 NPY 文件，包含起始、结束与填充位置。使用基准的 CLIP 词表。 在上面的打包命令末尾加上 `--input token-ids.npy` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-clip-text-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-clip-text/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/clip-text.json)
