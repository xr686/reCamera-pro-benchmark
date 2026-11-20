# CLIP ViT-B/32 图像塔

模型 ID 为 `clip-image`，任务为图文零样本分类。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 224×224 | FP16 | 短边 224、中心裁剪、RGB CLIP 标准化 | 512 维向量归一化，与对端图库做余弦检索 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py clip-image work/clip-image
examples/clip-image/demo.sh --onnx work/clip-image/ready/clip-image.onnx
```

### 使用已有 RKNN

```bash
examples/clip-image/demo.sh --weights /path/to/clip-image.rknn
```

实测 RKNN SHA-256 为 `8d8eec73a731a40438bcd98c190cc6a46f41c3adc6d308e0bfbc861cedefedc4`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-clip-image-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-clip-image/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/clip-image.json)
