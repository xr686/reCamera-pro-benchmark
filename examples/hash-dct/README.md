# DCT perceptual hash

模型 ID 为 `hash-dct`，任务为近重复图像检索。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 32×32 | FP16 | 灰度化并除以 255 | 逐位符号阈值、映射 ±1 和 L2 归一化 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py hash-dct work/hash-dct
examples/hash-dct/demo.sh --onnx work/hash-dct/ready/hash-dct.onnx
```

本模型调用 [build_field_vision.py](../../apps/npu-benchmark/tools/build_field_vision.py)。

### 使用已有 RKNN

```bash
examples/hash-dct/demo.sh --weights /path/to/hash-dct.rknn
```

实测 RKNN SHA-256 为 `54800b6b3a19d39070a05c5a3550300a9636f63f26f29e8531693ca980d795db`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-hash-dct-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-hash-dct/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/field/hash-verified.json)
