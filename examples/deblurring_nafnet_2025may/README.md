# NAFNet

模型 ID 为 `deblurring_nafnet_2025may`，任务为运动去模糊。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 384×384 | FP16 | RGB / 255 | 转 BGR、缩放并裁到 0–255 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py deblurring_nafnet_2025may work/deblurring_nafnet_2025may
examples/deblurring_nafnet_2025may/demo.sh --onnx work/deblurring_nafnet_2025may/ready/deblurring_nafnet_2025may.onnx
```

本模型调用 [fold_weights.py](../../apps/npu-benchmark/tools/fold_weights.py) → [specialize_naf_pool.py](../../apps/npu-benchmark/tools/specialize_naf_pool.py)。

需要先按 1×3×384×384 静态化，再运行 specialize_naf_pool.py；详见基准应用 README。

### 使用已有 RKNN

```bash
examples/deblurring_nafnet_2025may/demo.sh --weights /path/to/deblurring_nafnet_2025may.rknn
```

实测 RKNN SHA-256 为 `a209027e49695347c5455e6bdd5b0b8eab1d7a96f3f0b53efc2a3dad4e4566b7`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-deblurring-nafnet-2025may-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-deblurring-nafnet-2025may/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-3.json) · [模型来源](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/deblurring_nafnet/deblurring_nafnet_2025may.onnx)
