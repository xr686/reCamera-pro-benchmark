# Zero-mean SAD

模型 ID 为 `stereo-zsad`，任务为双目视差估计。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 160×96 × 2 | FP16 | 双灰度帧堆叠、归一化 | 32 个视差代价逐像素 argmin |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py stereo-zsad work/stereo-zsad
examples/stereo-zsad/demo.sh --onnx work/stereo-zsad/ready/stereo-zsad.onnx
```

本模型调用 [export_wave3_geometry.py](../../apps/npu-benchmark/tools/export_wave3_geometry.py)。

### 使用已有 RKNN

```bash
examples/stereo-zsad/demo.sh --weights /path/to/stereo-zsad.rknn
```

实测 RKNN SHA-256 为 `9e940f262b9b75e988177315c365b7801aa101c099adf5c32fbfb161daaf8f1d`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入形状为 `[2, 96, 160]`、类型为 `uint8` 的 NumPy 数组，以 NPY 格式保存。 在上面的打包命令末尾加上 `--input sample.npy` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-stereo-zsad-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-stereo-zsad/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/extended/stereo.json)
