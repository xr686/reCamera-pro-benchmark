# Block DCT energy probe · noise-level

模型 ID 为 `noise-level-dct`，任务为高斯噪声强度识别。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 64×64 | FP16 | 灰度化并除以 255 | 类别分数排序 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py noise-level-dct work/noise-level-dct --imagenette /path/to/imagenette2-160
examples/noise-level-dct/demo.sh --onnx work/noise-level-dct/ready/noise-level-dct.onnx
```

本模型调用 [train_camera_quality.py](../../apps/npu-benchmark/tools/train_camera_quality.py)。

### 使用已有 RKNN

```bash
examples/noise-level-dct/demo.sh --weights /path/to/noise-level-dct.rknn
```

实测 RKNN SHA-256 为 `99f89697d794190d04ba626c3b1a9d27bd49f1a6ea9a0f91ff55720020d060fb`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-noise-level-dct-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-noise-level-dct/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/sequence/noise-level.json)
