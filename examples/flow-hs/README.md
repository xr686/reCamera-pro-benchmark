# Horn–Schunck (12 iterations)

模型 ID 为 `flow-hs`，任务为稠密光流估计。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 160×112 × 2 | FP16 | 双灰度帧堆叠、归一化 | 转为水平和垂直位移图 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py flow-hs work/flow-hs
examples/flow-hs/demo.sh --onnx work/flow-hs/ready/flow-hs.onnx
```

本模型调用 [build_field_vision.py](../../apps/npu-benchmark/tools/build_field_vision.py)。

### 使用已有 RKNN

```bash
examples/flow-hs/demo.sh --weights /path/to/flow-hs.rknn
```

实测 RKNN SHA-256 为 `5a39c3a30c073945df215ae5f9d6290d62a81e0471d8592f58fab8b31372d60f`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入形状为 `[2, 112, 160]`、类型为 `uint8` 的 NumPy 数组，以 NPY 格式保存。 在上面的打包命令末尾加上 `--input sample.npy` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-flow-hs-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-flow-hs/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/field/field-flow.json)
