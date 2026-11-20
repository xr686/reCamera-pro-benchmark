# Shades of Gray (p=4)

模型 ID 为 `wb-shades`，任务为光源颜色估计。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 96×96 | FP16 | 缩放、BGR 转 RGB、除以 255 | 取 RGB 光源三分量 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py wb-shades work/wb-shades
examples/wb-shades/demo.sh --onnx work/wb-shades/ready/wb-shades.onnx
```

本模型调用 [build_field_vision.py](../../apps/npu-benchmark/tools/build_field_vision.py)。

### 使用已有 RKNN

```bash
examples/wb-shades/demo.sh --weights /path/to/wb-shades.rknn
```

实测 RKNN SHA-256 为 `95109b488571c5f71cdbb197ef378a652432a3a01b26751932f94771ac823eba`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-wb-shades-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-wb-shades/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/field/field-wb.json)
