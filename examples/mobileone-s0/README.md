# MobileOne-S0

模型 ID 为 `mobileone-s0`，任务为图像分类。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 224×224 | FP16 | 权重对应短边缩放、中心裁剪、RGB 标准化 | logits 排序 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py mobileone-s0 work/mobileone-s0
examples/mobileone-s0/demo.sh --onnx work/mobileone-s0/ready/mobileone-s0.onnx
```

本模型调用 [export_timm_suite.py](../../apps/npu-benchmark/tools/export_timm_suite.py)。

### 使用已有 RKNN

```bash
examples/mobileone-s0/demo.sh --weights /path/to/mobileone-s0.rknn
```

实测 RKNN SHA-256 为 `5ee23702871c0e5a8c1e1b9911d385696e89b8bcec86eca0f4e60bd1540557fd`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-mobileone-s0-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-mobileone-s0/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/compact.json) · [模型来源](https://huggingface.co/timm/mobileone_s0.apple_in1k)
