# ESPCN ×3

模型 ID 为 `espcn-x3`，任务为图像超分辨率。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 224×224 | FP16 | 取亮度 Y / 255 | 亮度图裁到 [0,1] |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py espcn-x3 work/espcn-x3
examples/espcn-x3/demo.sh --onnx work/espcn-x3/ready/espcn-x3.onnx
```

### 使用已有 RKNN

```bash
examples/espcn-x3/demo.sh --weights /path/to/espcn-x3.rknn
```

实测 RKNN SHA-256 为 `661e92d2922946daa4afc4ad26ad385cdc03405ef4899f57447311c5dc347595`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-espcn-x3-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-espcn-x3/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-3.json) · [模型来源](https://github.com/onnx/models/blob/c5cae0f1942c8d7727372c7b1f6b485aa39e4421/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx)
