# VitTrack

模型 ID 为 `object_tracking_vittrack_2023sep`，任务为单目标跟踪。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 128×128 + 256×256 | FP16 | 模板与搜索区 BGR 逐通道标准化 | 响应图乘余弦窗、中心偏移与尺度回归 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

### 从源模型构建

导出和训练脚本使用主机的模型开发环境。最后一条命令在 RKNN-Toolkit2 2.3.2 环境执行。

```bash
python3 examples/export.py object_tracking_vittrack_2023sep work/object_tracking_vittrack_2023sep
examples/object_tracking_vittrack_2023sep/demo.sh --onnx work/object_tracking_vittrack_2023sep/ready/object_tracking_vittrack_2023sep.onnx
```

### 使用已有 RKNN

```bash
examples/object_tracking_vittrack_2023sep/demo.sh --weights /path/to/object_tracking_vittrack_2023sep.rknn
```

实测 RKNN SHA-256 为 `f5090f193d8e505ef9a450680515899fff1dde233faf91380d52c7b12c42bce0`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

第一个文件为目标模板，第二个为搜索区域。 在上面的打包命令末尾加上 `--input template.jpg --second search.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-object-tracking-vittrack-2023sep-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-object-tracking-vittrack-2023sep/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-1.json) · [模型来源](https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx)
