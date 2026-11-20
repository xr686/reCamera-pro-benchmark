# MiniFASNet V2

模型 ID 为 `liveness_minifasnet_fp16`，任务为人脸防伪。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 80×80 | FP16 | BGR uint8 | 类别 argmax |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

实测权重随 SDK 的 `face-recognition` 应用提供。准备含 `models/liveness_minifasnet_fp16.rknn` 的该应用安装包，然后运行：

```bash
examples/liveness_minifasnet_fp16/demo.sh --from-pack /path/to/face-recognition-arm64.tar.gz
```

[SDK 应用源码](https://github.com/Seeed-Studio/recamera-pro-ext-api/tree/main/apps/face-recognition)

### 使用已有 RKNN

```bash
examples/liveness_minifasnet_fp16/demo.sh --weights /path/to/liveness_minifasnet_fp16.rknn
```

实测 RKNN SHA-256 为 `c6d33e994e1bebb26c19bd9728e14505fcbb2e605b6a84373e3d15824b19eac1`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-liveness-minifasnet-fp16-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-liveness-minifasnet-fp16/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-2.json)
