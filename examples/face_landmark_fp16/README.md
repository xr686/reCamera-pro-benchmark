# MediaPipe FaceMesh

模型 ID 为 `face_landmark_fp16`，任务为人脸关键点。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 192×192 | FP16 | RGB uint8 | 提取 468 个关键点的平面坐标 |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

实测权重随 SDK 的 `facemesh-reader` 应用提供。准备含 `models/face_landmark_fp16.rknn` 的该应用安装包，然后运行：

```bash
examples/face_landmark_fp16/demo.sh --from-pack /path/to/facemesh-reader-arm64.tar.gz
```

[SDK 应用源码](https://github.com/Seeed-Studio/recamera-pro-ext-api/tree/main/apps/facemesh-reader)

### 使用已有 RKNN

```bash
examples/face_landmark_fp16/demo.sh --weights /path/to/face_landmark_fp16.rknn
```

实测 RKNN SHA-256 为 `9b6b39ed295971dd0d044c10e80625de0376dd850514b2ef25da5f25d29e986d`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-face-landmark-fp16-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-face-landmark-fp16/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-2.json)
