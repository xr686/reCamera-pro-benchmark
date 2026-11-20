# SCRFD-500M

模型 ID 为 `scrfd500m_640_fp16`，任务为人脸检测。

| 输入 | 实测格式 | 前处理 | 预测输出 |
|---|---|---|---|
| 640×640 | FP16 | RGB uint8 | 距离回归、NMS |

## 构建应用

在 benchmark 仓库根目录执行以下命令。主机环境和 SDK 设置见[示例指南](../README.md)。

实测权重随 SDK 的 `face-recognition` 应用提供。准备含 `models/scrfd500m_640_fp16.rknn` 的该应用安装包，然后运行：

```bash
examples/scrfd500m_640_fp16/demo.sh --from-pack /path/to/face-recognition-arm64.tar.gz
```

[SDK 应用源码](https://github.com/Seeed-Studio/recamera-pro-ext-api/tree/main/apps/face-recognition)

### 使用已有 RKNN

```bash
examples/scrfd500m_640_fp16/demo.sh --weights /path/to/scrfd500m_640_fp16.rknn
```

实测 RKNN SHA-256 为 `74ab137a2d026a56630c8eed14fb591fda778a7567b05c3e9c0751a3a6756774`。自行重编译的兼容权重可附加 `--rebuilt`。

## 运行示例

传入 PNG 或 JPEG 图片。 在上面的打包命令末尾加上 `--input photo.jpg` 即可使用自己的输入。省略输入会生成一次用于检查推理链路的样本。

在设备应用中心安装 `dist/demo-scrfd500m-640-fp16-1.0.1-arm64.tar.gz` 并启动。应用取得独占 NPU 后执行一次推理。结果页显示预测摘要，完整输出可用 SSH 读取。

```bash
ssh "$DEVICE" cat /userdata/local/appdata/demo-scrfd500m-640-fp16/prediction.json
```

较大的张量保存到同目录的 `prediction.npz`，JSON 中的 `npz_key` 指向对应数组。结束示例时在应用中心停止应用。

[测量记录](../../benchmark_results/app/runs/previous-85/legacy-2.json)
