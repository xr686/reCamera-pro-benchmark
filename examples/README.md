# reCamera Pro 模型示例

[按任务浏览全部 270 个模型配置](MODELS.md)。每个目录都有模型 ID、输入格式、输出定义和可以直接执行的构建命令。构建工具只把选中的权重放进一个独立 App。适用设备为装有兼容 App Center、Python 3.11 和 RKNN Runtime 2.3.2 的 RV1126B reCamera Pro。

## 跑通第一个模型

在主机准备 Git LFS、Python 和 [reCamera Pro Extension API SDK](https://github.com/Seeed-Studio/recamera-pro-ext-api)。使用已有 RKNN 打包时，主机使用 Python 标准库和 SDK 自带的打包器。

```bash
# 从本 benchmark 仓库根目录运行
git clone https://github.com/Seeed-Studio/recamera-pro-ext-api ../recamera-pro-ext-api
export RECAMERA_SDK_ROOT="$(cd ../recamera-pro-ext-api && pwd)"
git lfs pull --include="models/app-architectures/starnet.rknn"
examples/starnet/demo.sh --input photo.jpg
```

在设备的应用中心安装 `dist/demo-starnet-1.0.1-arm64.tar.gz`，启动 **StarNet** 示例。结果页显示本次预测的五个 ImageNet 类别索引。应用只推理一次，并向后来打开结果页的用户发送已计算的结果。未提供照片时使用生成的图案。完整输出包括输入来源、同步推理毫秒数和预测值。

```bash
export DEVICE=root@recamera.local
ssh "$DEVICE" cat /userdata/local/appdata/demo-starnet/prediction.json
```

也可用 SDK 的部署脚本通过 SSH 上传、安装、启动、验证并停止应用。主机须已信任目标设备 SSH 密钥，密码认证可使用脚本的 `--ask-password` 参数（具体选项见 `--help`）。

```bash
python3 "$RECAMERA_SDK_ROOT/skill/recamera-pysdk/scripts/deploy_app.py" \
  --host recamera.local --user root --archive dist/demo-starnet-1.0.1-arm64.tar.gz \
  --action verify --report dist/demo-starnet-device.json
```

更换输入或权重后，重新打包时递增 `--sequence`，例如 `--sequence 2` 对应版本 `1.0.2`。不同模型使用不同 App ID。每次只启动一个 NPU 示例，结束后可在应用中心停止它。

## 权重和工具

- **24 个随库 RKNN** 可从 Git LFS 单独获取。[下载清单](../models/app-architectures/README.md)
- **264 个有源图的模型** 提供 `examples/export.py MODEL_ID WORK`。脚本按该模型目录里记录的配方获取上游权重、导出或训练模型，以及执行所需的图变换。部分模型需要额外的数据集参数，命令见各自页面。已有转换就绪的 ONNX 时，`examples/build.py MODEL_ID --onnx MODEL.onnx` 会编译 RV1126B FP16 RKNN 并打包。
- **6 个直接使用 SDK 预编译权重的模型** 可从对应的 depth-estimation、face-recognition 或 facemesh-reader 应用安装包中提取。传入 `--from-pack` 即可打包。全部 270 项也支持传入 `--weights` 或含该模型的原始 benchmark 资源包，并核对实测哈希。
- 源图导出在 x86 Linux 的独立 Python 环境运行。根据所选脚本安装依赖。分类配方使用 PyTorch 2.6、TorchVision 0.21、timm 1.0.19 和 ONNX。转换环境使用 RKNN-Toolkit2 2.3.2。音频、文本和训练脚本还可能用到 transformers、scikit-learn、SciPy、OpenCV、Kornia 或 onnxsim。M-LSD 的 TFLite 导出使用 TensorFlow 2.16 和 tf2onnx。直接使用 RKNN 的用户可以跳过这些主机依赖。
- 配方需要原始数据时，`--imagenette` 指向含 train、val 的 Imagenette2-160 根目录，`--voc` 指向 VOC2012 根目录，`--previous-fixtures` 指向含 groups.json 和 imagenette-extra 的历史 fixtures 目录。Oxford-IIIT Pet 归档包含 images、annotations/trimaps、trainval.txt 和 test.txt。部分训练脚本会为同一任务训练多个变体，最终只打包所选模型。重编译结果由自己的编译器和训练环境决定。[历史权重哈希](../benchmark_results/app/model-hashes.csv)
- 默认输入是用于验证执行路径的生成样本。使用自己的图片、文本或 NPY 数组时，示例复用实测的前后处理。对 CLIP，可以传入 `--gallery`，NPZ 包含 vectors 和 labels；省略时输出归一化向量。[任务实现](../apps/npu-benchmark)
- 上游已有的原生 YOLO 可执行文件使用仓库现成的 CMake 和交叉编译脚本。[原生编译和运行说明](../native/README.md) · [VLM 独立应用测量](../benchmark_results/app/vlm.md)

## 验证记录

270 个示例均使用对应的历史实测 RKNN 单独构建了 App Center 安装包，并核对了包内模型 ID、哈希和发布锁。主机按各模型的输入约定检查了全部生成样本。以下四个示例另在 RV1126B 上验证了安装、启停、结果订阅和预测文件。

| 示例 | 输入与权重 | 设备输出 |
|---|---|---|
| [StarNet](starnet/README.md) | Imagenette 图片 ILSVRC2012_val_00022252.JPEG，实测 RKNN | Top-1 为 571，与标签一致 |
| [Airfoil PLS](airfoil-pls/README.md) | 生成的数值输入，实测 RKNN | 回归值 75.75 |
| [Coffee 时序分类](sig-coffee/README.md) | 公开训练数据重新训练并编译，生成的序列输入 | 类别排序为 0、1 |
| [ESPCN ×3](espcn-x3/README.md) | 公开 ONNX 重新编译，生成的图案 | 输出 672×672 的 float32 图像，可从 prediction.npz 取出 |

如需在本地复核示例索引或输入，运行以下命令。第二条命令需要主机安装 NumPy、OpenCV 和 SDK Kit 依赖。历史任务精度和性能数据见[完整测量表](../benchmark_results/app/measurements.csv)。

```bash
python3 examples/verify.py
PYTHONPATH="$RECAMERA_SDK_ROOT" python3 examples/verify.py --prepare-inputs
```
