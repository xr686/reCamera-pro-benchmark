# reCamera Pro NPU 基准应用

按模型整理的真机结果见[架构专项报告](ARCHITECTURES-1.3.2.md)。

在 RV1126B 上以 batch 1 顺序加载模型，记录前处理、同步推理、后处理、串行吞吐率、任务质量和独占状态。V1.3.2 的第五批模型见[第五批实测报告](FIELD-SEQUENCE-1.3.2.md)，第四批模型见[第四批实测报告](FIELD-1.3.2.md)，此前的 131 个配置见[第三批报告](EXTENDED-1.3.2.md)，前 85 项的逐模型数据见[首批报告](FIRMWARE-1.3.2.md)。V1.3.1 的 39 个配置、样本和转换边界见[上一版报告](MULTITASK.md)。三个 ImageNet 分类器的三轮重复性和 FP32 对照见[分类专项报告](BENCHMARK.md)。

## 从本仓库准备应用和资源

在 reCamera Pro Extension API SDK 的工作副本中，将本目录复制为 `apps/npu-benchmark`。按下文“重建模型和应用”生成打包所需的三个基础分类权重，并用 SDK 的 `market/packaging/build.py` 打包。SDK 需提供 manifest 声明的 Python 3.11、Kit、ABI 和 RKNN 运行时。[SDK 源码](https://github.com/Seeed-Studio/recamera-pro-ext-api)

架构专项可直接用本仓库的 LFS 权重及公开验证图复测。从 benchmark 仓库根目录运行以下命令，生成 StarNet 的资源包。将设备上的“模型 ID”设为 `starnet`。可在命令尾部追加属于最近 16 个模型且已有权重的 ID。打包工具按归档的 SHA-256 校验模型、图片和标注。[架构权重](../../models/app-architectures/README.md) · [专项指标和模型 ID](../../benchmark_results/app/RV1126B-architectures.csv)

```bash
git lfs pull --include="models/app-architectures/starnet.rknn"
curl -fL https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz -o imagenette2-160.tgz
tar -xzf imagenette2-160.tgz
python3 apps/npu-benchmark/tools/pack_imagenette_run.py \
  benchmark_results/app/imagenette-100 models/app-architectures \
  imagenette2-160/val architectures-resources.tar.gz starnet
```

下文中的历史资源包及应用安装包名称对应各轮本地归档。其余配置可按对应报告和 tools 中的转换、训练、评测脚本准备。公开逐次记录可按 `series/run` 查阅。[测量记录](../../benchmark_results/app/runs) · [全量结果](../../benchmark_results/app/measurements.csv)

## 在设备上复测

1. 在应用中心安装 `npu-benchmark-0.8.0-arm64.tar.gz`，将 `architectures-resources.tar.gz` 解压到 `/userdata/local/appdata/npu-benchmark/`。这个资源包包含独立运行架构基准所需的模型、标注、图片和校验清单。
2. 在“测试范围”选择“新增分类模型”（默认），留空“模型 ID”即可测量新架构。填写 ID 可单测，例如 `starnet`，也可以逗号分隔选择多个。需要复现历史配置时，安装对应历史资源包后填写 ID，或选择“全部已安装配置”。
3. 记下正在使用 NPU 的应用，在应用中心暂停它们，再启动 **NPU 模型基准测试**。基准取得固件 broker 独占租约，并等待三个间隔一秒的零负载读数。应用在测量过程中定期检查租约和打开 NPU 的进程。
4. 日志出现 `Benchmark complete` 后，从 `/userdata/local/appdata/npu-benchmark/report.md` 读取最新完整报告。同目录 `report.json` 保存逐样本质量、逐次时延、设备信息和独占检查。跨任务运行期间可读取 `multitask-progress.json` 查看已完成的模型。使用 UTC 时间区分不同轮次。停止基准应用，恢复之前运行的应用。第四批报告、累计 CSV 和原始证据另存于同目录的 `FIELD-1.3.2.md`、`RV1126B-field-combined.csv`、`field-evidence.tar.gz`。

从主机部署资源并取回结果，其中 `DEVICE` 为设备地址。

```sh
DEVICE=recamera.local  # 替换为设备主机名或 IP
ssh root@"$DEVICE" 'mkdir -p /userdata/local/appdata/npu-benchmark'
gzip -dc architectures-resources.tar.gz | ssh root@"$DEVICE" 'tar -xf - -C /userdata/local/appdata/npu-benchmark'
ssh root@"$DEVICE" 'cat /userdata/local/appdata/npu-benchmark/report.json' > board-report.json
```

## 算法和资源

`task_models.py` 定义原始网络的输入尺寸、通道格式、测试集和解码器。`task_inputs.py` 实现图像、模板对、波形和声学特征的前处理。`task_decode.py` 包含每种输出头的解码。`task_quality.py` 使用标注计算质量，`task_suite.py` 执行板端测量和数据校验。

自备 ONNX 时，可用 `tools/compile_onnx.py SOURCE.onnx TARGET.rknn` 以 RKNN-Toolkit2 2.3.2 针对 rv1126b 编译。动态输入通过重复的 `--shape` 参数按原图输入顺序指定。MediaPipe 输入布局、BlazePose 分数限幅、NeMo 固定帧长和常量权重折叠分别见 `tools/mediapipe_inputs.py`、`tools/clip_detection_logits.py`、`tools/static_asr.py` 和 `tools/fold_weights.py`。NAFNet 在权重折叠后先用 onnxsim 固定输入 `lq:1,3,384,384`，再以 `tools/specialize_naf_pool.py` 将正好覆盖全幅的均值窗化简为全局池化。源图和实际运行权重的 SHA-256 见资源包中的 `multitask/suite.json`。对只有预编译 RKNN 的模型，清单记录运行权重 SHA-256。

资源工作目录包含 `sources/<ID>.onnx`、`compiled/<ID>.rknn`、`fixtures/groups.json` 及该文件引用的媒体和标注。样本生成过程见 `tools/make_task_samples.py` 、`tools/more_labels.py` 和 `tools/natural_samples.py`。后者接收下载缓存目录、fixtures 目录两个参数，生成 3 段真实视频的跟踪帧对及 ESC-50 声音样本。在具有 NumPy、OpenCV、ONNX Runtime 的主机上运行 `tools/task_reference.py 工作目录` 可重算 CPU 输出指纹和清单。NeMo 词表位于工作目录的 `aux`，改写图位于 `folded`。用以下命令从已有清单和样本重打资源包。

```sh
python3 apps/npu-benchmark/tools/pack_resources.py 工作目录 分类fixtures目录 multitask-resources.tar.gz
```

## 准备同一批标注图片

在主机 SDK 目录运行下列命令。测试集使用 Imagenette2-160 验证集的文件夹标签。每类按相对路径的 SHA-256 排序取前 30 张，按类别交错排列。脚本生成逐图 SHA-256，设备在测试前校验。图片单独部署。[Imagenette 数据说明](https://github.com/fastai/imagenette)

```sh
curl -fL https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz -o imagenette2-160.tgz
tar -xzf imagenette2-160.tgz
python3 apps/npu-benchmark/tools/prepare_samples.py imagenette2-160/val fixtures
tar -cf - fixtures | ssh root@设备地址 'mkdir -p /userdata/local/appdata/npu-benchmark && tar -xf - -C /userdata/local/appdata/npu-benchmark'
```

## 重建模型和应用

在 SDK 目录运行。第一步使用主机 PyTorch 2.6 和 TorchVision 0.21 的 ImageNet-1K V1 权重，导出 opset 12、固定输入形状的 ONNX。第二步切换到主机 RKNN-Toolkit2 2.3.2 环境，编译 RV1126B FP16 模型并更新 manifest 中的体积和哈希。模型权重缓存由 TorchVision 管理。

```sh
python apps/npu-benchmark/tools/export_models.py build/benchmark-onnx
# 切换到 RKNN-Toolkit2 2.3.2 的 Python 环境
python apps/npu-benchmark/tools/convert_models.py build/benchmark-onnx
python3 market/packaging/build.py apps/npu-benchmark --out dist/
```

已从干净检出重新导出并编译三个基础分类器，成功生成 0.8.0 arm64 安装包。本次重编译的三个 RKNN 与历史板端记录的 SHA-256 均不同。转换脚本会更新本地应用 manifest；归档结果保留当时实际运行的权重摘要。需要逐字节一致的模型时，可从仓库提供的 24 个 LFS 权重中选择，并与[模型哈希清单](../../benchmark_results/app/model-hashes.csv)核对。

可用同一批图片核对原始 FP32 模型的预测。在安装 PyTorch、TorchVision、OpenCV 和本项目 Kit 依赖的主机环境运行以下命令。输出顺序与设备报告的 samples 相同。

```sh
PYTHONPATH=. python apps/npu-benchmark/tools/reference.py fixtures reference.json
```

三个模型的结构及权重来源见 [ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)、[ShuffleNet V2 x1.0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.shufflenet_v2_x1_0.html) 和 [MobileNet V2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)。

## 复核已归档的结果

原始证据包中的 `verified.json` 是整轮板端记录，`*-audit.json` 给出相应的 AppMgr 采样和恢复状态。若某个模型在独立轮追加，报告的“设备状态和原始证据”表会列出对应 JSON。可在主机用 NumPy 重建汇总表，按实测时间顺序传入板端 JSON。后轮仅覆盖其实际复测的模型。

```sh
python3 apps/npu-benchmark/tools/render_multitask.py \
  --app apps/npu-benchmark --assets 工作目录 \
  --notes apps/npu-benchmark/tools/evaluation-notes.md --out MULTITASK.md \
  verified.json tracker.json deblur.json
```


## 重建新增算法

交付的资源包包含已编译的权重、用于板端计量的输入、标注和校验清单。新增资源安装在 `multitask-extra/`，两个资源目录可独立升级。算法代码见 `extra_tasks.py`、`signal_catalog.py` 和 `clip_benchmark.py`。主机端训练和转换依赖 PyTorch 2.6、TorchVision 0.21、timm、ONNX/ONNX Runtime、onnxsim 和 RKNN-Toolkit2 2.3.2。训练与处理输入数据还用到 scikit-learn、SciPy、Pillow、OpenCV。

- `tools/fetch_extra_sources.py WORK` 获取固定哈希的原始权重、模型定义、词表和公开数据归档。TorchVision 和 timm 的官方预训练权重由各自的导出脚本读取，具体 checkpoint 和转换参数随资源包中的 `suite.json` 给出。
- `tools/export_vision_suite.py WORK 模型ID`、`tools/export_timm_suite.py WORK 模型ID`、`tools/export_specialists.py WORK 模型ID` 导出图像网络。`tools/train_signal_suite.py WORK 数据集名` 和 `tools/train_commands_digits.py WORK` 训练时序、指令和数字模型，按原始训练/测试划分写出样本。先创建 `WORK/sources`、`WORK/compiled`。
- 普通静态网络用 `tools/compile_onnx.py SOURCE.onnx TARGET.rknn` 编译。CLIP 两个输入使用 `--shape 1,3,224,224` 和 `--shape 1,20`。SuperPoint 先调用 `tools/specialize_superpoint.py`，Mosaic 调用 `tools/specialize_style.py`；两个脚本均接收源与目标 ONNX 路径。M-LSD 先用 tf2onnx 的 `--tflite` 选项转出完整 ONNX，再用 `tools/export_mlsd_heads.py` 导出 CPU 对照的中心响应和端点张量，用 `tools/mediapipe_inputs.py` 包装 NHWC 输入后编译。
- 在主机 SDK 根目录设置 `PYTHONPATH=.`，运行 `tools/prepare_extra_suite.py WORK 分类fixtures目录 VOC2012目录 本地图像对目录` 生成图库、原始 ONNX 输出指纹及清单。配套资源已包含所有固定图像对；原始工程的图像对目录格式为 `frame_N/{1.ppm,2.ppm,H_1_2}`。`tools/score_extra_onnx.py WORK` 使用相同样本计算 CPU 对照质量。`tools/pack_resources.py --extra WORK 分类fixtures目录 multitask-extra-resources.tar.gz` 校验所有引用文件后生成增量包。
- `tools/render_firmware_suite.py` 读取完整真机记录、对应 AppMgr 审计及 CPU 对照，可重新输出此次 Markdown 和 CSV。逐模型的应用 ID、数据集、处理器、采样量及指标定义见实测报告。

## 第三批模型的重建

本轮资源使用 `multitask-extended/`。板端输入契约、输出处理、任务指标由 `extended_models.py`、`extended_tasks.py` 与共用计时器定义。来源权重和训练设置见每个配置的 SHA-256 清单及原始证据。以下主机脚本位于应用的 `tools/`，以工作目录作为第一个参数。

- 下载素材用 `fetch_wave3_data.py`、`fetch_wave3_mvtec.py` 和 `fetch_wave3_audio.py`。音频脚本的第二个参数为 ESC-50 官方 esc50.csv。图像块评估使用前一批图像对及其单应矩阵。Motorcycle 来自 scikit-image 的 stereo_motorcycle 文件，VOC 框标注使用 VOC2012 的 Annotations 和 ImageSets/Main/val.txt。
- 图像分类使用 `export_wave3_backbone.py` 指定模型 ID，几何组使用 `export_wave3_geometry.py`（需 Kornia 0.7.4）。YOLOv5s-ReLU、YOLOv10n 的源图来自 Rockchip rknn_model_zoo。工具针对固定输入调用 `compile_onnx.py` 编译 FP16 RKNN。
- `train_wave3_supervised.py` 分别接受 tabular、spam、fashion；`train_wave3_restore.py` 的第二个参数是首批分类 fixtures，`train_wave3_audio.py` 的第二个参数是 mini_speech_commands.zip。工业异常模型由 `train_wave3_mvtec.py` 训练。x-vector 编译前对原始 ONNX 运行 `lower_speaker_pool.py 原图 目标图`，将二维空间均值等价改写为全局平均池化。
- 将 ONNX 放在工作目录的 sources/，相同 ID 的 RKNN 放在 compiled/。在 SDK 的 Python 路径下运行 `prepare_wave3.py 工作目录 第二批fixtures目录 Motorcycle数据目录 VOC2012目录` 并执行 `pack_resources.py --extended 工作目录 工作目录 multitask-extended-resources.tar.gz`。该流程用 ONNX Runtime 检查独立输入的任务质量，记录输入、原图和板端权重的 SHA-256。
- 解开新增证据包后，results/ 下的计划文件与对应审计包含实测模型 ID 和分组。所有者检查、CPU 对照、阶段均值和累计表可通过 `render_wave3.py --help` 所列参数重新生成，previous 输入为上一轮 85 项的 CSV。

## 第四批模型与资源

`field_models.py` 注册输入契约，`field_tasks.py` 实现光流、音高、光源以及多标签指标。运行 `tools/fetch_wave4.py WORK` 下载公开数据并计算来源哈希。准备 `WORK/{data,sources,compiled,fixtures,logs,release}` 和前一批的 Imagenette 验证图，使用 CPU PyTorch、TorchVision、timm、scikit-learn、NumPy、OpenCV、ONNX Runtime 和 RKNN-Toolkit2。

- `tools/train_field_images.py WORK DOMAIN VOC2012目录` 的 DOMAIN 为 eurosat、flowers、gts 或 voc-tags。每种输入分别训练四个模型，保存训练、测试源图清单。TF Flowers 图像许可随第四批资源的 source-notices/ 目录提供。
- `tools/train_field_vectors.py WORK` 生成表格与文本模型。`tools/build_field_vision.py WORK 第二批fixtures目录` 导出光流、光源估计和感知哈希。`tools/build_field_audio.py WORK` 训练或生成音频模型和独立测试输入。
- 在 RKNN-Toolkit2 环境运行 `tools/build_field_models.py WORK` 导出并编译 8 个预训练 ImageNet 网络。传入其他模型 ID 时会编译对应的 `sources/ID.onnx`。`tools/prepare_field.py WORK 第二批fixtures目录` 会运行原始 ONNX，生成逐模型任务对照、输入指纹与 SHA-256 清单。
- `tools/pack_resources.py --field WORK WORK multitask-field-resources.tar.gz` 生成资源包。设备端对独立组分别运行后，用 `tools/render_wave3.py --field --work WORK --previous 上一轮累计CSV --output FIELD-1.3.2.md 计划文件...` 重新生成文档和累计 CSV。每份分组报告都需要相应的 AppMgr 审计。

## 第五批模型与资源

在同一工作目录下创建 data、sources、compiled、fixtures、logs、results、release 子目录。主机使用 PyTorch、TorchVision、timm、scikit-learn、NumPy、OpenCV、Pillow、ONNX Runtime 和 RKNN-Toolkit2。运行准备脚本时将 SDK 根目录加入 PYTHONPATH。

1. 运行 `tools/build_sequence_backbones.py WORK` 导出并编译八个预训练骨干。`tools/fetch_sequence_data.py WORK` 下载 12 个公开序列数据集，`tools/train_sequence_models.py WORK` 训练三套序列分类器。
2. 运行 `tools/train_camera_quality.py WORK imagenette2-160目录`、`tools/train_sequence_objects.py WORK VOC2012目录`、`tools/train_sequence_pets.py WORK oxford-iiit-pet.tgz`。Pet 压缩包使用 fast.ai 镜像中的同名数据集，包括官方图像、trimap 和拆分文件。[Oxford-IIIT Pet 镜像](https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz)
3. 用 `tools/build_field_models.py WORK 模型ID...` 将其余的 sources/ID.onnx 编译为 RKNN。对 pet-breed-shuffle 先用 `tools/lower_speaker_pool.py sources/ID.onnx data/ID-rv.onnx` 等价改写空间均值，再用 `tools/compile_onnx.py data/ID-rv.onnx compiled/ID.rknn` 编译。两种 SqueezeNet 分割器的转换对照见实测报告。保留原始 sources/ID.onnx 以生成 CPU 参考。`tools/prepare_sequence.py WORK 第四批fixtures目录` 计算 CPU 质量参考和资源摘要。`tools/pack_resources.py --sequence WORK WORK multitask-sequence-resources.tar.gz` 创建新资源包。
4. 应用中心的“模型 ID”支持单项或逗号分隔的清单。实测完成后，设备持久化目录内的 FIELD-SEQUENCE-1.3.2.md、RV1126B-sequence-combined.csv 和 sequence-evidence.tar.gz 分别包含第五批报告、全量指标和逐次测量记录。`tools/render_wave3.py --sequence --work WORK --previous 第四批累计CSV --output 报告路径 分组计划...` 可重新生成报告。


## 架构专项的重建

`architecture_models.py` 为每个模型记录一个实测版本、官方 checkpoint、输入尺寸和结构说明。历史 ImageNet 分类骨干的合并规则位于 `tools/render_architectures.py` 的 HISTORICAL 表。使用该脚本时提供上一轮累计 CSV，工具会检查每个历史分类 ID 恰好归入一个代表模型。

1. 在主机 RKNN 环境准备 WORK 下的 sources、compiled、data、logs、fixtures、results、release 目录。运行 `tools/build_architectures.py WORK` 导出原始 ONNX 并编译为 RV1126B FP16 RKNN。参数可附加一个或多个架构 ID。使用公开的 timm 预训练权重。
2. 在 PYTHONPATH 中加入 SDK 根目录，运行 `tools/prepare_architectures.py WORK 上轮fixtures目录 架构ID...`，复用相同的 100 张分类验证图。原始 ONNX 用于主机对照。打包用 `tools/pack_resources.py --architectures WORK WORK architectures-resources.tar.gz`。
3. 在真机通过基准应用测试并归档结果。运行 `tools/render_architectures.py --work WORK --previous 上轮累计CSV --output ARCHITECTURES-1.3.2.md 分组计划...`。逐次计时、独占审计和模型来源随 architecture-evidence.tar.gz 提供。
