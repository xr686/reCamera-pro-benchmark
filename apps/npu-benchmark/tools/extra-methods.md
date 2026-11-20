## 输入、模型和精度的含义

| 测试组 | 输入和处理 | 质量指标与样本 |
|---|---|---|
| ImageNet 分类 | 每个预训练权重采用对应的短边缩放、中心裁剪、RGB 和逐通道标准化。OpenCV 完成插值，输出对 1000 类排序。 | 新增分类器使用同一组 Imagenette 验证图，每类 10 张，共 100 张。三个历史对照模型使用每类 30 张、共 300 张。两个样本量分别列在表中。[TorchVision 模型](https://docs.pytorch.org/vision/stable/models.html) · [timm](https://github.com/huggingface/pytorch-image-models) · [Imagenette](https://github.com/fastai/imagenette) |
| 语义分割、显著目标 | LR-ASPP、DeepLabV3、FCN 对 256×256 输入逐像素取 argmax，按有效像素累计 21 类混淆矩阵。U²-NetP 在 320×320 的首级显著图上归一化、以 0.5 二值化。 | 从 VOC2012 分割验证集按文件名哈希取 20 张。分割 mIoU 对有并集类别平均，忽略标签 255。显著目标用 VOC 各类前景的并集评分，是代理指标。[VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) · [U²-Net](https://github.com/xuebinqin/U-2-Net) |
| 特征匹配、线段检测 | SuperPoint 和 XFeat 读入 256×192 灰度图，输出最多 256 个 NMS 后的点及双线性采样描述子。M-LSD Tiny 处理 320×320 四通道输入，按中心得分选前 200 并解码端点。 | 本地带单应矩阵图像对 12 组，共 24 张；双向最近邻匹配后，以 3 px 重投影误差判断内点，同时给匹配数。吞吐量以单张特征提取计，配对和真值评分在计时之外。线段用 16 张合成图，以平均端点距离 ≤10 px 一对一匹配。[SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) · [XFeat](https://github.com/verlab/accelerated_features) · [M-LSD](https://github.com/navervision/mlsd) |
| 去噪、低照度、风格迁移 | DnCNN 预测噪声残差；FFDNet 采用 RGB 和固定 σ=25/255；Zero-DCE 回归像素亮度曲线；Mosaic 输出 RGB 风格化像素。 | 每个配置 10 张图。去噪噪声为 σ=25 的合成高斯噪声，低照度输入由干净图经 γ=2.2 生成，分别计算相对干净目标的 PSNR。风格迁移报告与原始 CPU ONNX 像素输出的平均绝对误差，未做主观美学评分。[KAIR](https://github.com/cszn/KAIR) · [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) · [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style) |
| 语音指令、手写数字 | DS-CNN、TC-ResNet 在板端从 16 kHz 单声道波形计算 40×98 的 log-Mel 谱；LeNet-5 接收 32×32 灰度数字图，输出类别。 | 两个指令模型均使用 8 类、1600 段训练和 96 段测试，说话人互斥。LeNet-5 用 sklearn Digits 分层划分，1347 张训练、80 张独立测试。三者均从训练划分学习，报告 Top-1。[语音数据](https://www.tensorflow.org/tutorials/audio/simple_audio) · [Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) |
| 时序和光谱 | 14 个公开数据集分别训练。包含时序残差、FCN、扩张卷积、多尺度卷积、深度可分卷积、MLP、收缩 LDA、岭分类器和 RBF 原型网络九种算法。板端从原始浮点序列开始，用训练集统计量归一化，再执行 RKNN 与分类解码。 | 保留各数据集官方 TRAIN/TEST 划分。训练分层最多 1200 条，测试不重复地按类轮取最多 60 条。报告的是这些训练实例在所列样本的准确率。例如 Epilepsy 数据集的标签是模拟动作。每个模型的训练量、类别、源文件哈希随资源清单提供。[UCR/UEA 数据档案](https://www.timeseriesclassification.com/) |
| 图文检索 | CLIP 图像塔处理 224×224 RGB，文本塔在板端将英文提示词编码为 20 个 BPE token。两个塔分别产生 512 维单位向量并进行余弦检索。 | 10 类 Imagenette 共 100 图，每类 3 个英文模板共 30 条文本。每个塔的表中指标使用 CPU ONNX 生成的对端图库。另行使用两侧板端实测向量计算双向检索结果。该测试为封闭的 10 类检索。[Rockchip CLIP](https://github.com/airockchip/rknn_model_zoo/tree/bad6c7334531becaf90a561988519b7bec34d0ab/examples/clip) · [CLIP](https://github.com/openai/CLIP) |

原有 39 个配置的逐模型处理与样本定义见应用目录的 `MULTITASK.md`。本轮在同一台设备的新固件上重新测量。设备仍采用自动调频。基准覆盖固定输入上的串行同步路径。摄像头采集、文件读取、评分、日志和 UI 刷新均在三段耗时之外。

## 两项转换复测

- SuperPoint 首次转换后检测头仍可用，描述子却全零。在 12 对图上平均每对只有 1 个双向匹配，3 px 内点率为 0%。对描述子缩放并用 1×1 卷积计算通道平方和后，等价图在 24 张输入上的主机最大绝对偏差为 1.2×10⁻⁷。板端重测平均每对得到 159.5 个匹配，内点率 93.63%。总表采用复测权重。
- Mosaic 首轮运行中断。将固定尺寸下的反射填充和实例归一化展开后，独立复测完成。10 张图的板端像素输出与原始 CPU 模型平均相差 7.96 个灰度级（0–255）；风格参考采用逐像素比较。原始尝试及替代权重的哈希随证据归档。
