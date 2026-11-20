## 样本与质量指标

| 测试组 | 数据、训练和评分口径 |
|---|---|
| ImageNet 分类 | 8 个 timm 预训练模型使用上一轮相同的 100 张 Imagenette 验证图。每类 10 张，从完整的 1000 类输出中计算 Top-1/Top-5。MobileViT-XS 的输入为 256×256。权重名称、插值方式、裁剪大小和归一化参数随清单提供。[timm](https://github.com/huggingface/pytorch-image-models) · [Imagenette](https://github.com/fastai/imagenette) |
| 目标检测 | 在 VOC2012 Main 验证集按文件名哈希选择 32 张原图，直接缩放到 640×640。将 VOC 20 类映射到 COCO 类别，保留 difficult 标注，按类别匹配并累计 F1@IoU0.5，分数阈值 0.30。YOLOv10n 采用 one-to-one 检测头，YOLOv5s-ReLU 使用锚框和 NMS。指标为这批图片的 F1。[VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) · [Rockchip YOLOv10](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov10) · [Rockchip YOLOv5](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov5) |
| 双目视差 | 四个 7×7 块匹配代价在 RKNN 上计算 0 至 31 像素的 32 层代价体。CPU 对代价取最小值。从 Motorcycle、Cones、Teddy 三个有真值视差的场景共取 18 个部分重叠的裁剪，每次输入一对 160×96 灰度图。按横向缩放比调整真值，对搜索范围内、两侧都有对应内容的有限真值像素累计绝对误差和 1 px 内比例。[Middlebury 双目数据](https://vision.middlebury.edu/stereo/data/) · [Motorcycle 数据说明](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.stereo_motorcycle) |
| 图像块检索 | 使用 TFeat、HardNet 和 SOSNet 的公开预训练权重。由上一轮 12 对图像和其已知单应矩阵构造 96 对 32×32 灰度块，用矩阵把第二幅图重采样到第一幅图的坐标系。共提取 192 个描述子，将每个查询块与全部 96 个图库块匹配。这衡量给定对齐条件下的描述子检索。跨图库打分在计时之外。[TFeat](https://github.com/vbalnt/tfeat) · [HardNet](https://github.com/DagnyT/hardnet) · [SOSNet](https://github.com/yuruntian/SOSNet) |
| Bayer 与 JPEG 图像复原 | 从 Imagenette 按源图划分 200 张训练图、20 张独立测试图，测试图分辨率 96×96。去马赛克从 RGB 合成 RGGB，比较双线性插值和学习残差的 CNN，覆盖整图计算 RGB PSNR。JPEG 使用质量因子 15，比较本次训练的三通道 AR-CNN 与扩张卷积残差网络。直接使用 JPEG 解码图在相同样本上为 25.711 dB。Bayer 样本模拟彩色滤光阵列采样，尚未覆盖传感器黑电平和 RAW 噪声。[AR-CNN](https://arxiv.org/abs/1504.00274) |
| 工业外观异常 | 从 MVTec AD 的 bottle 和 grid 各取 72 张官方训练集正常图。所有模型按类别独立拟合。PaDiM 从两个预训练 ResNet-18 特征层抽取 8 个固定通道，在每个网格拟合正则化协方差与均值。Student–teacher 以轻量 CNN 拟合教师特征，卷积自编码器学习图像重建。使用异常图最高 1% 网格均值作为图像分数。瓶子测试 20 张正常、24 张缺陷，网格测试 20 张正常、40 张缺陷，报告图像级 AUROC。数据缩放为 128×128，缺陷尺寸会影响结果。[MVTec AD 与数据使用条件](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) · [镜像与样本索引](https://huggingface.co/datasets/Voxel51/mvtec-ad) · [PaDiM](https://arxiv.org/abs/2011.08785) |
| 服饰分类 | Network in Network、Highway MLP 和小型 Patch MLP-Mixer 使用相同的 Fashion-MNIST 官方训练集每类 1000 张，训练 8 轮。评估从官方测试集每类取 12 张，共 120 张，计算 Top-1。[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) |
| 说话人检索 | 选取 Mini Speech Commands 中的 120 名训练说话人、2132 段音频，训练两个紧凑语音嵌入网络。另选 24 名训练未见的说话人，每人 4 个不同口令，1 段注册、3 段查询。96 段分别提取特征，用 72 个查询对 24 人图库计算 Rank-1。输入从一秒 PCM 开始计时，图库间余弦比较在计时之外。[Speech Commands](https://www.tensorflow.org/tutorials/audio/simple_audio) |
| 语音存在性 | 1 秒片段级二分类，训练集 384 段语音、384 段环境声。测试用来自不同说话人的 64 段语音和 ESC-50 第 5 折 32 个独立录音中的 64 段环境声。环境声避开人物非语音声音类别。报告均衡片段上的正确率，不涉及语音起止点定位。[ESC-50](https://github.com/karolpiczak/ESC-50) |
| 英文短信分类 | UCI SMS Spam Collection 按归一化全文去重后分层切分，测试各取 64 条正常、64 条垃圾短信。比较多项式朴素贝叶斯、线性 SVM 和 MLP。统一从短信文本开始，单词和双词组映射到 512 个计数特征，报告准确率和垃圾类 F1。[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| 表格回归 | 三种回归方法分别训练于三个 UCI 数据集，共九份权重。相同输入行在同一划分，标准化只使用训练集。翼型、红酒、建筑的训练行数为 1127、1203、576，每个测试集取 96 个互异输入。翼型输出声压级，MAE 单位 dB；红酒输出感官评分，单位为分；建筑输出数据集的采暖负荷 Y1，以其原标度报告 MAE。[Airfoil](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) · [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) · [Energy Efficiency](https://archive.ics.uci.edu/dataset/242/energy+efficiency) |

### 板端与主机的差异

归一化互相关（NCC）在低纹理区域要用两个接近的数计算方差，FP16 对该步骤较敏感。初次转换在双目样本上的板端 MAE 为 1.941 px、1 px 内比例为 71.53%。将强度中心化并缩放后，独占复测为 1.457 px、79.64%，同图 CPU ONNX 为 1.435 px、81.64%。总表取复测权重，证据保留两次结果。

PaDiM 在 bottle 上的图像 AUROC 为 0.931，在 grid 上为 0.604。两个类别均用各自 72 张正常图拟合，选择工业检测模型时应按实际产品和缺陷类型核对质量。图像块检索的高分也与已提供的几何对齐有关。
