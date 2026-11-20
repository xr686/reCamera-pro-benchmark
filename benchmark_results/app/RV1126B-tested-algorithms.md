# reCamera Pro 已测算法总览

RV1126B 固件 V1.3.2 的 NPU 基准累计完成 270 个不同 ID 的配置，覆盖 100 个任务标签。下表按既定骨干口径列出 50 个 ImageNet 分类架构族。其他任务的表格保留实际运行的网络、模型头或经典算法名称；同一实现可出现在多个任务中。

## 分类架构族

历史分类评测的 43 个配置合并为 34 族，最近一次新增 16 族。代表模型以真机索引为准。ResNet 包括按本项目口径合并的 ResNeXt、SK-ResNet 和 ResNeSt；在分割、定位或品种识别中复用该骨干也不增加分类架构数。

| 架构族 | 实测代表 | 同族其他分类配置 ID |
|---|---|---|
| ResNet | ResNet-18 | resnext50_32x4d、skresnet、resnest |
| MobileNet | MobileNetV3 Small | mobilenet-v2、mobilenetv4-small |
| ShuffleNet | ShuffleNet V2 ×0.5 | shufflenet-v2 |
| GhostNet | GhostNet 1.0 | repghost |
| RegNet | RegNetX-400MF | regnety-004 |
| ConvNeXt | ConvNeXt Tiny | convnext-atto |
| SqueezeNet | SqueezeNet 1.1 | — |
| MNASNet | MNASNet 0.5 | — |
| MobileOne | MobileOne-S0 | — |
| RepVGG | RepVGG-A0 | — |
| EdgeNeXt | EdgeNeXt-XXS | — |
| DenseNet | DenseNet-121 | — |
| EfficientNet | EfficientNet-B0 | — |
| Inception | GoogLeNet | — |
| Swin Transformer | Swin Transformer Tiny | — |
| ViT | ViT-B/32 | — |
| MixNet | MixNet-S | — |
| ReXNet | ReXNet-100 | — |
| DLA | DLA-34 | — |
| HRNet | HRNet-W18 Small V2 | — |
| PP-LCNet | PP-LCNet ×1 | — |
| DPN | DPN-68 | — |
| EfficientFormer | EfficientFormer-L1 | — |
| MobileViT | MobileViT-XS | — |
| PoolFormer | poolformer_s12 | — |
| VoVNet | ese_vovnet19b_dw | — |
| FastViT | fastvit_t8 | — |
| LeViT | levit_128s | — |
| CoaT | coat_lite_tiny | — |
| FasterNet | FasterNet-T0 | — |
| PVT | PVTv2-B0 | — |
| GERNet | GERNet-S | — |
| FBNet | FBNet-C | — |
| HardCoRe-NAS | HardCoRe-NAS-A | — |
| EfficientViT (CGA) | efficientvit_m0.r224_in1k | — |
| StarNet | starnet_s1.in1k | — |
| SwiftFormer | swiftformer_xs.dist_in1k | — |
| SelecSLS | selecsls42b.in1k | — |
| PP-HGNet | hgnet_tiny.ssld_in1k | — |
| XCiT | xcit_nano_12_p16_224.fb_dist_in1k | — |
| FocalNet | focalnet_tiny_srf.ms_in1k | — |
| Xception | xception41p.ra3_in1k | — |
| ConvMixer | convmixer_1024_20_ks9_p14.in1k | — |
| DaViT | davit_tiny.msft_in1k | — |
| gMLP | gmlp_s16_224.ra3_in1k | — |
| GC ViT | gcvit_xxtiny.in1k | — |
| Hiera | hiera_tiny_224.mae_in1k_ft_in1k | — |
| TNT | tnt_s_patch16_224.in1k | — |
| MaxViT | maxvit_tiny_tf_224.in1k | — |
| VOLO | volo_d1_224.sail_in1k | — |

[分类架构索引 CSV](RV1126B-ImageNet-family-index.csv) · [最近 16 个架构的速度与精度](../../apps/npu-benchmark/ARCHITECTURES-1.3.2.md)

## 其余 99 个任务标签的实测算法

同一算法在多个数据集上重新训练时，下表保留任务对应关系。输入、精度、分阶段时延和有效样本数均收录在[270 项逐配置 CSV](measurements.csv)。

| 任务 | 已测模型或算法 |
|---|---|
| Bayer 去马赛克 | Bilinear RGGB、Residual demosaicing CNN |
| Car 传感器序列分类 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |
| JPEG 压缩等级识别 | Block DCT energy probe、Directional derivative energy probe、Three-stage texture CNN |
| JPEG 去压缩伪影 | AR-CNN、Dilated residual CNN |
| Wi-Fi 房间定位 | Diagonal QDA、Nearest subclass prototype、Random ReLU features + ridge |
| 主目标框回归 | Convolutional pyramid、Depthwise pyramid、Pretrained SqueezeNet pyramid |
| 交通标志识别 | ResNet-18 ridge probe、SqueezeNet ridge probe、Gabor filter bank + ridge、Multiscale CNN |
| 人体姿态 | BlazePose Landmarks |
| 人体检测 | BlazePose Detector |
| 人像分割 | PP-HumanSeg |
| 人脸关键点 | MediaPipe FaceMesh |
| 人脸检测 | YuNet、RetinaFace MobileNet、SCRFD-500M |
| 人脸特征 | SFace、ArcFace MobileFaceNet |
| 人脸质量 | EDiffIQA Tiny |
| 人脸轮廓序列分类 | Two-layer MLP |
| 人脸防伪 | MiniFASNet V2、MiniFASNet V1SE |
| 以文检图 | CLIP ViT-B/32 文本塔 |
| 低照度增强 | Zero-DCE |
| 光源颜色估计 | Gray World、Shades of Gray (p=4)、First-order Gray Edge |
| 制冷设备负荷分类 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |
| 单目标跟踪 | VitTrack |
| 单目相对深度 | MiDaS v2.1 Small |
| 单音基频估计 | YIN difference function、Normalized autocorrelation |
| 双目视差估计 | SAD block matching、Zero-mean SAD、Gradient block matching、Normalized cross-correlation |
| 双音多频按键识别 | Quadrature tone bank、Spectral-bin MLP |
| 发动机噪声故障分类 | Temporal FCN |
| 叶片轮廓树种识别 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |
| 咖啡品种光谱分类 | Shrinkage LDA |
| 图像修复 | LaMa |
| 图像去噪 | DnCNN σ25、FFDNet Color σ25 |
| 图像多标签识别 | ResNet-18 ridge probe、SqueezeNet ridge probe、Gabor filter bank + ridge、Multiscale CNN |
| 图像超分辨率 | ESPCN ×3 |
| 图文零样本分类 | CLIP ViT-B/32 图像塔 |
| 地震信号分类 | Depthwise temporal CNN |
| 场景语义分割 | LR-ASPP MobileNetV3、DeepLabV3 MobileNetV3、FCN ResNet-50、PP-LiteSeg |
| 声音事件分类 | YAMNet |
| 失焦模糊等级识别 | Block DCT energy probe、Directional derivative energy probe、Three-stage texture CNN |
| 实例分割 | YOLOv5s-seg |
| 宠物品种识别 | ResNet-18 ridge probe、ShuffleNetV2 ridge probe、MobileNetV3 Small ridge probe |
| 家电屏幕类型识别 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |
| 小麦籽粒品种识别 | Diagonal QDA、Nearest subclass prototype、Random ReLU features + ridge |
| 局部图像块检索 | TFeat、HardNet、SOSNet |
| 局部特征匹配 | XFeat、SuperPoint |
| 工业外观异常检测 | PaDiM (8 features)、Student–teacher CNN、Convolutional autoencoder |
| 建筑采暖负荷回归 | PLS regression、Extreme Learning Machine、Random Fourier ridge |
| 心搏波形分类 | Residual temporal CNN |
| 手写数字识别 | LeNet-5 |
| 手写轨迹识别 | Multiscale Inception CNN |
| 手掌检测 | BlazePalm |
| 手部关键点 | MediaPipe Hand Landmarks |
| 手部动作轨迹分类 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |
| 提示框内单实例分割 | Convolutional pyramid、Depthwise pyramid、Pretrained SqueezeNet pyramid |
| 文字定位 | PP-OCRv3 EN Det、PP-OCRv3 CN Det |
| 文字识别 | CRNN EN |
| 旋转目标检测 | GGHL |
| 无提示宠物前景分割 | Convolutional pyramid、Depthwise pyramid、Pretrained SqueezeNet pyramid |
| 昆虫翅振声分类 | Temporal FCN |
| 显著目标提取 | U²-NetP |
| 晶圆工艺异常分类 | Dilated temporal CNN |
| 服饰品类识别 | Network in Network、Highway MLP、Patch MLP-Mixer |
| 果泥光谱鉴别 | Ridge classifier |
| 模拟动作识别 | Residual temporal CNN |
| 模拟过程瞬态分类 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |
| 橄榄油产地光谱分类 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |
| 火腿蛋白图谱分类 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |
| 片段语音存在性分类 | Log-Mel shrinkage LDA、Spectrogram CNN |
| 牛肉光谱类别识别 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |
| 用电曲线季节分类 | Two-layer MLP |
| 用电设备识别 | Multiscale Inception CNN |
| 电网稳定性分类 | Diagonal QDA、Nearest subclass prototype、Random ReLU features + ridge |
| 目标检测 | NanoDet-Plus-m、YOLOX-s、YOLOv5s-ReLU、YOLOv10n |
| 短信垃圾信息识别 | Multinomial Naive Bayes、Linear SVM、Hashed n-gram MLP |
| 稠密光流估计 | Horn–Schunck (12 iterations)、Dense Lucas–Kanade (7×7)、2-D block matching (49 offsets) |
| 箭头轮廓分类 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |
| 红酒感官评分回归 | PLS regression、Extreme Learning Machine、Random Fourier ridge |
| 线段检测 | M-LSD Tiny |
| 翼型气动噪声回归 | PLS regression、Extreme Learning Machine、Random Fourier ridge |
| 肉类光谱分类 | RBF prototype network |
| 腕部活动识别 | Depthwise temporal CNN |
| 艺术风格迁移 | Mosaic TransformerNet |
| 花卉品类识别 | ResNet-18 ridge probe、SqueezeNet ridge probe、Gabor filter bank + ridge、Multiscale CNN |
| 英文短评情感分类 | Bernoulli Naive Bayes、N-gram ridge classifier、Averaged linear perceptron |
| 英文问题类型识别 | Bernoulli Naive Bayes、N-gram ridge classifier、Averaged linear perceptron |
| 葡萄干品种识别 | Diagonal QDA、Nearest subclass prototype、Random ReLU features + ridge |
| 行人重识别 | Youtu ReID |
| 表情识别 | MobileFaceNet FER |
| 语音指令识别 | DS-CNN、TC-ResNet |
| 语音转文字 | NeMo Conformer Small CTC、NeMo Citrinet-512、Wav2Vec2 Base 960h |
| 说话人检索 | Compact x-vector TDNN、Residual speaker CNN |
| 车牌四角检测 | LPD-YuNet |
| 车牌字符识别 | LPRNet |
| 边缘检测 | DexiNed |
| 运动去模糊 | NAFNet |
| 运动模糊方向识别 | Block DCT energy probe、Directional derivative energy probe、Three-stage texture CNN |
| 近重复图像检索 | DCT perceptual hash、Difference hash、Low-frequency DCT descriptor |
| 遥感土地利用分类 | ResNet-18 ridge probe、SqueezeNet ridge probe、Gabor filter bank + ridge、Multiscale CNN |
| 飞机轮廓机型识别 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |
| 高斯噪声强度识别 | Block DCT energy probe、Directional derivative energy probe、Three-stage texture CNN |
| 鱼类轮廓识别 | Subsequence distance transform、Cosine basis ridge、Random convolution max/mean ridge |

## 视觉语言模型专项

场景描述应用另行在真机验证了 SmolVLM-256M-Instruct 与 InternVL3-1B。这两项独立于上述 270 个配置。[VLM 应用测量](vlm.md)
