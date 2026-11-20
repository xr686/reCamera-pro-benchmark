# 模型示例索引

共 270 个模型配置。点击模型名查看输入格式、获取权重、编译和安装命令。[运行指南](README.md)

## Bayer 去马赛克

- [Bilinear RGGB](bayer-linear/README.md) · `bayer-linear`
- [Residual demosaicing CNN](bayer-residual/README.md) · `bayer-residual`

## Car 传感器序列分类

- [Subsequence distance transform · Car](car-shapelets/README.md) · `car-shapelets`
- [Cosine basis ridge · Car](car-cosine/README.md) · `car-cosine`
- [Random convolution max/mean ridge · Car](car-randomconv/README.md) · `car-randomconv`

## JPEG 压缩等级识别

- [Block DCT energy probe · jpeg-level](jpeg-level-dct/README.md) · `jpeg-level-dct`
- [Directional derivative energy probe · jpeg-level](jpeg-level-derivative/README.md) · `jpeg-level-derivative`
- [Three-stage texture CNN · jpeg-level](jpeg-level-stackcnn/README.md) · `jpeg-level-stackcnn`

## JPEG 去压缩伪影

- [AR-CNN](jpeg-arcnn/README.md) · `jpeg-arcnn`
- [Dilated residual CNN](jpeg-dilated/README.md) · `jpeg-dilated`

## Wi-Fi 房间定位

- [Diagonal QDA · wifi](wifi-qda/README.md) · `wifi-qda`
- [Nearest subclass prototype · wifi](wifi-lvq/README.md) · `wifi-lvq`
- [Random ReLU features + ridge · wifi](wifi-randomnet/README.md) · `wifi-randomnet`

## 主目标框回归

- [Convolutional pyramid · voc-dominant](voc-dominant-plain/README.md) · `voc-dominant-plain`
- [Depthwise pyramid · voc-dominant](voc-dominant-depthwise/README.md) · `voc-dominant-depthwise`
- [Pretrained SqueezeNet pyramid · voc-dominant](voc-dominant-fire/README.md) · `voc-dominant-fire`

## 交通标志识别

- [ResNet-18 ridge probe · gts](gts-resprobe/README.md) · `gts-resprobe`
- [SqueezeNet ridge probe · gts](gts-squeezeprobe/README.md) · `gts-squeezeprobe`
- [Gabor filter bank + ridge · gts](gts-gabor/README.md) · `gts-gabor`
- [Multiscale CNN · gts](gts-branches/README.md) · `gts-branches`

## 人体姿态

- [BlazePose Landmarks](pose_estimation_mediapipe_2023mar/README.md) · `pose_estimation_mediapipe_2023mar`

## 人体检测

- [BlazePose Detector](person_detection_mediapipe_2023mar/README.md) · `person_detection_mediapipe_2023mar`

## 人像分割

- [PP-HumanSeg](human_segmentation_pphumanseg_2023mar/README.md) · `human_segmentation_pphumanseg_2023mar`

## 人脸关键点

- [MediaPipe FaceMesh](face_landmark_fp16/README.md) · `face_landmark_fp16`

## 人脸检测

- [YuNet](face_detection_yunet_2023mar/README.md) · `face_detection_yunet_2023mar`
- [RetinaFace MobileNet](RetinaFace_mobile320/README.md) · `RetinaFace_mobile320`
- [SCRFD-500M](scrfd500m_640_fp16/README.md) · `scrfd500m_640_fp16`

## 人脸特征

- [SFace](face_recognition_sface_2021dec/README.md) · `face_recognition_sface_2021dec`
- [ArcFace MobileFaceNet](arcface_mbf_fp16/README.md) · `arcface_mbf_fp16`

## 人脸质量

- [EDiffIQA Tiny](ediffiqa_tiny_jun2024/README.md) · `ediffiqa_tiny_jun2024`

## 人脸轮廓序列分类

- [Two-layer MLP · FaceAll](sig-faceall/README.md) · `sig-faceall`

## 人脸防伪

- [MiniFASNet V2](liveness_minifasnet_fp16/README.md) · `liveness_minifasnet_fp16`
- [MiniFASNet V1SE](liveness_minifasnet_v1se_fp16/README.md) · `liveness_minifasnet_v1se_fp16`

## 以文检图

- [CLIP ViT-B/32 文本塔](clip-text/README.md) · `clip-text`

## 低照度增强

- [Zero-DCE](zero-dce/README.md) · `zero-dce`

## 光源颜色估计

- [Gray World](wb-gray/README.md) · `wb-gray`
- [Shades of Gray (p=4)](wb-shades/README.md) · `wb-shades`
- [First-order Gray Edge](wb-edge/README.md) · `wb-edge`

## 制冷设备负荷分类

- [Subsequence distance transform · RefrigerationDevices](refrigerationdevices-shapelets/README.md) · `refrigerationdevices-shapelets`
- [Cosine basis ridge · RefrigerationDevices](refrigerationdevices-cosine/README.md) · `refrigerationdevices-cosine`
- [Random convolution max/mean ridge · RefrigerationDevices](refrigerationdevices-randomconv/README.md) · `refrigerationdevices-randomconv`

## 单目标跟踪

- [VitTrack](object_tracking_vittrack_2023sep/README.md) · `object_tracking_vittrack_2023sep`

## 单目相对深度

- [MiDaS v2.1 Small](depth_midas_v21_small_256_fp16/README.md) · `depth_midas_v21_small_256_fp16`

## 单音基频估计

- [YIN difference function](pitch-yin/README.md) · `pitch-yin`
- [Normalized autocorrelation](pitch-acf/README.md) · `pitch-acf`

## 双目视差估计

- [SAD block matching](stereo-sad/README.md) · `stereo-sad`
- [Zero-mean SAD](stereo-zsad/README.md) · `stereo-zsad`
- [Gradient block matching](stereo-gradient/README.md) · `stereo-gradient`
- [Normalized cross-correlation](stereo-ncc/README.md) · `stereo-ncc`

## 双音多频按键识别

- [Quadrature tone bank](dtmf-filter/README.md) · `dtmf-filter`
- [Spectral-bin MLP](dtmf-mlp/README.md) · `dtmf-mlp`

## 发动机噪声故障分类

- [Temporal FCN · FordA](sig-forda/README.md) · `sig-forda`

## 叶片轮廓树种识别

- [Subsequence distance transform · SwedishLeaf](swedishleaf-shapelets/README.md) · `swedishleaf-shapelets`
- [Cosine basis ridge · SwedishLeaf](swedishleaf-cosine/README.md) · `swedishleaf-cosine`
- [Random convolution max/mean ridge · SwedishLeaf](swedishleaf-randomconv/README.md) · `swedishleaf-randomconv`

## 咖啡品种光谱分类

- [Shrinkage LDA · Coffee](sig-coffee/README.md) · `sig-coffee`

## 图像修复

- [LaMa](inpainting_lama_2025jan/README.md) · `inpainting_lama_2025jan`

## 图像分类

- [GhostNet 1.0](ghostnet/README.md) · `ghostnet`
- [MobileOne-S0](mobileone-s0/README.md) · `mobileone-s0`
- [SqueezeNet 1.1](squeezenet1_1/README.md) · `squeezenet1_1`
- [MNASNet 0.5](mnasnet0_5/README.md) · `mnasnet0_5`
- [MobileNetV3 Small](mobilenet_v3_small/README.md) · `mobilenet_v3_small`
- [ShuffleNet V2 ×0.5](shufflenet_v2_x0_5/README.md) · `shufflenet_v2_x0_5`
- [RepVGG-A0](repvgg-a0/README.md) · `repvgg-a0`
- [EdgeNeXt-XXS](edgenext-xxs/README.md) · `edgenext-xxs`
- [DenseNet-121](densenet121/README.md) · `densenet121`
- [EfficientNet-B0](efficientnet_b0/README.md) · `efficientnet_b0`
- [RegNetX-400MF](regnet_x_400mf/README.md) · `regnet_x_400mf`
- [GoogLeNet](googlenet/README.md) · `googlenet`
- [ResNeXt-50 32×4d](resnext50_32x4d/README.md) · `resnext50_32x4d`
- [ConvNeXt Tiny](convnext_tiny/README.md) · `convnext_tiny`
- [Swin Transformer Tiny](swin_t/README.md) · `swin_t`
- [ViT-B/32](vit_b_32/README.md) · `vit_b_32`
- [ResNet-18](resnet18/README.md) · `resnet18`
- [ShuffleNet V2 x1.0](shufflenet-v2/README.md) · `shufflenet-v2`
- [MobileNet V2](mobilenet-v2/README.md) · `mobilenet-v2`
- [MixNet-S](mixnet-s/README.md) · `mixnet-s`
- [ReXNet-100](rexnet/README.md) · `rexnet`
- [DLA-34](dla34/README.md) · `dla34`
- [HRNet-W18 Small V2](hrnet-small/README.md) · `hrnet-small`
- [PP-LCNet ×1](lcnet/README.md) · `lcnet`
- [DPN-68](dpn68/README.md) · `dpn68`
- [EfficientFormer-L1](efficientformer/README.md) · `efficientformer`
- [MobileViT-XS](mobilevit-xs/README.md) · `mobilevit-xs`
- [poolformer_s12](poolformer/README.md) · `poolformer`
- [repghostnet_100](repghost/README.md) · `repghost`
- [ese_vovnet19b_dw](vovnet/README.md) · `vovnet`
- [fastvit_t8](fastvit/README.md) · `fastvit`
- [levit_128s](levit/README.md) · `levit`
- [skresnet18](skresnet/README.md) · `skresnet`
- [resnest14d](resnest/README.md) · `resnest`
- [coat_lite_tiny](coat-lite/README.md) · `coat-lite`
- [FasterNet-T0](fasternet-t0/README.md) · `fasternet-t0`
- [PVTv2-B0](pvtv2-b0/README.md) · `pvtv2-b0`
- [MobileNetV4 Conv Small](mobilenetv4-small/README.md) · `mobilenetv4-small`
- [GERNet-S](gernet-s/README.md) · `gernet-s`
- [FBNet-C](fbnetc/README.md) · `fbnetc`
- [HardCoRe-NAS-A](hardcorenas-a/README.md) · `hardcorenas-a`
- [RegNetY-400MF](regnety-004/README.md) · `regnety-004`
- [ConvNeXt Atto](convnext-atto/README.md) · `convnext-atto`
- [efficientvit_m0.r224_in1k](efficientvit-cga/README.md) · `efficientvit-cga`
- [starnet_s1.in1k](starnet/README.md) · `starnet`
- [swiftformer_xs.dist_in1k](swiftformer/README.md) · `swiftformer`
- [selecsls42b.in1k](selecsls/README.md) · `selecsls`
- [hgnet_tiny.ssld_in1k](hgnet/README.md) · `hgnet`
- [xcit_nano_12_p16_224.fb_dist_in1k](xcit/README.md) · `xcit`
- [focalnet_tiny_srf.ms_in1k](focalnet/README.md) · `focalnet`
- [xception41p.ra3_in1k](xception/README.md) · `xception`
- [convmixer_1024_20_ks9_p14.in1k](convmixer/README.md) · `convmixer`
- [davit_tiny.msft_in1k](davit/README.md) · `davit`
- [gmlp_s16_224.ra3_in1k](gmlp/README.md) · `gmlp`
- [gcvit_xxtiny.in1k](gcvit/README.md) · `gcvit`
- [hiera_tiny_224.mae_in1k_ft_in1k](hiera/README.md) · `hiera`
- [tnt_s_patch16_224.in1k](tnt/README.md) · `tnt`
- [maxvit_tiny_tf_224.in1k](maxvit/README.md) · `maxvit`
- [volo_d1_224.sail_in1k](volo/README.md) · `volo`

## 图像去噪

- [DnCNN σ25](dncnn25/README.md) · `dncnn25`
- [FFDNet Color σ25](ffdnet25/README.md) · `ffdnet25`

## 图像多标签识别

- [ResNet-18 ridge probe · voc-tags](voc-tags-resprobe/README.md) · `voc-tags-resprobe`
- [SqueezeNet ridge probe · voc-tags](voc-tags-squeezeprobe/README.md) · `voc-tags-squeezeprobe`
- [Gabor filter bank + ridge · voc-tags](voc-tags-gabor/README.md) · `voc-tags-gabor`
- [Multiscale CNN · voc-tags](voc-tags-branches/README.md) · `voc-tags-branches`

## 图像超分辨率

- [ESPCN ×3](espcn-x3/README.md) · `espcn-x3`

## 图文零样本分类

- [CLIP ViT-B/32 图像塔](clip-image/README.md) · `clip-image`

## 地震信号分类

- [Depthwise temporal CNN · Earthquakes](sig-earthquakes/README.md) · `sig-earthquakes`

## 场景语义分割

- [LR-ASPP MobileNetV3](lraspp_mobilenet_v3_large/README.md) · `lraspp_mobilenet_v3_large`
- [DeepLabV3 MobileNetV3](deeplabv3_mobilenet_v3_large/README.md) · `deeplabv3_mobilenet_v3_large`
- [FCN ResNet-50](fcn_resnet50/README.md) · `fcn_resnet50`
- [PP-LiteSeg](pp_liteseg_cityscapes/README.md) · `pp_liteseg_cityscapes`

## 声音事件分类

- [YAMNet](yamnet_3s/README.md) · `yamnet_3s`

## 失焦模糊等级识别

- [Block DCT energy probe · focus-level](focus-level-dct/README.md) · `focus-level-dct`
- [Directional derivative energy probe · focus-level](focus-level-derivative/README.md) · `focus-level-derivative`
- [Three-stage texture CNN · focus-level](focus-level-stackcnn/README.md) · `focus-level-stackcnn`

## 实例分割

- [YOLOv5s-seg](yolov5s-seg/README.md) · `yolov5s-seg`

## 宠物品种识别

- [ResNet-18 ridge probe · Pet](pet-breed-resnet/README.md) · `pet-breed-resnet`
- [ShuffleNetV2 ridge probe · Pet](pet-breed-shuffle/README.md) · `pet-breed-shuffle`
- [MobileNetV3 Small ridge probe · Pet](pet-breed-mobile/README.md) · `pet-breed-mobile`

## 家电屏幕类型识别

- [Subsequence distance transform · ScreenType](screentype-shapelets/README.md) · `screentype-shapelets`
- [Cosine basis ridge · ScreenType](screentype-cosine/README.md) · `screentype-cosine`
- [Random convolution max/mean ridge · ScreenType](screentype-randomconv/README.md) · `screentype-randomconv`

## 小麦籽粒品种识别

- [Diagonal QDA · seeds](seeds-qda/README.md) · `seeds-qda`
- [Nearest subclass prototype · seeds](seeds-lvq/README.md) · `seeds-lvq`
- [Random ReLU features + ridge · seeds](seeds-randomnet/README.md) · `seeds-randomnet`

## 局部图像块检索

- [TFeat](tfeat/README.md) · `tfeat`
- [HardNet](hardnet/README.md) · `hardnet`
- [SOSNet](sosnet/README.md) · `sosnet`

## 局部特征匹配

- [XFeat](xfeat/README.md) · `xfeat`
- [SuperPoint](superpoint/README.md) · `superpoint`

## 工业外观异常检测

- [PaDiM (8 features) · bottle](mvtec-bottle-padim/README.md) · `mvtec-bottle-padim`
- [Student–teacher CNN · bottle](mvtec-bottle-student/README.md) · `mvtec-bottle-student`
- [Convolutional autoencoder · bottle](mvtec-bottle-ae/README.md) · `mvtec-bottle-ae`
- [PaDiM (8 features) · grid](mvtec-grid-padim/README.md) · `mvtec-grid-padim`
- [Student–teacher CNN · grid](mvtec-grid-student/README.md) · `mvtec-grid-student`
- [Convolutional autoencoder · grid](mvtec-grid-ae/README.md) · `mvtec-grid-ae`

## 建筑采暖负荷回归

- [PLS regression · energy](energy-pls/README.md) · `energy-pls`
- [Extreme Learning Machine · energy](energy-elm/README.md) · `energy-elm`
- [Random Fourier ridge · energy](energy-rff/README.md) · `energy-rff`

## 心搏波形分类

- [Residual temporal CNN · ECG5000](sig-ecg5000/README.md) · `sig-ecg5000`

## 手写数字识别

- [LeNet-5](lenet-digits/README.md) · `lenet-digits`

## 手写轨迹识别

- [Multiscale Inception CNN · Handwriting](sig-handwriting/README.md) · `sig-handwriting`

## 手掌检测

- [BlazePalm](palm_detection_mediapipe_2023feb/README.md) · `palm_detection_mediapipe_2023feb`

## 手部关键点

- [MediaPipe Hand Landmarks](handpose_estimation_mediapipe_2023feb/README.md) · `handpose_estimation_mediapipe_2023feb`

## 手部动作轨迹分类

- [Subsequence distance transform · GunPoint](gunpoint-shapelets/README.md) · `gunpoint-shapelets`
- [Cosine basis ridge · GunPoint](gunpoint-cosine/README.md) · `gunpoint-cosine`
- [Random convolution max/mean ridge · GunPoint](gunpoint-randomconv/README.md) · `gunpoint-randomconv`

## 提示框内单实例分割

- [Convolutional pyramid · voc-prompt](voc-prompt-plain/README.md) · `voc-prompt-plain`
- [Depthwise pyramid · voc-prompt](voc-prompt-depthwise/README.md) · `voc-prompt-depthwise`
- [Pretrained SqueezeNet pyramid · voc-prompt](voc-prompt-fire/README.md) · `voc-prompt-fire`

## 文字定位

- [PP-OCRv3 EN Det](text_detection_en_ppocrv3_2023may/README.md) · `text_detection_en_ppocrv3_2023may`
- [PP-OCRv3 CN Det](text_detection_cn_ppocrv3_2023may/README.md) · `text_detection_cn_ppocrv3_2023may`

## 文字识别

- [CRNN EN](text_recognition_CRNN_EN_2021sep/README.md) · `text_recognition_CRNN_EN_2021sep`

## 旋转目标检测

- [GGHL](gghl-dota/README.md) · `gghl-dota`

## 无提示宠物前景分割

- [Convolutional pyramid · Pet](pet-mask-plain/README.md) · `pet-mask-plain`
- [Depthwise pyramid · Pet](pet-mask-depthwise/README.md) · `pet-mask-depthwise`
- [Pretrained SqueezeNet pyramid · Pet](pet-mask-fire/README.md) · `pet-mask-fire`

## 昆虫翅振声分类

- [Temporal FCN · InsectWingbeatSound](sig-insectwingbeatsound/README.md) · `sig-insectwingbeatsound`

## 显著目标提取

- [U²-NetP](u2netp/README.md) · `u2netp`

## 晶圆工艺异常分类

- [Dilated temporal CNN · Wafer](sig-wafer/README.md) · `sig-wafer`

## 服饰品类识别

- [Network in Network](fashion-nin/README.md) · `fashion-nin`
- [Highway MLP](fashion-highway/README.md) · `fashion-highway`
- [Patch MLP-Mixer](fashion-mixer/README.md) · `fashion-mixer`

## 果泥光谱鉴别

- [Ridge classifier · Strawberry](sig-strawberry/README.md) · `sig-strawberry`

## 模拟动作识别

- [Residual temporal CNN · Epilepsy](sig-epilepsy/README.md) · `sig-epilepsy`

## 模拟过程瞬态分类

- [Subsequence distance transform · Trace](trace-shapelets/README.md) · `trace-shapelets`
- [Cosine basis ridge · Trace](trace-cosine/README.md) · `trace-cosine`
- [Random convolution max/mean ridge · Trace](trace-randomconv/README.md) · `trace-randomconv`

## 橄榄油产地光谱分类

- [Subsequence distance transform · OliveOil](oliveoil-shapelets/README.md) · `oliveoil-shapelets`
- [Cosine basis ridge · OliveOil](oliveoil-cosine/README.md) · `oliveoil-cosine`
- [Random convolution max/mean ridge · OliveOil](oliveoil-randomconv/README.md) · `oliveoil-randomconv`

## 火腿蛋白图谱分类

- [Subsequence distance transform · Ham](ham-shapelets/README.md) · `ham-shapelets`
- [Cosine basis ridge · Ham](ham-cosine/README.md) · `ham-cosine`
- [Random convolution max/mean ridge · Ham](ham-randomconv/README.md) · `ham-randomconv`

## 片段语音存在性分类

- [Log-Mel shrinkage LDA](presence-lda/README.md) · `presence-lda`
- [Spectrogram CNN](presence-cnn/README.md) · `presence-cnn`

## 牛肉光谱类别识别

- [Subsequence distance transform · Beef](beef-shapelets/README.md) · `beef-shapelets`
- [Cosine basis ridge · Beef](beef-cosine/README.md) · `beef-cosine`
- [Random convolution max/mean ridge · Beef](beef-randomconv/README.md) · `beef-randomconv`

## 用电曲线季节分类

- [Two-layer MLP · ItalyPowerDemand](sig-italypowerdemand/README.md) · `sig-italypowerdemand`

## 用电设备识别

- [Multiscale Inception CNN · ElectricDevices](sig-electricdevices/README.md) · `sig-electricdevices`

## 电网稳定性分类

- [Diagonal QDA · gridstab](gridstab-qda/README.md) · `gridstab-qda`
- [Nearest subclass prototype · gridstab](gridstab-lvq/README.md) · `gridstab-lvq`
- [Random ReLU features + ridge · gridstab](gridstab-randomnet/README.md) · `gridstab-randomnet`

## 目标检测

- [NanoDet-Plus-m](object_detection_nanodet_2022nov/README.md) · `object_detection_nanodet_2022nov`
- [YOLOX-s](object_detection_yolox_2022nov/README.md) · `object_detection_yolox_2022nov`
- [YOLOv5s-ReLU](yolov5-relu/README.md) · `yolov5-relu`
- [YOLOv10n](yolov10n/README.md) · `yolov10n`

## 短信垃圾信息识别

- [Multinomial Naive Bayes](sms-nb/README.md) · `sms-nb`
- [Linear SVM](sms-svm/README.md) · `sms-svm`
- [Hashed n-gram MLP](sms-mlp/README.md) · `sms-mlp`

## 稠密光流估计

- [Horn–Schunck (12 iterations)](flow-hs/README.md) · `flow-hs`
- [Dense Lucas–Kanade (7×7)](flow-lk/README.md) · `flow-lk`
- [2-D block matching (49 offsets)](flow-sad/README.md) · `flow-sad`

## 箭头轮廓分类

- [Subsequence distance transform · ArrowHead](arrowhead-shapelets/README.md) · `arrowhead-shapelets`
- [Cosine basis ridge · ArrowHead](arrowhead-cosine/README.md) · `arrowhead-cosine`
- [Random convolution max/mean ridge · ArrowHead](arrowhead-randomconv/README.md) · `arrowhead-randomconv`

## 红酒感官评分回归

- [PLS regression · wine](wine-pls/README.md) · `wine-pls`
- [Extreme Learning Machine · wine](wine-elm/README.md) · `wine-elm`
- [Random Fourier ridge · wine](wine-rff/README.md) · `wine-rff`

## 线段检测

- [M-LSD Tiny](mlsd/README.md) · `mlsd`

## 翼型气动噪声回归

- [PLS regression · airfoil](airfoil-pls/README.md) · `airfoil-pls`
- [Extreme Learning Machine · airfoil](airfoil-elm/README.md) · `airfoil-elm`
- [Random Fourier ridge · airfoil](airfoil-rff/README.md) · `airfoil-rff`

## 肉类光谱分类

- [RBF prototype network · Meat](sig-meat/README.md) · `sig-meat`

## 腕部活动识别

- [Depthwise temporal CNN · BasicMotions](sig-basicmotions/README.md) · `sig-basicmotions`

## 艺术风格迁移

- [Mosaic TransformerNet](mosaic/README.md) · `mosaic`

## 花卉品类识别

- [ResNet-18 ridge probe · flowers](flowers-resprobe/README.md) · `flowers-resprobe`
- [SqueezeNet ridge probe · flowers](flowers-squeezeprobe/README.md) · `flowers-squeezeprobe`
- [Gabor filter bank + ridge · flowers](flowers-gabor/README.md) · `flowers-gabor`
- [Multiscale CNN · flowers](flowers-branches/README.md) · `flowers-branches`

## 英文短评情感分类

- [Bernoulli Naive Bayes · sent](sent-bnb/README.md) · `sent-bnb`
- [N-gram ridge classifier · sent](sent-ridge/README.md) · `sent-ridge`
- [Averaged linear perceptron · sent](sent-perceptron/README.md) · `sent-perceptron`

## 英文问题类型识别

- [Bernoulli Naive Bayes · trec](trec-bnb/README.md) · `trec-bnb`
- [N-gram ridge classifier · trec](trec-ridge/README.md) · `trec-ridge`
- [Averaged linear perceptron · trec](trec-perceptron/README.md) · `trec-perceptron`

## 葡萄干品种识别

- [Diagonal QDA · raisins](raisins-qda/README.md) · `raisins-qda`
- [Nearest subclass prototype · raisins](raisins-lvq/README.md) · `raisins-lvq`
- [Random ReLU features + ridge · raisins](raisins-randomnet/README.md) · `raisins-randomnet`

## 行人重识别

- [Youtu ReID](person_reid_youtu_2021nov/README.md) · `person_reid_youtu_2021nov`

## 表情识别

- [MobileFaceNet FER](facial_expression_recognition_mobilefacenet_2022july/README.md) · `facial_expression_recognition_mobilefacenet_2022july`

## 语音指令识别

- [DS-CNN](ds-kws/README.md) · `ds-kws`
- [TC-ResNet](tc-kws/README.md) · `tc-kws`

## 语音转文字

- [NeMo Conformer Small CTC](nemo-ctc-6s/README.md) · `nemo-ctc-6s`
- [NeMo Citrinet-512](citrinet-6s/README.md) · `citrinet-6s`
- [Wav2Vec2 Base 960h · 6s](wav2vec2-6s/README.md) · `wav2vec2-6s`

## 说话人检索

- [Compact x-vector TDNN](speaker-tdnn/README.md) · `speaker-tdnn`
- [Residual speaker CNN](speaker-rescnn/README.md) · `speaker-rescnn`

## 车牌四角检测

- [LPD-YuNet](license_plate_detection_lpd_yunet_2023mar/README.md) · `license_plate_detection_lpd_yunet_2023mar`

## 车牌字符识别

- [LPRNet](lprnet/README.md) · `lprnet`

## 边缘检测

- [DexiNed](edge_detection_dexined_2024sep/README.md) · `edge_detection_dexined_2024sep`

## 运动去模糊

- [NAFNet](deblurring_nafnet_2025may/README.md) · `deblurring_nafnet_2025may`

## 运动模糊方向识别

- [Block DCT energy probe · blur-angle](blur-angle-dct/README.md) · `blur-angle-dct`
- [Directional derivative energy probe · blur-angle](blur-angle-derivative/README.md) · `blur-angle-derivative`
- [Three-stage texture CNN · blur-angle](blur-angle-stackcnn/README.md) · `blur-angle-stackcnn`

## 近重复图像检索

- [DCT perceptual hash](hash-dct/README.md) · `hash-dct`
- [Difference hash](hash-difference/README.md) · `hash-difference`
- [Low-frequency DCT descriptor](hash-spectrum/README.md) · `hash-spectrum`

## 遥感土地利用分类

- [ResNet-18 ridge probe · eurosat](eurosat-resprobe/README.md) · `eurosat-resprobe`
- [SqueezeNet ridge probe · eurosat](eurosat-squeezeprobe/README.md) · `eurosat-squeezeprobe`
- [Gabor filter bank + ridge · eurosat](eurosat-gabor/README.md) · `eurosat-gabor`
- [Multiscale CNN · eurosat](eurosat-branches/README.md) · `eurosat-branches`

## 飞机轮廓机型识别

- [Subsequence distance transform · Plane](plane-shapelets/README.md) · `plane-shapelets`
- [Cosine basis ridge · Plane](plane-cosine/README.md) · `plane-cosine`
- [Random convolution max/mean ridge · Plane](plane-randomconv/README.md) · `plane-randomconv`

## 高斯噪声强度识别

- [Block DCT energy probe · noise-level](noise-level-dct/README.md) · `noise-level-dct`
- [Directional derivative energy probe · noise-level](noise-level-derivative/README.md) · `noise-level-derivative`
- [Three-stage texture CNN · noise-level](noise-level-stackcnn/README.md) · `noise-level-stackcnn`

## 鱼类轮廓识别

- [Subsequence distance transform · Fish](fish-shapelets/README.md) · `fish-shapelets`
- [Cosine basis ridge · Fish](fish-cosine/README.md) · `fish-cosine`
- [Random convolution max/mean ridge · Fish](fish-randomconv/README.md) · `fish-randomconv`
