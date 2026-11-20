## 如何解读任务质量

| 测试输入 | 标签、指标及适用范围 |
|---|---|
| Imagenette、COCO8、COCO8-Pose | 分类为 10 类各 30 张，预测候选仍为 1000 类。普通检测为 4 张验证图片的全部标注框。F1 按类别和 IoU≥0.5 一对一匹配后汇总。BlazePose Detector 的输出框包围面部，额外回归髋中心。人体定位用双髋有标注的 ROI，首选检测髋中心到真值距离 ≤0.10 计命中。姿态从标注裁出 9 个 ROI，将 BlazePose 的 33 点映射到 COCO 17 点。PCK 用归一化平面距离 ≤0.10、标注可见性大于零的关节计数。 |
| HRSC2016、VOC2012、CamVid | GGHL 使用 8 张 HRSC 测试图，船舶四角框与 DOTA ship 类匹配，多边形 IoU≥0.5。VOC 的 12 张图有人像及实例 ID，前景 IoU 按像素汇总，实例 F1 逐人掩膜匹配，忽略边界 255。PP-LiteSeg 在 CamVid 10 张街景中对可映射的 10 类求每幅图中各类 IoU，再按有效图像-类别组合平均。 |
| Middlebury Motorcycle | 一组有视差真值的立体图，仅向模型输入左图。将预测对齐到真值分辨率后，在每四个像素抽取的有效位置计算 Pearson 相关系数，反映相对深度排序趋势。 |
| 剪刀石头布、真实跟踪视频、Market-1501 | 三手势各 5 图。手掌项统计正例中检出任一框的比例，缺少框标注和负例。关键点项用四指伸展规则计算三手势正确率，缺少逐关节真值。跟踪在 David、Dudek、FaceOcc2 各取 6 对间隔 5 帧的图像，由前帧标注定位模板和搜索区域，下一帧以真值框计算 IoU，是短时帧对测试。ReID 使用 20 个身份，每人一张查询图、一张跨摄像头图库图，以归一化特征余弦检索 Rank-1。 |
| 人脸检测、表情、特征、质量及防伪 | 检测使用 LZX 验证集 16 图。表情使用按文件夹标注的 CK+ 衍生图片。特征检索以文件名主体 ID 为准，留一法只对有同身份图库的查询计分，同源表情图可能外观非常相近。质量项比较 12 张清晰图分别与高斯模糊副本的分数，仅反映对这类退化的敏感性。FaceMesh 比较 7 个公共点，以眼间距归一化误差，参照点来自 dlib 发布的模型生成标注。防伪只含 Silent-Face 的 1 张真脸和 2 张攻击示例。 |
| 车牌和英文文字 | 四角检测使用 5 张 CCPD 示例及人工复核四角，字符识别使用其透视校正裁片和 1 张 Rockchip 示例，整串完全相同才算命中。两套 PP-OCRv3 检测权重均在相同的 12 张合成英文行图评分；CRNN 在对应 12 张单词裁片上评分。这组数据只验证英文，定位和识别分开计时。 |
| 图像复原、边缘 | 超分辨率以 8 张图的 Y 通道降采样输入，按 ×3 重建后的 Y-PSNR 评分，边缘裁 3 像素。去模糊使用已知的 9 像素水平运动模糊和干净目标。修复用固定矩形遮罩合成缺损，只在遮挡区计算 PSNR。两者各 8 张，PSNR 为各图 dB 均值。DexiNed 使用 4 张 BSDS500，≥40% 标注者同意的边界作参照，固定阈值和 ±2 像素容差。 |
| 环境声与英文 ASR | YAMNet 在 ESC-50 第 5 折选取 8 个可映射到 AudioSet 的类别，每类 2 段，截取每段中央 3 秒。以 521 类首选标签与单个映射标签严格比对。语音识别共享 9 段音频、107 个参考词，使用 Rockchip 示例及 LibriSpeech 子集。WER = 总替换、删除和插入词数 / 参考词数。采用离线固定窗、贪心解码。 |

数据与权重入口：[OpenCV Zoo 固定版本](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7)、[Rockchip RKNN Model Zoo 固定版本](https://github.com/airockchip/rknn_model_zoo/tree/bad6c7334531becaf90a561988519b7bec34d0ab)、[GGHL](https://github.com/Shank2358/GGHL)、[Wav2Vec2 Base 960h](https://huggingface.co/facebook/wav2vec2-base-960h)、[sherpa-onnx 的 NeMo CTC 模型](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html)。

数据及标注说明：[COCO8](https://docs.ultralytics.com/datasets/detect/coco8/)、[VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)、[CamVid](https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid)、[CCPD](https://github.com/detectRecog/CCPD)、[OpenCV 跟踪样本](https://github.com/opencv/opencv_extra/tree/ebf9366b3845efe1e54bf2440cf238fd68905bab/testdata/cv/tracking)、[dlib 测试标注](https://github.com/davisking/dlib/tree/master/examples/faces)、[Silent-Face 样本](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)、[BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)、[ESC-50](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6)、[LibriSpeech](https://www.openslr.org/12/)。

## 已验证的兼容性边界

- Wav2Vec2 Base 960h 的 20 秒静态图在这台 2 GB 设备上未能返回有效输出，随后 AppMgr 报告内存压力并暂时拒绝启动应用。表中仅收录已完成推理的 6 秒配置。20 秒配置的板端错误和系统保护记录随原始证据归档。
- BlazePose Detector 的原始分数头在 FP16 上曾输出非有限值。部署图将 sigmoid 之前的分数限幅到 ±16；在有限输入上，这一改写引起的概率绝对误差上界约为 1.2×10⁻⁷。该项 CPU 输出核对使用相同限幅的参考图。
- 三个 MediaPipe NHWC 输入图在编译前添加外部 NCHW 转置，再由板端传入运行时要求的 NHWC 缓冲区。对 DexiNed、LaMa 和 NAFNet，转换工具按 ONNX 的解量化定义折叠常量权重，保留源权重已有的量化误差。

VitTrack 的两路输入和响应窗按照 OpenCV TrackerVit 处理，使用通道均值/标准差和中心 Hann 窗。前一轮真实视频对照发现，仅输入 0–255 像素时平均框 IoU 为 0.054。表中跟踪数值采用修正后在同一组 18 对帧重新运行的结果。[OpenCV TrackerVit 实现](https://github.com/opencv/opencv/blob/4.x/modules/video/src/tracking/tracker_vit.cpp)

整轮测试完成并写入报告后，AppMgr 会短暂恢复内置应用，再重启场景字幕应用。因此审计采样的最后一条有时显示 `builtin`。计时区间按各模型的进程 PID、独占租约及句柄读数核对。

NAFNet 针对单张 384×384 图编译。在这个固定尺寸上，其 36 个局部均值窗口正好覆盖各层完整特征图。转换脚本检查窗口面积和边界填充方式，再将积分图求和换成全局平均池化。在 3 张输入上，两份 CPU ONNX 的最大逐像素绝对差为 6.26×10⁻⁷（输出范围约 0–1）。板端质量另用干净图像计算。

整轮记录 `verified.json` 来自应用 0.3.3。应用 0.3.4 修订了 VitTrack 的处理，跟踪结果采用 `tracker.json`。追加配置的独立轮也由 0.3.4 执行。发布资源使用修订后的输入与参考张量。

在相同的 8 张输入上，双三次插值的 Y-PSNR 为 28.43 dB，ESPCN 为 28.04 dB。运动模糊输入自身为 26.00 dB，NAFNet 输出为 25.96 dB。这两个网络在本次合成退化条件下均未提高平均 PSNR。LaMa 将遮挡区 PSNR 从填零时的 5.97 dB 提高到 14.41 dB。
