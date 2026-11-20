## 算法、样本和质量指标

| 任务 | 输入、训练与评估 |
|---|---|
| ImageNet 分类 | PoolFormer-S12、RepGhostNet、ESE-VoVNet、FastViT-T8、LeViT-128S、SK-ResNet-18、ResNeSt-14d、CoaT-Lite Tiny 使用各自的 timm 预训练权重。用前三轮相同的 100 张 Imagenette 验证图，对完整 1000 类排序。FastViT 输入 256×256，其余七个为 224×224。[timm](https://github.com/huggingface/pytorch-image-models) · [Imagenette](https://github.com/fastai/imagenette) |
| 遥感土地利用、花卉、交通标志 | 每个任务都比较四种算法。① 冻结 ImageNet ResNet-18 并拟合岭分类器；② 冻结 SqueezeNet 1.1 并拟合相同类型的分类器；③ 12 个 Gabor 滤波器与 RGB 统计特征接岭分类器；④ 从头训练 12 轮的多尺度 CNN。特征提取和分类头均进入 RKNN 图。为便于横向比较，四者输入均为 80×80 RGB。EuroSAT 的 10 类使用 1500 张训练、120 张测试；TF Flowers 的 5 类为 1000/100。两者以源文件哈希排序并取互斥图片，EuroSAT 未按地理区域隔离。GTSRB 使用官方分开的训练和测试集，每类 48/4 张，43 类共 2064/172 张，评估裁剪后的标志品类。[EuroSAT](https://github.com/phelber/EuroSAT) · [TF Flowers](https://www.tensorflow.org/datasets/catalog/tf_flowers) · [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html) |
| 图像多标签识别 | 在 VOC2012 Main/train、Main/val 中按文件名哈希顺序取图，直到每一类达到预设的正样本数。共 962 张训练、117 张验证。将每幅图的全部标注框映射为 20 维存在性标签，包含 difficult 对象。四种网络对整图输出 20 个分数，按类别分数排序、相同分数并列，对各类 AP 取宏平均。[PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) |
| 稠密光流 | 来自 Middlebury 的全部 8 个公开真值序列各取一对 frame10/frame11，缩放为 160×112 灰度，同时按横纵比例缩放真值位移。保留有限的有效真值，去除 3 px 图像边缘。比较 12 次迭代的 Horn–Schunck、带正则项的 7×7 Lucas–Kanade 和 ±3 px 搜索半径的 7×7 SAD 块匹配。全部输入共用一套评分，保留超过块匹配搜索范围的真值。EPE 是按有效像素累计的端点欧氏误差。三个都是单尺度实现，吞吐单位为图像对/s。[Middlebury Optical Flow](https://vision.middlebury.edu/flow/data/) |
| 光源颜色估计 | Gray World、四阶 Shades of Gray、一阶 Gray Edge 在 48 幅带已知 RGB 光源增益的合成漫反射场景上运行。每个场景包含随机表面反射率、平滑的空间阴影与量化截断。指标为预测光源与真值之间的平均夹角，单位为度。实拍 RAW、不同传感器光谱响应、混合光源的误差尚需独立测量。 |
| 近重复图像检索 | 从同一批 Imagenette 样本取 60 张图库图，每张生成曝光加 JPEG 压缩及裁剪缩放两种查询，共 180 次描述子提取、120 个查询。比较去直流的低频 DCT 均值阈值哈希、相邻亮度差哈希和连续低频 DCT 向量。Rank-1 判断最相似的图库项能否还原到源图。表中吞吐单位为单图描述子/s，图库比对在计时外。 |
| 双音多频识别 | 8 kHz、128 ms 单声道，标准 4×4 DTMF 频率组合，共 16 个按键。合成输入包含随机相位、振幅、频率偏差和高斯噪声。比较正交正弦滤波器组和以相同滤波特征训练的 MLP，后者使用 1600 段训练。测试用独立随机种子、每键 6 段，共 96 段。 |
| 单音基频估计 | 96 段合成谐波信号，90–410 Hz，含高斯噪声，每三段有一段去除基频分量。比较 YIN 累积均值归一化差分函数与归一化自相关，在 18–100 个采样点的延迟候选中选峰并抛物线插值。结果给出平均音分绝对误差与 50 音分以内的比例。音分对数标度上，一次倍频或半频错误约为 1200 音分。 |
| 表格分类 | 每项都训练对角协方差 QDA、类内 4 个 K-means 原型的最近中心分类器、96 个随机 ReLU 特征接岭分类器。先按相同输入去重，再随机分层划分，归一化只使用训练数据。Wi-Fi 七个接收强度用于区分 4 间房，训练/测试为 1440/96；电网稳定性为仿真数据，7199/48，模型仅使用 12 个物理输入，排除连续稳定性输出；小麦 151/59；葡萄干 647/48。报告各自测试子集上的 Top-1。[Wi-Fi](https://archive.ics.uci.edu/dataset/422/wireless+indoor+localization) · [电网稳定性](https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data) · [Seeds](https://archive.ics.uci.edu/dataset/236/seeds) · [Raisin](https://archive.ics.uci.edu/dataset/850/raisin) |
| 短评情感、问题分类 | 英文短评来自 UCI 的 Amazon、IMDb、Yelp 三个来源，统一文本后去重并分层划分，2234 条训练、60 条测试。TREC 使用官方训练和测试文件，排除与训练文本相同的查询，六种粗类共 5452 条训练、157 条测试。三种模型使用相同的 512 维单词和双词组计数特征；Bernoulli NB 在模型内转为出现标志，岭分类器和平均感知机使用计数。前处理含从文本生成特征的时间。[UCI Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences) · [TREC 问题分类](https://cogcomp.seas.upenn.edu/Data/QA/QC/) |

训练与测试源文件清单、每个配置的观测样本数、源 ONNX 和板端 RKNN 的摘要均保存在配套证据中。TF Flowers 缩放图的原始作者和许可可在证据包 source-notices/flowers-LICENSE.txt 中按训练或测试清单中的文件名查找。

### 二值图像哈希的转换复测

首版将二值阈值放在 RKNN 图中。两种哈希在板端的 Rank-1 都只有 2/120，主机结果为 116/120。0.6.1 保留 NPU 上的特征计算，改为输出带符号的连续值，在应用后处理里逐位比较阈值。用 180 张输入对比新旧主机实现，DCT 哈希的 11,340 位和差分哈希的 11,520 位全部一致。重新转换并部署后，二者的板端 Rank-1 都达到 116/120（96.67%）。累计表引用修正版计时，阈值计算计入后处理。证据包保留首版板端记录、模型摘要及逐位对照汇总。
