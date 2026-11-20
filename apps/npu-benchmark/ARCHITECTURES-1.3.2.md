# RV1126B 模型架构基准

reCamera Pro，固件 V1.3.2。本轮 16 个新增视觉分类模型，每项对应一个预训练权重。历史 ImageNet 分类记录共 43 个配置，按下述口径归为 34 个代表模型，合计 50 个代表模型。历史多任务总表的 254 是权重和任务配置数。

## 相同测试集上的板端结果

100 张 Imagenette 验证图，每类 10 张，按原始 ImageNet 1000 类排序。全部采用 batch 1、RV1126B FP16 RKNN，每模型预热 5 次，再依次测量相同的 100 张图片。按同步推理均值升序排列。

| 代表模型 | 输入 | RKNN MiB | 前处理 ms | 推理 ms | 推理 P95 ms | 后处理 ms | 三段 FPS | 含回收 FPS | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EfficientViT (CGA) | 224×224 | 6.3 | 13.35 | 16.24 | 18.04 | 0.27 | 33.49 | 8.27 | 59% | 86% |
| StarNet | 224×224 | 5.8 | 13.47 | 17.43 | 18.48 | 0.27 | 32.08 | 8.26 | 69% | 90% |
| SwiftFormer | 224×224 | 7.7 | 12.88 | 22.80 | 24.41 | 0.26 | 27.82 | 7.93 | 79% | 93% |
| SelecSLS | 224×224 | 62.1 | 13.44 | 29.50 | 32.46 | 0.27 | 23.14 | 7.49 | 83% | 95% |
| PP-HGNet | 224×224 | 28.8 | 12.89 | 33.12 | 35.44 | 0.27 | 21.61 | 7.20 | 83% | 99% |
| XCiT | 224×224 | 10.0 | 12.78 | 35.89 | 39.49 | 0.28 | 20.43 | 7.05 | 71% | 93% |
| FocalNet | 224×224 | 57.5 | 13.36 | 71.38 | 74.69 | 0.45 | 11.74 | 5.46 | 84% | 96% |
| Xception | 299×299 | 52.4 | 22.07 | 72.85 | 76.19 | 0.47 | 10.48 | 5.19 | 83% | 96% |
| ConvMixer | 224×224 | 48.3 | 13.06 | 74.53 | 78.02 | 0.47 | 11.36 | 5.35 | 81% | 96% |
| DaViT | 224×224 | 60.3 | 13.20 | 94.91 | 98.36 | 0.48 | 9.21 | 4.86 | 81% | 99% |
| gMLP | 224×224 | 40.8 | 13.66 | 112.65 | 121.71 | 0.47 | 7.89 | 4.41 | 80% | 98% |
| GC ViT | 224×224 | 38.4 | 13.49 | 145.07 | 157.97 | 0.47 | 6.29 | 3.86 | 78% | 96% |
| Hiera | 224×224 | 77.2 | 13.33 | 211.55 | 240.49 | 0.47 | 4.44 | 3.07 | 86% | 99% |
| TNT | 224×224 | 183.5 | 13.43 | 340.78 | 352.78 | 0.50 | 2.82 | 2.21 | 83% | 95% |
| MaxViT | 224×224 | 87.9 | 13.31 | 533.83 | 581.74 | 0.48 | 1.83 | 1.55 | 85% | 97% |
| VOLO | 224×224 | 90.5 | 12.94 | 593.36 | 628.61 | 0.49 | 1.65 | 1.42 | 82% | 95% |

## 架构与主机对照

| 代表模型 | 权重 / 实现 | 主要结构 | ONNX Top-1 / Top-5 | 板端与 ONNX 抽样余弦 |
|---|---|---|---:|---:|
| EfficientViT (CGA) | [efficientvit_m0.r224_in1k](https://huggingface.co/timm/efficientvit_m0.r224_in1k) | 级联分组注意力与夹层式布局 | 59% / 86% | 0.99995 |
| StarNet | [starnet_s1.in1k](https://huggingface.co/timm/starnet_s1.in1k) | 两个线性分支逐元素相乘的 Star 模块 | 69% / 90% | 0.99997 |
| SwiftFormer | [swiftformer_xs.dist_in1k](https://huggingface.co/timm/swiftformer_xs.dist_in1k) | 高效加性注意力 | 79% / 93% | 0.99997 |
| SelecSLS | [selecsls42b.in1k](https://huggingface.co/timm/selecsls42b.in1k) | 选择性长、短跳连与特征拼接 | 84% / 95% | 0.99997 |
| PP-HGNet | [hgnet_tiny.ssld_in1k](https://huggingface.co/timm/hgnet_tiny.ssld_in1k) | HG 块拼接多层特征并经通道聚合 | 82% / 99% | 0.99997 |
| XCiT | [xcit_nano_12_p16_224.fb_dist_in1k](https://huggingface.co/timm/xcit_nano_12_p16_224.fb_dist_in1k) | 跨协方差通道注意力与局部图块交互 | 71% / 93% | 0.99989 |
| FocalNet | [focalnet_tiny_srf.ms_in1k](https://huggingface.co/timm/focalnet_tiny_srf.ms_in1k) | 分层聚合局部与全局上下文的焦点调制 | 84% / 97% | 0.99962 |
| Xception | [xception41p.ra3_in1k](https://huggingface.co/timm/xception41p.ra3_in1k) | entry、middle、exit 三段式深度可分离卷积 | 83% / 96% | 0.99996 |
| ConvMixer | [convmixer_1024_20_ks9_p14.in1k](https://huggingface.co/timm/convmixer_1024_20_ks9_p14.in1k) | 固定分辨率图块上的空间深度卷积与逐点通道混合 | 81% / 96% | 0.99934 |
| DaViT | [davit_tiny.msft_in1k](https://huggingface.co/timm/davit_tiny.msft_in1k) | 交替执行空间窗口注意力与通道组注意力 | 81% / 99% | 0.99973 |
| gMLP | [gmlp_s16_224.ra3_in1k](https://huggingface.co/timm/gmlp_s16_224.ra3_in1k) | 空间门控单元执行位置混合 | 80% / 98% | 0.99999 |
| GC ViT | [gcvit_xxtiny.in1k](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-morevit/gcvit_xxtiny_224_nvidia-d1d86009.pth) | 跨窗口共享的全局查询与局部窗口交互 | 78% / 96% | 0.99994 |
| Hiera | [hiera_tiny_224.mae_in1k_ft_in1k](https://huggingface.co/timm/hiera_tiny_224.mae_in1k_ft_in1k) | mask unit 内注意力与查询池化分阶段降采样 | 87% / 99% | 0.99914 |
| TNT | [tnt_s_patch16_224.in1k](https://huggingface.co/timm/tnt_s_patch16_224.in1k) | 内层像素 token 与外层图块 token 分别建模 | 82% / 95% | 0.99964 |
| MaxViT | [maxvit_tiny_tf_224.in1k](https://huggingface.co/timm/maxvit_tiny_tf_224.in1k) | 同一阶段内组合块注意力、网格注意力与卷积 | 85% / 97% | 0.99998 |
| VOLO | [volo_d1_224.sail_in1k](https://huggingface.co/timm/volo_d1_224.sail_in1k) | Outlook attention 按位置重组局部邻域 | 87% / 99% | 0.92434 |

## 去重与测量口径

架构以主干的网络拓扑和核心计算模块分组。同一模型变更深度、宽度、分辨率、下游任务、分类头或权重只占一行。历史表中保守地将 ResNeXt、SK-ResNet、ResNeSt 归入 ResNet，将不同代的 MobileNet 归入 MobileNet。应用默认仅运行已安装的新增模型代表，历史数据可按模型 ID 查询。历史索引按已测 ImageNet 分类骨干建立，保留原始样本量和轮次。

前处理包括按权重设置缩放短边、中心裁剪、BGR 转 RGB、均值方差归一化及输入内存布局转换。短边和归一化参数逐模型保存在 CSV。推理包含输入提交至同步取回输出，后处理对 1000 个类别分数排序取前五。三段 FPS 按三段均值之和取倒数；含回收 FPS 另计 Python 应用每帧释放输出及垃圾回收。采集、读盘、图像解码和标注计分均在计时外。准确率仅适用于本次 100 图子集。

有效结果来自 8 份真机分组记录。每组通过 AppMgr 取得 NPU 独占租约，连续三次观察到零负载后预热。控制器轮询运行应用，共记录 192 次模型内设备持有者检查，只观察到基准进程及固件 broker rkipc。记录到的 NPU 频率为 800–800 MHz，温度为 76.5–86.9 °C。原始 ONNX 在相同的全部输入上评分，余弦为三个输入的抽样张量对照。

## 转换记录

- SwiftFormer 的二维空间均值在 RV1126B 编译图中使用全局平均池化再展平。对 100 张输入，原始与改写 ONNX 的最大 logit 绝对差为 1.91×10⁻⁶，Top-1 预测全部一致。
- VOLO 的原始 ONNX 使用 opset 18 的 Col2Im 表达邻域重叠相加。编译图按固定步长取邻域切片，再按输出位置的奇偶性累加，最后做像素重排。100 张输入上的最大 logit 绝对差为 1.10×10⁻⁵，Top-1 全部一致。
- VOLO 的最终编译日志有四条 Transpose 转至 CPU 的提示。其 593.36 ms 是板端混合执行的同步耗时。100 张样本的 Top-1 从主机原始 ONNX 的 87% 降至板端 82%，Top-5 从 99% 降至 95%，抽样 logit 余弦为 0.92434。其余 15 个架构的抽样余弦均高于 0.999。现有对照无法将精度差异归因于某个算子。
- MaxViT 的 RKNN-Toolkit2 2.3.2 构建按编译器诊断关闭单条图优化规则 `convert_rs_add_rs_to_rs_gather_elements`，主机和转换器读取同一份原始 ONNX。逐模型权重版本、原始 ONNX、改写图及板端 RKNN 摘要均可在原始记录包中核验。

候选架构核对了项目的历史实测总表及外部参考仓库的模型目录。本轮使用 timm 公开的分类权重，选择同一任务以固定样本和评分方式。[参考仓库](https://github.com/xr686/reCamera-pro-benchmark) · [timm](https://github.com/huggingface/pytorch-image-models) · [Imagenette](https://github.com/fastai/imagenette)
