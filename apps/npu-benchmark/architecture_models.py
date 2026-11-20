"""One representative checkpoint per new image-backbone family."""

# Model ID -> family, pretrained timm checkpoint, input side, distinguishing block.
FAMILIES = {
    "starnet": ("StarNet", "starnet_s1.in1k", 224, "两个线性分支逐元素相乘的 Star 模块"),
    "hgnet": ("PP-HGNet", "hgnet_tiny.ssld_in1k", 224, "HG 块拼接多层特征并经通道聚合"),
    "efficientvit-cga": ("EfficientViT (CGA)", "efficientvit_m0.r224_in1k", 224, "级联分组注意力与夹层式布局"),
    "swiftformer": ("SwiftFormer", "swiftformer_xs.dist_in1k", 224, "高效加性注意力"),
    "selecsls": ("SelecSLS", "selecsls42b.in1k", 224, "选择性长、短跳连与特征拼接"),
    "convmixer": ("ConvMixer", "convmixer_1024_20_ks9_p14.in1k", 224, "固定分辨率图块上的空间深度卷积与逐点通道混合"),
    "gmlp": ("gMLP", "gmlp_s16_224.ra3_in1k", 224, "空间门控单元执行位置混合"),
    "xception": ("Xception", "xception41p.ra3_in1k", 299, "entry、middle、exit 三段式深度可分离卷积"),
    "focalnet": ("FocalNet", "focalnet_tiny_srf.ms_in1k", 224, "分层聚合局部与全局上下文的焦点调制"),
    "davit": ("DaViT", "davit_tiny.msft_in1k", 224, "交替执行空间窗口注意力与通道组注意力"),
    "xcit": ("XCiT", "xcit_nano_12_p16_224.fb_dist_in1k", 224, "跨协方差通道注意力与局部图块交互"),
    "maxvit": ("MaxViT", "maxvit_tiny_tf_224.in1k", 224, "同一阶段内组合块注意力、网格注意力与卷积"),
    "tnt": ("TNT", "tnt_s_patch16_224.in1k", 224, "内层像素 token 与外层图块 token 分别建模"),
    "volo": ("VOLO", "volo_d1_224.sail_in1k", 224, "Outlook attention 按位置重组局部邻域"),
    "hiera": ("Hiera", "hiera_tiny_224.mae_in1k_ft_in1k", 224, "mask unit 内注意力与查询池化分阶段降采样"),
    "gcvit": ("GC ViT", "gcvit_xxtiny.in1k", 224, "跨窗口共享的全局查询与局部窗口交互"),
}


def contracts(model):
    return [
        model(key, family, "图像分类", "imagenette-extra", (side, side), "cropnet", "extra-class")
        for key, (family, _, side, _) in FAMILIES.items()
    ]
