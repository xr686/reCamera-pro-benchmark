"""RV1126B contracts for the sequence and camera-quality benchmark pack."""

BACKBONES = {
    "fasternet-t0": ("FasterNet-T0", "fasternet_t0.in1k", 224),
    "pvtv2-b0": ("PVTv2-B0", "pvt_v2_b0.in1k", 224),
    "mobilenetv4-small": (
        "MobileNetV4 Conv Small",
        "mobilenetv4_conv_small.e1200_r224_in1k",
        224,
    ),
    "gernet-s": ("GERNet-S", "gernet_s.idstcv_in1k", 224),
    "fbnetc": ("FBNet-C", "fbnetc_100.rmsp_in1k", 224),
    "hardcorenas-a": ("HardCoRe-NAS-A", "hardcorenas_a.miil_green_in1k", 224),
    "regnety-004": ("RegNetY-400MF", "regnety_004.tv2_in1k", 224),
    "convnext-atto": ("ConvNeXt Atto", "convnext_atto.d2_in1k", 224),
}
# UCR archive names and univariate input lengths.
SERIES = {
    "GunPoint": ("手部动作轨迹分类", 150),
    "Beef": ("牛肉光谱类别识别", 470),
    "OliveOil": ("橄榄油产地光谱分类", 570),
    "Trace": ("模拟过程瞬态分类", 275),
    "SwedishLeaf": ("叶片轮廓树种识别", 128),
    "Fish": ("鱼类轮廓识别", 463),
    "ArrowHead": ("箭头轮廓分类", 251),
    "Plane": ("飞机轮廓机型识别", 144),
    "Car": ("Car 传感器序列分类", 577),
    "Ham": ("火腿蛋白图谱分类", 431),
    "ScreenType": ("家电屏幕类型识别", 720),
    "RefrigerationDevices": ("制冷设备负荷分类", 720),
}
SEQUENCE_HEADS = {
    "shapelets": "Subsequence distance transform",
    "cosine": "Cosine basis ridge",
    "randomconv": "Random convolution max/mean ridge",
}
CAMERA = {
    "jpeg-level": "JPEG 压缩等级识别",
    "noise-level": "高斯噪声强度识别",
    "blur-angle": "运动模糊方向识别",
    "focus-level": "失焦模糊等级识别",
}
CAMERA_HEADS = {
    "dct": "Block DCT energy probe",
    "derivative": "Directional derivative energy probe",
    "stackcnn": "Three-stage texture CNN",
}


OBJECTS = {
    "voc-prompt": ("提示框内单实例分割", "prompt-mask"),
    "voc-dominant": ("主目标框回归", "dominant-box"),
}
OBJECT_HEADS = {
    "plain": "Convolutional pyramid",
    "depthwise": "Depthwise pyramid",
    "fire": "Pretrained SqueezeNet pyramid",
}


PET_BREEDS = {
    "resnet": "ResNet-18 ridge probe",
    "shuffle": "ShuffleNetV2 ridge probe",
    "mobile": "MobileNetV3 Small ridge probe",
}


def contracts(model):
    models = [
        model(
            k, n, "图像分类", "imagenette-extra", (side, side), "cropnet", "extra-class"
        )
        for k, (n, _, side) in BACKBONES.items()
    ]
    for ds, (task, length) in SERIES.items():
        for key, name in SEQUENCE_HEADS.items():
            models.append(
                model(
                    ds.lower() + "-" + key,
                    name + " · " + ds,
                    task,
                    ds,
                    (1, length),
                    "signal",
                    "extra-class",
                )
            )
    for ds, task in CAMERA.items():
        for key, name in CAMERA_HEADS.items():
            models.append(
                model(
                    ds + "-" + key,
                    name + " · " + ds,
                    task,
                    ds,
                    (64, 64),
                    "gray01",
                    "extra-class",
                )
            )
    for ds, (task, decoder) in OBJECTS.items():
        for key, name in OBJECT_HEADS.items():
            models.append(
                model(
                    ds + "-" + key,
                    name + " · " + ds,
                    task,
                    ds,
                    (96, 96),
                    "rgb01",
                    decoder,
                )
            )
    for key, name in PET_BREEDS.items():
        models.append(
            model(
                "pet-breed-" + key,
                name + " · Pet",
                "宠物品种识别",
                "pet-breed",
                (160, 160),
                "rgb01",
                "extra-class",
            )
        )
    for key, name in OBJECT_HEADS.items():
        models.append(
            model(
                "pet-mask-" + key,
                name + " · Pet",
                "无提示宠物前景分割",
                "pet-mask",
                (96, 96),
                "rgb01",
                "prompt-mask",
            )
        )
    return models
