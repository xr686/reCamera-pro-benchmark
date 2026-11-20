"""Fixed tensor contracts for the third benchmark resource pack."""

BACKBONES = {
    "mixnet-s": ("MixNet-S", "mixnet_s.ft_in1k"),
    "rexnet": ("ReXNet-100", "rexnet_100.nav_in1k"),
    "dla34": ("DLA-34", "dla34.in1k"),
    "hrnet-small": ("HRNet-W18 Small V2", "hrnet_w18_small_v2.ms_in1k"),
    "lcnet": ("PP-LCNet ×1", "lcnet_100.ra2_in1k"),
    "dpn68": ("DPN-68", "dpn68.mx_in1k"),
    "efficientformer": ("EfficientFormer-L1", "efficientformer_l1.snap_dist_in1k"),
    "mobilevit-xs": ("MobileViT-XS", "mobilevit_xs.cvnets_in1k"),
}

STEREO = {
    "stereo-sad": "SAD block matching",
    "stereo-zsad": "Zero-mean SAD",
    "stereo-gradient": "Gradient block matching",
    "stereo-ncc": "Normalized cross-correlation",
}
PATCH = {"tfeat": "TFeat", "hardnet": "HardNet", "sosnet": "SOSNet"}
APPAREL = {
    "fashion-nin": "Network in Network",
    "fashion-highway": "Highway MLP",
    "fashion-mixer": "Patch MLP-Mixer",
}
TEXT = {
    "sms-nb": "Multinomial Naive Bayes",
    "sms-svm": "Linear SVM",
    "sms-mlp": "Hashed n-gram MLP",
}
REGRESSION = {
    "airfoil": ("翼型气动噪声回归", 5),
    "wine": ("红酒感官评分回归", 11),
    "energy": ("建筑采暖负荷回归", 8),
}
REGRESSORS = {
    "pls": "PLS regression",
    "elm": "Extreme Learning Machine",
    "rff": "Random Fourier ridge",
}
RESTORE = {
    "bayer-linear": ("Bilinear RGGB", "Bayer 去马赛克", "bayer"),
    "bayer-residual": ("Residual demosaicing CNN", "Bayer 去马赛克", "bayer"),
    "jpeg-arcnn": ("AR-CNN", "JPEG 去压缩伪影", "jpeg"),
    "jpeg-dilated": ("Dilated residual CNN", "JPEG 去压缩伪影", "jpeg"),
}

ANOMALY = {
    "padim": "PaDiM (8 features)",
    "student": "Student–teacher CNN",
    "ae": "Convolutional autoencoder",
}


def contracts(model):
    return [
        model(k, n, t, g, (16000,), "logmel", d) for k, (n, t, g, d) in AUDIO.items()
    ] + (
        [
            model(
                k,
                v[0],
                "图像分类",
                "imagenette-extra",
                ((256, 256) if k == "mobilevit-xs" else (224, 224)),
                "cropnet",
                "extra-class",
            )
            for k, v in BACKBONES.items()
        ]
        + [
            model(
                k,
                n,
                "双目视差估计",
                "stereo",
                (160, 96),
                "stereo-pair",
                "disparity-map",
            )
            for k, n in STEREO.items()
        ]
        + [
            model(
                k,
                n,
                "局部图像块检索",
                "patch",
                (32, 32),
                "std-patch",
                "patch-embedding",
            )
            for k, n in PATCH.items()
        ]
        + [
            model(k, n, "服饰品类识别", "fashion", (28, 28), "gray01", "extra-class")
            for k, n in APPAREL.items()
        ]
        + [
            model(k, n, "短信垃圾信息识别", "sms", (512,), "hashed-text", "spam")
            for k, n in TEXT.items()
        ]
        + [
            model(
                f"{ds}-{kind}",
                f"{arch} · {ds}",
                task,
                ds,
                (size,),
                "tabular",
                "regression",
            )
            for ds, (task, size) in REGRESSION.items()
            for kind, arch in REGRESSORS.items()
        ]
        + [
            model(
                k,
                name,
                task,
                corpus,
                (96, 96),
                ("bayer4" if corpus == "bayer" else "rgb01"),
                "photometric",
            )
            for k, (name, task, corpus) in RESTORE.items()
        ]
        + [
            model(
                f"mvtec-{ds}-{key}",
                f"{name} · {ds}",
                "工业外观异常检测",
                "mvtec-" + ds,
                (128, 128),
                "rgb01",
                "anomaly-score",
            )
            for ds in ("bottle", "grid")
            for key, name in ANOMALY.items()
        ]
        + [
            model(k, name, "目标检测", "voc-objects", (640, 640), "rgb01", k)
            for k, name in (("yolov5-relu", "YOLOv5s-ReLU"), ("yolov10n", "YOLOv10n"))
        ]
    )


AUDIO = {
    "speaker-tdnn": ("Compact x-vector TDNN", "说话人检索", "speaker", "embedding"),
    "speaker-rescnn": ("Residual speaker CNN", "说话人检索", "speaker", "embedding"),
    "presence-lda": (
        "Log-Mel shrinkage LDA",
        "片段语音存在性分类",
        "presence",
        "extra-class",
    ),
    "presence-cnn": (
        "Spectrogram CNN",
        "片段语音存在性分类",
        "presence",
        "extra-class",
    ),
}
