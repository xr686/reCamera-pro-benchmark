"""Input and output contracts for the fourth benchmark resource pack."""

BACKBONES = {
    "poolformer": "poolformer_s12.sail_in1k",
    "repghost": "repghostnet_100.in1k",
    "vovnet": "ese_vovnet19b_dw.ra_in1k",
    "fastvit": "fastvit_t8.apple_in1k",
    "levit": "levit_128s.fb_dist_in1k",
    "skresnet": "skresnet18.ra_in1k",
    "resnest": "resnest14d.gluon_in1k",
    "coat-lite": "coat_lite_tiny.in1k",
}

DOMAINS = {
    "eurosat": "遥感土地利用分类",
    "flowers": "花卉品类识别",
    "gts": "交通标志识别",
    "voc-tags": "图像多标签识别",
}
HEADS = {
    "resprobe": "ResNet-18 ridge probe",
    "squeezeprobe": "SqueezeNet ridge probe",
    "gabor": "Gabor filter bank + ridge",
    "branches": "Multiscale CNN",
}
TABLES = {
    "wifi": ("Wi-Fi 房间定位", 7),
    "gridstab": ("电网稳定性分类", 12),
    "seeds": ("小麦籽粒品种识别", 7),
    "raisins": ("葡萄干品种识别", 7),
}
ESTIMATORS = {
    "qda": "Diagonal QDA",
    "lvq": "Nearest subclass prototype",
    "randomnet": "Random ReLU features + ridge",
}
LANG = {"sent": "英文短评情感分类", "trec": "英文问题类型识别"}
LANG_MODELS = {
    "bnb": "Bernoulli Naive Bayes",
    "ridge": "N-gram ridge classifier",
    "perceptron": "Averaged linear perceptron",
}
FLOW = {
    "flow-hs": "Horn–Schunck (12 iterations)",
    "flow-lk": "Dense Lucas–Kanade (7×7)",
    "flow-sad": "2-D block matching (49 offsets)",
}
COLOR = {
    "wb-gray": "Gray World",
    "wb-shades": "Shades of Gray (p=4)",
    "wb-edge": "First-order Gray Edge",
}
COPIES = {
    "hash-dct": "DCT perceptual hash",
    "hash-difference": "Difference hash",
    "hash-spectrum": "Low-frequency DCT descriptor",
}
TONES = {
    "dtmf-filter": "Quadrature tone bank",
    "dtmf-mlp": "Spectral-bin MLP",
    "pitch-yin": "YIN difference function",
    "pitch-acf": "Normalized autocorrelation",
}


def contracts(model):
    catalog = [
        model(
            k,
            BACKBONES[k].split(".")[0],
            "图像分类",
            "imagenette-extra",
            ((256, 256) if k == "fastvit" else (224, 224)),
            "cropnet",
            "extra-class",
        )
        for k in BACKBONES
    ]
    for ds, task in DOMAINS.items():
        for key, name in HEADS.items():
            catalog.append(
                model(
                    ds + "-" + key,
                    name + " · " + ds,
                    task,
                    ds,
                    (80, 80),
                    "rgb01",
                    "multi-label" if ds == "voc-tags" else "extra-class",
                )
            )
    for ds, (task, size) in TABLES.items():
        for key, name in ESTIMATORS.items():
            catalog.append(
                model(
                    ds + "-" + key,
                    name + " · " + ds,
                    task,
                    ds,
                    (size,),
                    "tabular",
                    "extra-class",
                )
            )
    for ds, task in LANG.items():
        for key, name in LANG_MODELS.items():
            catalog.append(
                model(
                    ds + "-" + key,
                    name + " · " + ds,
                    task,
                    ds,
                    (512,),
                    "hashed-text",
                    "extra-class",
                )
            )
    catalog.extend(
        model(
            k,
            n,
            "稠密光流估计",
            "flow",
            (160, 112),
            "stereo-pair",
            "flow-cost" if k == "flow-sad" else "flow-vector",
        )
        for k, n in FLOW.items()
    )
    catalog.extend(
        model(k, n, "光源颜色估计", "wb", (96, 96), "rgb01", "illumination")
        for k, n in COLOR.items()
    )
    catalog.extend(
        model(k, n, "近重复图像检索", "copies", (32, 32), "gray01", "embedding" if k=="hash-spectrum" else "binary-hash")
        for k, n in COPIES.items()
    )
    for k, n in TONES.items():
        dtmf = k.startswith("dtmf")
        catalog.append(
            model(
                k,
                n,
                "双音多频按键识别" if dtmf else "单音基频估计",
                "dtmf" if dtmf else "pitch",
                (1024,),
                "short-wave",
                "extra-class" if dtmf else "pitch",
            )
        )
    return catalog
