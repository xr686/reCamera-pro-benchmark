"""Contracts, output decoding and labelled metrics for the second benchmark resource pack."""

from collections import defaultdict
import cv2
import numpy as np
from signal_catalog import ARCH_NAMES, SIGNALS

VISION = {
    "ghostnet": "GhostNet 1.0",
    "mobileone-s0": "MobileOne-S0",
    "repvgg-a0": "RepVGG-A0",
    "edgenext-xxs": "EdgeNeXt-XXS",
    "squeezenet1_1": "SqueezeNet 1.1",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mnasnet0_5": "MNASNet 0.5",
    "mobilenet_v3_small": "MobileNetV3 Small",
    "regnet_x_400mf": "RegNetX-400MF",
    "googlenet": "GoogLeNet",
    "resnext50_32x4d": "ResNeXt-50 32×4d",
    "convnext_tiny": "ConvNeXt Tiny",
    "swin_t": "Swin Transformer Tiny",
    "vit_b_32": "ViT-B/32",
    "shufflenet_v2_x0_5": "ShuffleNet V2 ×0.5",
}
SEG = {
    "lraspp_mobilenet_v3_large": "LR-ASPP MobileNetV3",
    "deeplabv3_mobilenet_v3_large": "DeepLabV3 MobileNetV3",
    "fcn_resnet50": "FCN ResNet-50",
}
SPECIAL = [
    (
        "clip-image",
        "CLIP ViT-B/32 图像塔",
        "图文零样本分类",
        "clip-images",
        (224, 224),
        "clipimage",
        "clip-rank",
    ),
    (
        "clip-text",
        "CLIP ViT-B/32 文本塔",
        "以文检图",
        "clip-texts",
        (20,),
        "cliptext",
        "clip-rank",
    ),
    (
        "mlsd",
        "M-LSD Tiny",
        "线段检测",
        "line-scenes",
        (320, 320),
        "line4",
        "lines",
        "nhwc",
    ),
    (
        "u2netp",
        "U²-NetP",
        "显著目标提取",
        "voc-extra",
        (320, 320),
        "saliency",
        "saliency",
    ),
    (
        "zero-dce",
        "Zero-DCE",
        "低照度增强",
        "exposure",
        (192, 192),
        "rgb01",
        "photometric",
    ),
    (
        "dncnn25",
        "DnCNN σ25",
        "图像去噪",
        "noisy-gray",
        (128, 128),
        "gray01",
        "photometric",
    ),
    (
        "ffdnet25",
        "FFDNet Color σ25",
        "图像去噪",
        "noisy-rgb",
        (128, 128),
        "rgb01",
        "photometric",
    ),
    (
        "mosaic",
        "Mosaic TransformerNet",
        "艺术风格迁移",
        "style",
        (224, 224),
        "rgb",
        "stylize",
    ),
    (
        "superpoint",
        "SuperPoint",
        "局部特征匹配",
        "homography",
        (256, 192),
        "gray01",
        "features",
    ),
    ("xfeat", "XFeat", "局部特征匹配", "homography", (256, 192), "gray01", "features"),
    ("ds-kws", "DS-CNN", "语音指令识别", "commands", (16000,), "logmel", "extra-class"),
    (
        "tc-kws",
        "TC-ResNet",
        "语音指令识别",
        "commands",
        (16000,),
        "logmel",
        "extra-class",
    ),
    (
        "lenet-digits",
        "LeNet-5",
        "手写数字识别",
        "digits",
        (32, 32),
        "gray01",
        "extra-class",
    ),
]


def contracts(model):
    return (
        [
            model(
                k,
                n,
                "图像分类",
                "imagenette-extra",
                (224, 224),
                "cropnet",
                "extra-class",
            )
            for k, n in VISION.items()
        ]
        + [
            model(k, n, "场景语义分割", "voc-extra", (256, 256), "imagenet", "voc21")
            for k, n in SEG.items()
        ]
        + [model(*row) for row in SPECIAL]
        + [
            model(
                "sig-" + ds.lower(),
                f"{ARCH_NAMES[arch]} · {ds}",
                task,
                "sig-" + ds.lower(),
                (ch, n),
                "signal",
                "extra-class",
            )
            for ds, arch, task, ch, n in SIGNALS
        ]
    )


def log_mel(samples):
    pcm = np.zeros(16000, np.float32)
    flat = np.asarray(samples).ravel()
    pcm[: min(len(flat), 16000)] = flat[:16000]
    frames = np.lib.stride_tricks.sliding_window_view(pcm, 400)[::160]
    power = np.abs(np.fft.rfft(frames * np.hanning(400), n=512)) ** 2
    edges = 700 * np.expm1(np.linspace(np.log1p(20 / 700), np.log1p(7600 / 700), 42))
    hz = np.arange(257) * 16000 / 512
    filters = np.maximum(
        0,
        np.minimum(
            (hz[:, None] - edges[:-2]) / (edges[1:-1] - edges[:-2]),
            (edges[2:] - hz[:, None]) / (edges[2:] - edges[1:-1]),
        ),
    )
    return np.log(np.maximum(power @ filters, 1e-7)).T.astype(np.float32)


def prepare_extra(model, data, metadata):
    if model.pixels == "line4":
        img = cv2.resize(data, model.size, interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        return [
            np.concatenate([img, np.ones((*img.shape[:2], 1), np.float32)], axis=2)[
                None
            ]
        ]
    if model.pixels == "signal":
        norm = metadata["normalization"]
        x = (data - np.asarray(norm["mean"], np.float32)[:, None]) / np.asarray(
            norm["std"], np.float32
        )[:, None]
        return [np.ascontiguousarray(x[None, :, None, :])]
    if model.pixels == "logmel":
        return [log_mel(data)[None, None]]
    if model.pixels == "gray01":
        x = cv2.cvtColor(cv2.resize(data, model.size), cv2.COLOR_BGR2GRAY)
        return [x.astype(np.float32)[None, None] / 255]
    if model.pixels == "saliency":
        x = cv2.cvtColor(cv2.resize(data, model.size), cv2.COLOR_BGR2RGB).astype(
            np.float32
        )
        x = x / max(float(x.max()), 1.0)
    else:
        spec = metadata["transform"]
        side = spec["resize"][0]
        h, w = data.shape[:2]
        inter = (
            cv2.INTER_CUBIC if spec["interpolation"] == "bicubic" else cv2.INTER_LINEAR
        )
        im = cv2.resize(
            data,
            (int(w * side / min(h, w)), int(h * side / min(h, w))),
            interpolation=inter,
        )
        crop = model.size[0]
        y, x0 = (int(round((v - crop) / 2)) for v in im.shape[:2])
        x = (
            cv2.cvtColor(im[y : y + crop, x0 : x0 + crop], cv2.COLOR_BGR2RGB).astype(
                np.float32
            )
            / 255
        )
    spec = metadata.get("transform", {}) if metadata else {}
    x = (x - np.array(spec.get("mean", [0.485, 0.456, 0.406]), np.float32)) / np.array(
        spec.get("std", [0.229, 0.224, 0.225]), np.float32
    )
    return [np.ascontiguousarray(x.transpose(2, 0, 1)[None])]


def features(model, outputs):
    if model.file == "superpoint":
        logits, desc = (x[0] for x in outputs)
    else:
        desc, logits = outputs[0][0], outputs[1][0]
    exp = np.exp(logits - logits.max(0, keepdims=True))
    prob = (exp / exp.sum(0))[:-1]
    _, h, w = prob.shape
    heat = prob.reshape(8, 8, h, w).transpose(2, 0, 3, 1).reshape(h * 8, w * 8)
    if model.file == "xfeat":
        heat *= cv2.resize(outputs[2][0, 0], (8 * w, 8 * h))
    keep = (heat >= cv2.dilate(heat, np.ones((7, 7), np.uint8))) & (heat > 0.001)
    keep[:5] = keep[-5:] = False
    keep[:, :5] = keep[:, -5:] = False
    y, x = np.where(keep)
    ids = np.argsort(heat[y, x])[::-1][:256]
    x, y = x[ids], y[ids]
    xy = np.column_stack((x, y)).astype(np.float32)
    if len(xy):
        # Bilinear samples in the dense descriptor map, normalized for cosine matching.
        uv = np.maximum(0, np.minimum((xy + 0.5) / 8 - 0.5, np.array([w - 1, h - 1])))
        ix, iy = uv.astype(int).T
        fx, fy = (uv - np.floor(uv)).T
        jx, jy = np.minimum(ix + 1, w - 1), np.minimum(iy + 1, h - 1)
        vec = (
            desc[:, iy, ix] * (1 - fx) * (1 - fy)
            + desc[:, iy, jx] * fx * (1 - fy)
            + desc[:, jy, ix] * (1 - fx) * fy
            + desc[:, jy, jx] * fx * fy
        ).T
        vec = (
            vec / np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        ).astype(np.float32)
    else:
        vec = np.empty((0, desc.shape[0]), np.float32)
    return dict(xy=xy, descriptors=vec)


def decode_extra(model, out):
    k = model.decoder
    if k == "lines":
        score = out[0].reshape(-1)
        ids = np.argsort(score)[-200:][::-1]
        ids = ids[score[ids] > 0.10]
        y, x = np.divmod(ids, 160)
        d = out[1].reshape(160, 160, 4)[y, x].reshape(-1, 2, 2)
        segments = (d + np.column_stack([x, y])[:, None]) * 2
        return {"segments": segments[np.linalg.norm(d[:, 0] - d[:, 1], axis=1) > 20]}
    if k == "features":
        return features(model, out)
    if k == "extra-class":
        return {"top": np.argsort(out[0].ravel())[-5:][::-1].tolist()}
    if k == "voc21":
        return {"mask": out[0][0].argmax(0).astype(np.uint8)}
    if k == "saliency":
        mask = out[0].squeeze()
        return {"mask": (mask - mask.min()) / max(float(np.ptp(mask)), 1e-8) > 0.5}
    image = out[0][0].transpose(1, 2, 0)
    scale = 1 if k == "stylize" else 255
    image = np.rint(np.clip(image * scale, 0, 255)).astype(np.uint8)
    return {
        "image": cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[-1] == 3
        else image[:, :, 0]
    }


def quality_extra(root, model, rec, p):
    k = model.decoder
    if k == "clip-rank":
        return {
            "Top-1": [int(rec["label"] == p["top"][0]), 1],
            "Top-5": [int(rec["label"] in p["top"]), 1],
            "clip_projection": p["projection"],
            "prediction": p["top"][0],
        }
    if k == "lines":
        gt = np.asarray(rec["segments"])
        used = set()
        hits = 0
        for seg in p["segments"]:
            distance = np.minimum(
                np.linalg.norm(seg - gt, axis=2).mean(1),
                np.linalg.norm(seg[::-1] - gt, axis=2).mean(1),
            )
            for i in np.argsort(distance):
                if i not in used and distance[i] <= 10:
                    used.add(i)
                    hits += 1
                    break
        return {"端点 F1@10px": [2 * hits, len(gt) + len(p["segments"])]}
    if k == "extra-class":
        res = {
            "Top-1": [int(p["top"][0] == rec["label"]), 1],
            "prediction": p["top"][0],
        }
        if model.pixels == "cropnet":
            res["Top-5"] = [int(rec["label"] in p["top"]), 1]
        return res
    if k == "features":
        return {"pair_info": (rec["pair_id"], rec["side"], rec["homography"], p)}
    if k in ("voc21", "saliency"):
        gt = cv2.resize(
            cv2.imread(str(root / rec["mask"]), 0),
            model.size,
            interpolation=cv2.INTER_NEAREST,
        )
        pred = p["mask"]
        valid = gt != 255
        if k == "saliency":
            fg = (gt != 0) & valid
            return {
                "VOC 前景 IoU": [
                    int((fg & pred).sum()),
                    int(((fg | pred) & valid).sum()),
                ]
            }
        cm = np.bincount(
            21 * gt[valid].astype(int) + pred[valid], minlength=441
        ).reshape(21, 21)
        return {"voc_confusion": cm}
    im = p["image"]
    gt = cv2.imread(str(root / rec["target"]), 0 if im.ndim == 2 else 1)
    err = (im.astype(float) - gt) / 255
    if k == "stylize":
        return {"相对 CPU 图像 MAE (0–255)": [float(np.abs(err).mean() * 255), 1]}
    return {"PSNR dB": [float(-10 * np.log10(max(np.mean(err**2), 1e-12))), 1]}


def summarize_extra(rows):
    if "voc_confusion" in rows[0]:
        cm = np.sum([r.pop("voc_confusion") for r in rows], axis=0)
        union = cm.sum(1) + cm.sum(0) - cm.diagonal()
        return {
            "mIoU（有并集类别）": float(
                (cm.diagonal()[union > 0] / union[union > 0]).mean()
            )
        }
    pairs = defaultdict(dict)
    for row in rows:
        key, side, H, feats = row.pop("pair_info")
        pairs[key][side] = feats
        pairs[key]["H"] = np.asarray(H)
    hits = total = 0
    for pair in pairs.values():
        a, b = pair[0], pair[1]
        if len(a["xy"]) and len(b["xy"]):
            sim = a["descriptors"] @ b["descriptors"].T
            best = sim.argmax(1)
            i = np.flatnonzero(sim.argmax(0)[best] == np.arange(len(best)))
            warp = cv2.perspectiveTransform(a["xy"][None], pair["H"])[0]
            hits += int((np.linalg.norm(warp[i] - b["xy"][best[i]], axis=1) <= 3).sum())
            total += len(i)
    return {
        "匹配内点率@3px": hits / max(1, total),
        "每对图平均匹配数": total / len(pairs),
    }
