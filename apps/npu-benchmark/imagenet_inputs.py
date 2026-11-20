"""Image preprocessing and top-five decoding shared by the base classifier demos."""

import cv2
import numpy as np

MODELS = {"resnet18": "ResNet-18", "shufflenet-v2": "ShuffleNet V2 x1.0", "mobilenet-v2": "MobileNet V2"}


def preprocess(bgr):
    h, w = bgr.shape[:2]
    resized = cv2.resize(bgr, (256 * w // min(h, w), 256 * h // min(h, w)),
                         interpolation=cv2.INTER_LINEAR)
    y, x = (round((d - 224) / 2) for d in resized.shape[:2])
    return cv2.cvtColor(resized[y:y + 224, x:x + 224], cv2.COLOR_BGR2RGB)[None, ...]


def top_five(outputs):
    if outputs is None or len(outputs) != 1:
        raise ValueError("Expected one RKNN output tensor")
    logits = outputs[0].reshape(-1)
    if logits.size != 1000 or not np.isfinite(logits).all():
        raise ValueError("Expected 1000 finite ImageNet logits")
    return np.argsort(-logits, kind="stable")[:5].tolist()
