"""Decode and evaluate a single prominent box or a prompted per-instance foreground mask."""

import cv2
import numpy as np


def decode_object(kind, outputs):
    if kind == "prompt-mask":
        return {"mask": outputs[0].squeeze() > 0.0}
    box = np.clip(outputs[0].reshape(4), 0, 1)
    return {"box": np.r_[np.minimum(box[:2], box[2:]), np.maximum(box[:2], box[2:])]}


def object_quality(root, kind, rec, pred):
    if kind == "prompt-mask":
        mask = cv2.imread(str(root / rec["mask"]), cv2.IMREAD_GRAYSCALE)
        valid = mask != 255
        target = mask == 1
        estimate = pred["mask"] & valid
        return {
            "前景 IoU": [
                float((target & estimate).sum() / max(1, (target | estimate).sum())),
                1,
            ]
        }
    box = pred["box"]
    gt = np.asarray(rec["bbox"])
    intersection = np.maximum(
        0, np.minimum(box[2:], gt[2:]) - np.maximum(box[:2], gt[:2])
    ).prod()
    union = np.prod(box[2:] - box[:2]) + np.prod(gt[2:] - gt[:2]) - intersection
    iou = float(intersection / max(1e-9, union))
    return {"平均框 IoU": [iou, 1], "IoU≥0.5": [int(iou >= 0.5), 1]}
