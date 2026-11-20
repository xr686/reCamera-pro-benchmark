"""Prepare inputs and evaluate stereo, descriptor, Bayer, text, tabular and YOLO heads."""

import re
import zlib
import cv2
import numpy as np


def text_vector(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    terms = tokens + [a + " " + b for a, b in zip(tokens, tokens[1:])]
    vector = np.zeros(512, np.float32)
    for term in terms:
        vector[zlib.crc32(term.encode()) % len(vector)] += 1
    return vector


def prepare_extended(model, data, metadata):
    if model.pixels == "hashed-text":
        return [text_vector(data[0])[None]]
    if model.pixels == "tabular":
        return [
            (
                (data - np.asarray(metadata["mean"], np.float32))
                / np.asarray(metadata["std"], np.float32)
            )[None]
        ]
    if model.pixels == "stereo-pair":
        return [np.ascontiguousarray(data[None], np.float32) / 255]
    if model.pixels == "bayer4":
        gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        return [
            np.stack([gray[y::2, x::2] for y, x in ((0, 0), (0, 1), (1, 0), (1, 1))])[
                None
            ]
        ]
    x = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    ddof = int(model.file == "hardnet")
    den = x.std(ddof=ddof) + 1e-6 if ddof else np.sqrt(x.var() + 1e-5)
    return [((x - x.mean()) / den)[None, None]]


def yolo_outputs(model, out):
    from task_decode import grid, detections, xyxy

    boxes = []
    probs = []
    anchors = [
        [(10, 13), (16, 30), (33, 23)],
        [(30, 61), (62, 45), (59, 119)],
        [(116, 90), (156, 198), (373, 326)],
    ]
    for i, stride in enumerate((8, 16, 32)):
        g = grid(640 // stride)
        if model.file == "yolov5-relu":
            rows = out[i].reshape(3, 85, -1).transpose(0, 2, 1)
            for row, anchor in zip(rows, anchors[i]):
                boxes.append(
                    xyxy(
                        (row[:, :2] * 2 - 0.5 + g) * stride,
                        (2 * row[:, 2:4]) ** 2 * anchor,
                    )
                )
                probs.append(row[:, 4:5] * row[:, 5:])
        else:
            d = out[2 * i].reshape(4, 16, -1)
            e = np.exp(d - d.max(1, keepdims=True))
            offsets = (e * np.arange(16)[None, :, None]).sum(1) / e.sum(1)
            boxes.append(
                np.c_[g + 0.5 - offsets[:2].T, g + 0.5 + offsets[2:].T] * stride
            )
            probs.append(out[2 * i + 1].reshape(80, -1).T)
    b = np.concatenate(boxes)
    p = np.concatenate(probs)
    if model.file == "yolov5-relu":
        return detections(b, p.max(1), p.argmax(1), model.size)
    # One-to-one (NMS-free) head: threshold and take the highest scoring box/class pairs.
    ix, cls = np.nonzero(p >= 0.30)
    order = np.argsort(-p[ix, cls])[:300]
    ix, cls = ix[order], cls[order]
    return {"boxes": np.column_stack([cls, p[ix, cls], b[ix] / 640])}


def decode_extended(model, out):
    if model.decoder == "anomaly-score":
        scores = np.sort(out[0].ravel())
        return {"score": float(scores[-max(1, len(scores) // 100) :].mean())}
    if model.decoder == "disparity-map":
        return {"map": out[0][0].argmin(0)}
    if model.decoder == "patch-embedding":
        x = out[0].ravel()
        return {"embedding": x / max(np.linalg.norm(x), 1e-9)}
    if model.decoder == "spam":
        return {"label": int(out[0].argmax())}
    if model.decoder == "regression":
        return {"value": float(out[0].ravel()[0])}
    return yolo_outputs(model, out)


def quality_extended(root, model, rec, p):
    if model.decoder == "anomaly-score":
        return {"anomaly_score": p["score"], "label": rec["label"]}
    if model.decoder == "patch-embedding":
        return {
            "patch": True,
            "subject": rec["subject"],
            "role": rec["role"],
            "embedding": p["embedding"].tolist(),
        }
    if model.decoder == "spam":
        y = rec["label"]
        h = p["label"]
        return {
            "准确率": [int(y == h), 1],
            "垃圾短信 F1": [2 * int(y == h == 1), int(y) + int(h)],
        }
    if model.decoder == "regression":
        return {"MAE": [abs(rec["value"] - p["value"]), 1]}
    gt = np.load(root / rec["disparity"])
    valid = np.isfinite(gt)
    diff = np.abs(p["map"][valid] - gt[valid])
    return {
        "视差 MAE (px)": [float(diff.sum()), int(valid.sum())],
        "误差≤1px": [int((diff <= 1).sum()), int(valid.sum())],
    }


def anomaly_auc(rows):
    normal = np.array([r["anomaly_score"] for r in rows if r["label"] == 0])
    defect = np.array([r["anomaly_score"] for r in rows if r["label"] == 1])
    diff = defect[:, None] - normal
    return {
        "图像 AUROC": float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)
    }
