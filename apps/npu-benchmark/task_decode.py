"""Decode RKNN outputs in network coordinates; score annotations in the same space."""

from functools import lru_cache
import cv2
import numpy as np
from kit.runtime.postprocess import db_ocr


def grid(side):
    return np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def xyxy(center, size):
    return np.column_stack((center - size / 2, center + size / 2))


def nms(boxes, scores, labels, threshold=0.3):
    selected = []
    for label in np.unique(labels[scores >= threshold]):
        ids = np.flatnonzero((labels == label) & (scores >= threshold))
        rects = boxes[ids].copy()
        rects[:, 2:] -= rects[:, :2]
        keep = cv2.dnn.NMSBoxes(rects.tolist(), scores[ids].tolist(), threshold, 0.45)
        selected.extend(ids[np.asarray(keep, dtype=int).ravel()])
    return sorted(selected, key=lambda i: -scores[i])


def detections(boxes, scores, classes, size):
    ids = nms(boxes, scores, classes)
    return {
        "boxes": np.column_stack(
            (classes[ids], scores[ids], boxes[ids] / np.tile(size, 2))
        )
    }


def objects(kind, outputs, size):
    if kind == "yolox":
        rows = outputs[0].reshape(-1, 85)
        grids = np.concatenate([grid(size[0] // s) for s in (8, 16, 32)])
        stride = np.concatenate(
            [np.full(((size[0] // s) ** 2, 1), s) for s in (8, 16, 32)]
        )
        boxes = xyxy((rows[:, :2] + grids) * stride, np.exp(rows[:, 2:4]) * stride)
        probs = rows[:, 4:5] * rows[:, 5:]
    else:
        boxes, probs = [], []
        for stride, cls, reg in zip((8, 16, 32), outputs[:3], outputs[3:]):
            reg = reg.reshape(-1, 4, 8)
            exp = np.exp(reg - reg.max(-1, keepdims=True))
            dist = exp @ np.arange(8) / exp.sum(-1) * stride
            center = grid(size[0] // stride) * stride + (stride - 1) / 2
            boxes.append(np.column_stack((center - dist[:, :2], center + dist[:, 2:])))
            probs.append(cls.reshape(-1, 80))
        boxes, probs = np.concatenate(boxes), np.concatenate(probs)
    return detections(boxes, probs.max(1), probs.argmax(1), size)


def instances(outputs, size):
    boxes, scores, classes, coeffs = [], [], [], []
    anchors = [
        [(10, 13), (16, 30), (33, 23)],
        [(30, 61), (62, 45), (59, 119)],
        [(116, 90), (156, 198), (373, 326)],
    ]
    for i, stride in enumerate((8, 16, 32)):
        side = size[0] // stride
        pred = outputs[2 * i].reshape(3, 85, -1).transpose(0, 2, 1)
        coeff = outputs[2 * i + 1].reshape(3, 32, -1).transpose(0, 2, 1)
        prob = pred[:, :, 4] * pred[:, :, 5:].max(-1)
        a, j = np.where(prob > 0.3)
        hits = pred[a, j]
        center = (2 * hits[:, :2] - 0.5 + grid(side)[j]) * stride
        wh = (2 * hits[:, 2:4]) ** 2 * np.array(anchors[i])[a]
        boxes.extend(xyxy(center, wh))
        scores.extend(prob[a, j])
        classes.extend(hits[:, 5:].argmax(1))
        coeffs.extend(coeff[a, j])
    proto = outputs[-1].reshape(32, 160, 160)
    masks = []
    boxes, scores, classes = (
        np.asarray(boxes).reshape(-1, 4),
        np.array(scores),
        np.array(classes),
    )
    ids = nms(boxes, scores, classes)
    for i in ids:
        mask = (np.asarray(coeffs[i]) @ proto.reshape(32, -1)).reshape(160, 160) > 0
        x, y, r, b = np.clip(np.ceil(boxes[i] / 4), 0, 160).astype(int)
        cropped = np.zeros_like(mask)
        cropped[y:b, x:r] = mask[y:b, x:r]
        masks.append(cropped)
    return {
        "boxes": np.column_stack((classes[ids], scores[ids], boxes[ids] / 640)),
        "masks": masks,
    }


def quad_iou(a, b):
    a, b = (cv2.convexHull(np.asarray(x, np.float32).reshape(-1, 2)) for x in (a, b))
    overlap = max(0, float(cv2.intersectConvexConvex(a, b)[0]))
    return overlap / max(1e-12, cv2.contourArea(a) + cv2.contourArea(b) - overlap)


def oriented(output, size):
    rows = output.reshape(-1, 29)
    scores = rows[:, 13] * rows[:, 14:].max(1)
    kept = []
    for i in np.flatnonzero(scores > 0.3)[np.argsort(-scores[scores > 0.3])]:
        row = rows[i]
        cx, cy, w, h = row[:4]
        x, y, r, b = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        u, v, p, q = row[4:8] if row[8] <= 0.9 else (0, 0, 0, 0)
        quad = (
            np.asarray([(x + u * w, y), (r, y + v * h), (r - p * w, b), (x, b - q * h)])
            / size
        )
        cls = int(row[14:].argmax())
        if all(c != cls or quad_iou(quad, other) < 0.45 for c, other in kept):
            kept.append((cls, quad))
    return {"quads": kept}


@lru_cache(maxsize=8)
def priors(w, h, kind):
    if kind == "platebox":
        specs = [
            (8, [10, 16, 24]),
            (16, [32, 48]),
            (32, [64, 96]),
            (64, [128, 192, 256]),
        ]
    else:
        specs = [(8, [16, 32]), (16, [64, 128]), (32, [256, 512])]
    return np.asarray(
        [
            ((x + 0.5) * s, (y + 0.5) * s, k, k)
            for s, sizes in specs
            for y in range(h // s)
            for x in range(w // s)
            for k in sizes
        ],
        np.float32,
    )


def face_boxes(kind, out, size):
    w, h = size
    if kind in ("retina", "platebox"):
        p = priors(w, h, kind)
        loc, conf = out[0].reshape(len(p), -1), out[1].reshape(len(p), 2)[:, 1]
        if kind == "platebox":
            scores = np.sqrt(conf * np.clip(out[2].ravel(), 0, 1))
            offsets = loc[:, [4, 5, 6, 7, 10, 11, 12, 13]].reshape(-1, 4, 2)
            corners = p[:, :2, None].transpose(0, 2, 1) + offsets * 0.1 * p[:, None, 2:]
            boxes = np.column_stack((corners.min(1), corners.max(1)))
            keep = nms(boxes, scores, np.zeros(len(scores), int))
            return {"quads": [(0, corners[i] / size) for i in keep]}
        boxes = xyxy(
            p[:, :2] + loc[:, :2] * 0.1 * p[:, 2:], p[:, 2:] * np.exp(loc[:, 2:] * 0.2)
        )
        return detections(boxes, conf, np.zeros(len(p)), size)
    boxes, scores = [], []
    for i, s in enumerate((8, 16, 32)):
        centers = grid(w // s) * s
        if kind == "yunet":
            prob = np.sqrt(np.clip(out[i].ravel() * out[i + 3].ravel(), 0, 1))
            reg = out[i + 6].reshape(-1, 4)
            box = xyxy(
                centers + reg[:, :2] * s, np.exp(np.clip(reg[:, 2:], -30, 30)) * s
            )
        else:
            n = len(centers) * 2
            # SCRFD builds can permute tensors. Resolve each head by row count and width.
            prob = next(x for x in out if x.size == n and x.shape[-1] == 1).ravel()
            reg = (
                next(x for x in out if x.size == n * 4 and x.shape[-1] == 4).reshape(
                    -1, 4
                )
                * s
            )
            centers = np.repeat(centers, 2, axis=0)
            box = np.column_stack((centers - reg[:, :2], centers + reg[:, 2:]))
        boxes.extend(box)
        scores.extend(prob)
    return detections(
        np.asarray(boxes), np.asarray(scores), np.zeros(len(scores)), size
    )


def mediapipe_boxes(kind, out, size):
    palm = kind == "palm"
    spec = [(24, 2), (12, 6)] if palm else [(28, 2), (14, 2), (7, 6)]
    anchors = np.concatenate(
        [np.repeat((grid(n) + 0.5) / n, count, axis=0) for n, count in spec]
    )
    reg = out[0 if palm else 1].reshape(len(anchors), -1)
    scores = sigmoid(out[1 if palm else 0].ravel())
    boxes = xyxy(anchors * size + reg[:, :2], reg[:, 2:4])
    if kind == "person":
        keep = nms(boxes, scores, np.zeros(len(scores)))
        centers = anchors + reg[:, 4:6] / size
        return {"hip_centers": centers[keep]}
    return detections(boxes, scores, np.zeros(len(scores)), size)


def ctc(ids, alphabet, blank):
    ids = np.asarray(ids).ravel()
    ids = ids[np.r_[True, ids[1:] != ids[:-1]]]
    return "".join(alphabet[n] for n in ids if n != blank).strip()


def decode(model, out, tokens=None, gallery=None):
    if model.decoder == "clip-rank":
        from clip_benchmark import rank

        return rank(out, gallery)
    k, size = model.decoder, model.size
    if k in ("prompt-mask", "dominant-box"):
        from sequence_tasks import decode_object
        return decode_object(k, out)
    if k in ("multi-label", "flow-cost", "flow-vector", "illumination", "pitch"):
        from field_tasks import decode_field
        return decode_field(model,out)
    if k in ("disparity-map", "patch-embedding", "anomaly-score", "spam", "regression", "yolov5-relu", "yolov10n"):
        from extended_tasks import decode_extended
        return decode_extended(model, out)
    if k in (
        "extra-class",
        "voc21",
        "saliency",
        "photometric",
        "stylize",
        "features",
        "lines",
    ):
        from extra_tasks import decode_extra

        return decode_extra(model, out)
    if k in ("yolox", "nanodet"):
        return objects(k, out, size)
    if k == "instances":
        return instances(out, size)
    if k == "gghl":
        return oriented(out[0], size)
    if k in ("retina", "platebox", "yunet", "scrfd"):
        return face_boxes(k, out, size)
    if k in ("palm", "person"):
        return mediapipe_boxes(k, out, size)
    if k == "textboxes":
        return {
            "quads": [
                (0, np.asarray(v["quad"]) / size) for v in db_ocr.decode(out, None)
            ]
        }
    if k == "track":
        score, wh, offset = (x.squeeze() for x in out)
        hann = np.hanning(18)[1:-1]
        window = np.outer(hann, hann)
        y, x = np.unravel_index((score * window).argmax(), score.shape)
        center = (np.array([x, y]) + offset[:, y, x]) / 16
        return {
            "boxes": np.array(
                [
                    [
                        0,
                        score[y, x],
                        *(center - wh[:, y, x] / 2),
                        *(center + wh[:, y, x] / 2),
                    ]
                ]
            )
        }
    if k in ("pose", "hand", "mesh"):
        dims = {"pose": 5, "hand": 3, "mesh": 3}[k]
        tensor = next(t for t in out if t.size == 1404) if k == "mesh" else out[0]
        return {
            "points": tensor.reshape(-1, dims)[:, :2] / size,
            "presence": float(out[1].ravel()[0]) if k == "hand" else 1.0,
        }
    if k == "semantic":
        return {"mask": out[0][0].argmax(axis=0).astype(np.uint8)}
    if k == "depth":
        x = out[0].squeeze()
        return {"map": cv2.normalize(x, None, 0, 1, cv2.NORM_MINMAX)}
    if k in ("embedding", "binary-hash"):
        x = out[0].ravel()
        if k == "binary-hash":
            x = np.where(x > 0, np.float32(1), np.float32(-1))
        return {"vector": x / max(np.linalg.norm(x), 1e-12)}
    if k == "sound":
        return {"class": int(out[2].reshape(-1, 521).mean(0).argmax())}
    if k == "class":
        return {"class": int(out[0].argmax())}
    if k == "scalar":
        return {"score": float(out[0].item())}
    if k == "plate":
        alphabet = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZIO-"
        return {"text": ctc(out[0][0].argmax(0), alphabet, len(alphabet) - 1)}
    if k == "word":
        return {
            "text": ctc(
                out[0].reshape(-1, 37).argmax(1),
                "_0123456789abcdefghijklmnopqrstuvwxyz",
                0,
            )
        }
    if k == "edge":
        return {
            "edge": cv2.normalize(
                sigmoid(out[-1].squeeze()), None, 0.0, 1.0, cv2.NORM_MINMAX
            )
        }
    if k == "subword":
        ids = out[0].reshape(-1, len(tokens)).argmax(1)
        return {
            "text": ctc(ids, tokens, len(tokens) - 1).replace("▁", " ").strip().upper()
        }
    if k == "speech":
        alphabet = ["", "", "", "", " "] + list(
            "ETAONIHSRDLUMWCF GYPBVK'XJQZ".replace(" ", "")
        )
        return {"text": ctc(out[0].reshape(-1, 32).argmax(1), alphabet, 0)}
    if k in ("restore", "inpaint"):
        image = out[0].squeeze(0).transpose(1, 2, 0)
        if k == "restore":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) * 255
        return {"image": np.clip(image, 0, 255).astype(np.uint8)}
    if k == "sr":
        return {"image": np.clip(out[0].squeeze(), 0, 1)}
    raise ValueError(k)
