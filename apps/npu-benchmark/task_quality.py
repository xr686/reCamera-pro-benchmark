"""Per-fixture quality measurements. Timing repeats never increase the quality denominator."""

import cv2
import numpy as np
from task_decode import quad_iou

POSE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(
        0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])
    ).prod()
    area = lambda v: np.maximum(0, v[2:] - v[:2]).prod()
    return float(intersection / max(1e-12, area(a) + area(b) - intersection))


def matches(pred, true, iou):
    used = set()
    hits = 0
    for cls, coords in pred:
        candidates = [
            (iou(coords, other), j)
            for j, (label, other) in enumerate(true)
            if j not in used and cls == label
        ]
        if candidates:
            value, j = max(candidates)
            if value >= 0.5:
                hits += 1
                used.add(j)
    return [hits, len(pred), len(true)]


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a):
        cur = [i + 1]
        for j, y in enumerate(b):
            cur.append(min(cur[-1] + 1, prev[j + 1] + 1, prev[j] + (x != y)))
        prev = cur
    return prev[-1]


def quality(root, m, rec, p):
    k = m.decoder
    if k in ("prompt-mask", "dominant-box"):
        from sequence_tasks import object_quality
        return object_quality(root, k, rec, p)
    if k in ("multi-label", "flow-cost", "flow-vector", "illumination", "pitch"):
        from field_tasks import quality_field
        return quality_field(root,m,rec,p)
    if k in ("disparity-map", "patch-embedding", "anomaly-score", "spam", "regression"):
        from extended_tasks import quality_extended
        return quality_extended(root,m,rec,p)
    if k in (
        "extra-class",
        "voc21",
        "saliency",
        "photometric",
        "stylize",
        "features",
        "lines",
        "clip-rank",
    ):
        from extra_tasks import quality_extra

        return quality_extra(root, m, rec, p)
    if k in ("word", "plate", "speech", "subword"):
        text = p["text"]
        if k in ("speech", "subword"):
            return {
                "WER": [
                    edit_distance(rec["text"].split(), text.split()),
                    len(rec["text"].split()),
                ],
                "prediction": text,
            }
        return {"整串正确率": [int(text == rec["text"]), 1], "prediction": text}
    if k == "class" and "live" in rec:
        return {
            "活体二分类正确率": [int((p["class"] == 1) == bool(rec["live"])), 1],
            "prediction": p["class"],
        }
    if k == "scalar":
        return {"pair": rec["pair"], "blurred": rec["blurred"], "score": p["score"]}
    if k == "mesh":
        points = p["points"][[33, 133, 362, 263, 1, 61, 291]]
        gt = np.array(rec["facepoints"])
        iod = np.linalg.norm((gt[0] + gt[1] - gt[2] - gt[3]) / 2)
        return {
            "7 点 NME / 眼距": [
                float(np.linalg.norm(points - gt, axis=1).mean() / max(iod, 1e-8)),
                1,
            ]
        }
    if k == "palm":
        return {"单手图检出率": [int(len(p["boxes"]) > 0), 1]}
    if k == "edge":
        gt = (
            cv2.resize(
                cv2.imread(str(root / rec["edges"]), 0),
                p["edge"].shape[::-1],
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        pred = p["edge"] > 0.5
        kernel = np.ones((5, 5), np.uint8)
        near_gt = cv2.dilate(gt.astype(np.uint8), kernel) > 0
        near_pred = cv2.dilate(pred.astype(np.uint8), kernel) > 0
        precision = (pred & near_gt).sum() / max(1, pred.sum())
        recall = (gt & near_pred).sum() / max(1, gt.sum())
        return {
            "边缘 F1±2px": [
                float(2 * precision * recall / max(1e-9, precision + recall)),
                1,
            ]
        }
    if k == "semantic" and "citymask" in rec:
        gt = cv2.resize(
            cv2.imread(str(root / rec["citymask"]), 0),
            p["mask"].shape[::-1],
            interpolation=cv2.INTER_NEAREST,
        )
        # CamVid index -> comparable Cityscapes trainId. Ignore other GT classes.
        mapping = {0: 10, 1: 2, 2: 5, 3: 0, 4: 1, 5: 8, 6: 7, 7: 4, 8: 13, 9: 11}
        valid = np.isin(gt, list(mapping))
        numer = 0
        denom = 0
        for cid, target in mapping.items():
            actual = gt == cid
            pred = (p["mask"] == target) & valid
            union = (actual | pred).sum()
            if union:
                numer += (actual & pred).sum() / union
                denom += 1
        return {"跨数据集类别 IoU 均值": [float(numer), int(denom)]}
    if k == "sound" or (
        k == "class" and "expression" in rec and "expression" in m.file
    ):
        label = rec["label"] if k == "sound" else rec["expression"]
        return {"Top-1": [int(label == p["class"]), 1], "prediction": p["class"]}
    if k == "person":
        hips = np.array(rec["points"])[[11, 12]]
        if np.any(hips[:, 2] <= 0):
            return {}
        candidates = p["hip_centers"]
        center = hips[:, :2].mean(0)
        hit = len(candidates) > 0 and np.linalg.norm(candidates[0] - center) <= 0.10
        return {"首选髋中心 PCK@0.10 ROI": [int(hit), 1]}
    if k in ("retina", "scrfd", "yunet", "yolox", "nanodet", "yolov5-relu", "yolov10n"):
        gt = [(int(row[0]), row[1:]) for row in rec["boxes"]]
        return {
            "match50": matches(
                [(int(row[0]), row[2:]) for row in p["boxes"] if "classes" not in rec or int(row[0]) in rec["classes"]], gt, box_iou
            )
        }
    if k in ("textboxes", "gghl", "platebox"):
        cls = 6 if k == "gghl" else 0
        return {
            "match50": matches(
                [(c, q) for c, q in p["quads"] if c == cls],
                [(cls, q) for q in rec["quads"]],
                quad_iou,
            )
        }
    if k == "track":
        return {"平均框 IoU": [box_iou(p["boxes"][0, 2:], rec["box"]), 1]}
    if k in ("instances", "semantic") and "mask" in rec:
        gt = cv2.imread(str(root / rec["mask"]), cv2.IMREAD_GRAYSCALE)
        if k == "instances":
            pred = np.zeros((160, 160), np.uint8)
            for box, mask in zip(p["boxes"], p["masks"]):
                if box[0] == 0:
                    pred |= mask.astype(np.uint8)
        else:
            pred = p["mask"]
        gt = cv2.resize(gt, pred.shape[::-1], interpolation=cv2.INTER_NEAREST)
        pred = (pred == 1) & (gt != 255)
        scores = {
            "人像前景 IoU": [
                int((pred & (gt == 1)).sum()),
                int((pred | (gt == 1)).sum()),
            ]
        }
        if k == "instances":
            objects = cv2.resize(
                cv2.imread(str(root / rec["objectmask"]), 0),
                pred.shape[::-1],
                interpolation=cv2.INTER_NEAREST,
            )
            true = [
                (0, (objects == i) & (gt != 255))
                for i in np.unique(objects[gt == 1])
                if i not in (0, 255)
            ]
            proposed = [
                (0, mask & (gt != 255))
                for mask, box in zip(p["masks"], p["boxes"])
                if int(box[0]) == 0
            ]
            scores["match50"] = matches(
                proposed, true, lambda a, b: np.sum(a & b) / max(1, np.sum(a | b))
            )
        return scores
    if k == "depth":
        gt = np.load(root / rec["disparity"])
        pred = cv2.resize(p["map"], gt.shape[::-1])[::4, ::4]
        gt = gt[::4, ::4]
        valid = np.isfinite(gt) & (gt > 0)
        return {
            "与真值视差 Pearson r": [
                float(np.corrcoef(gt[valid], pred[valid])[0, 1]),
                1,
            ]
        }
    if k in ("restore", "inpaint"):
        gt = cv2.resize(cv2.imread(str(root / rec["restore"])), m.size)
        err = (gt.astype(float) - p["image"]) / 255
        if k == "inpaint":
            valid = (
                cv2.resize(
                    cv2.imread(str(root / rec["hole"]), 0),
                    m.size,
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
            err = err[valid]
        return {
            ("遮挡区 PSNR dB" if k == "inpaint" else "PSNR dB"): [
                float(-10 * np.log10(max(np.mean(err**2), 1e-12))),
                1,
            ]
        }
    if k == "sr":
        gt = (
            cv2.imread(str(root / rec["target"]), cv2.IMREAD_GRAYSCALE).astype(
                np.float32
            )
            / 255
        )
        e = (gt - p["image"])[3:-3, 3:-3]
        return {
            "Y-PSNR dB": [float(-10 * np.log10(max(float(np.mean(e**2)), 1e-12))), 1]
        }
    if k == "pose":
        gt = np.array(rec["points"])
        valid = gt[:, 2] > 0
        error = np.linalg.norm(p["points"][POSE_TO_COCO] - gt[:, :2], axis=1)
        return {"PCK@0.10 ROI": [int(((error <= 0.1) & valid).sum()), int(valid.sum())]}
    if k == "hand":
        xy = p["points"]
        extended = []
        for start in (5, 9, 13, 17):
            a = xy[start] - xy[start + 1]
            b = xy[start + 3] - xy[start + 1]
            extended.append(np.dot(a, b) < -0.6 * np.linalg.norm(a) * np.linalg.norm(b))
        gesture = {0: "rock", 4: "paper"}.get(
            sum(extended),
            "scissors" if extended == [True, True, False, False] else "unknown",
        )
        return {
            "三手势正确率": [int(gesture == rec["gesture"] and p["presence"] > 0.5), 1],
            "prediction": gesture,
        }
    if k in ("embedding", "binary-hash") and "subject" in rec:
        return {
            "subject": rec["subject"],
            "role": rec.get("role"),
            "embedding": p["vector"].tolist(),
        }
    return {}


def summarize(rows):
    if rows and "tag_scores" in rows[0]:
        from field_tasks import tag_map
        return tag_map(rows)
    if rows and ("pair_info" in rows[0] or "voc_confusion" in rows[0]):
        from extra_tasks import summarize_extra

        return summarize_extra(rows)
    if rows and "anomaly_score" in rows[0]:
        from extended_tasks import anomaly_auc
        return anomaly_auc(rows)
    measurements = {}
    for row in rows:
        for key, value in row.items():
            if key in (
                "prediction",
                "clip_projection",
                "embedding",
                "subject",
                "role",
                "pair",
                "blurred",
                "score",
                "match50",
                "patch",
            ):
                continue
            measurements.setdefault(key, []).append(value)
    result = {
        k: float(sum(v[0] for v in values) / sum(v[1] for v in values))
        for k, values in measurements.items()
        if sum(v[1] for v in values)
    }
    if rows and "match50" in rows[0]:
        tp, p, g = np.sum([r["match50"] for r in rows], axis=0)
        result["F1@IoU0.5"] = float(2 * tp / max(1, p + g))
    if rows and "subject" in rows[0]:
        ids = [r["subject"] for r in rows]
        vectors = np.array([r["embedding"] for r in rows])
        sim = vectors @ vectors.T
        np.fill_diagonal(sim, -np.inf)
        if rows[0].get("role"):
            sim[:, [i for i, r in enumerate(rows) if r["role"] != "gallery"]] = -np.inf
            eligible = [i for i, r in enumerate(rows) if r["role"] == "query"]
        else:
            eligible = [i for i, label in enumerate(ids) if ids.count(label) > 1]
        if eligible:
            result["图像块检索 Rank-1" if rows[0].get("role") and rows[0].get("patch") else "同身份 Rank-1"] = sum(
                ids[i] == ids[sim[i].argmax()] for i in eligible
            ) / len(eligible)
    if rows and "blurred" in rows[0]:
        pairs = {}
        for row in rows:
            pairs.setdefault(row["pair"], {})[row["blurred"]] = row["score"]
        result["清晰图质量分更高比例"] = sum(
            p[False] > p[True] for p in pairs.values()
        ) / len(pairs)
    return result
