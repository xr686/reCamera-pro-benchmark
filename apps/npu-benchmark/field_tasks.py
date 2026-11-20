"""Task-specific decoding and quality measures for optical flow, illumination and tags."""

import numpy as np


def decode_field(model, out):
    x = out[0][0]
    if model.decoder == "flow-cost":
        idx = x.argmin(0)
        return {"flow": np.stack([idx % 7 - 3, idx // 7 - 3], -1)}
    if model.decoder == "flow-vector":
        return {"flow": x.transpose(1, 2, 0)}
    if model.decoder == "illumination":
        return {"light": x.ravel()}
    if model.decoder == "multi-label":
        return {"scores": x.ravel()}
    # Each bin corresponds to an integer lag from 18 to 100 samples at 8 kHz.
    score = x.ravel()
    j = int(score.argmax())
    delta = 0.0
    if 0 < j < len(score) - 1:
        a, b, c = map(float, score[j - 1 : j + 2])
        den = a - 2 * b + c
        if abs(den) > 1e-10:
            delta = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return {"hz": 8000 / (18 + j + delta)}


def quality_field(root, model, rec, p):
    if model.decoder in ("flow-vector", "flow-cost"):
        gt = np.load(root / rec["flow_target"])
        valid = np.isfinite(gt).all(-1)
        d = np.linalg.norm((p["flow"] - gt)[valid], axis=-1)
        return {
            "EPE px": [float(d.sum()), int(d.size)],
            "误差≤1px": [int((d <= 1).sum()), int(d.size)],
        }
    if model.decoder == "illumination":
        x = np.asarray(rec["gains"])
        y = p["light"]
        cos = np.dot(x, y) / max(1e-12, np.linalg.norm(x) * np.linalg.norm(y))
        return {"角误差 °": [float(np.degrees(np.arccos(np.clip(cos, -1, 1)))), 1]}
    if model.decoder == "multi-label":
        return {"tag_scores": p["scores"].tolist(), "targets": rec["labels"]}
    e = abs(1200 * np.log2(p["hz"] / rec["hz"]))
    return {"音分 MAE": [float(e), 1], "50 音分内": [int(e <= 50), 1]}


def tag_map(rows):
    y = np.array([r["targets"] for r in rows])
    sc = np.array([r["tag_scores"] for r in rows])
    aps = []
    for col in range(y.shape[1]):
        labels = y[:, col]
        values = sc[:, col]
        if labels.sum():
            ids = np.argsort(-values, kind="stable")
            hits = labels[ids]
            v = values[ids]
            # Evaluate precision at each distinct threshold (ties share a threshold).
            end = np.r_[np.where(v[:-1] != v[1:])[0], len(v) - 1]
            cumulative = hits.cumsum()[end]
            aps.append(
                np.sum(np.diff(np.r_[0, cumulative]) * cumulative / (end + 1))
                / labels.sum()
            )
    return {"类别宏平均 AP": float(np.mean(aps))}
