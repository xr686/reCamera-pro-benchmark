"""Train compact signal baselines on the archive's TRAIN partition and export static ONNX."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signal_catalog import SIGNALS


def read_split(archive, name, split):
    with zipfile.ZipFile(archive) as z:
        member = next(
            v
            for v in z.namelist()
            if Path(v).name.lower() == f"{name}_{split}.ts".lower()
        )
        content = z.read(member).decode("utf-8-sig", errors="replace")
    rows, labels = [], []
    for line in content.lower().split("@data", 1)[1].splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        *dims, label = line.strip().split(":")
        rows.append([np.asarray(v.split(","), np.float32) for v in dims])
        labels.append(label.strip())
    return np.asarray(rows, np.float32), np.asarray(labels)


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.core = nn.Sequential(
            nn.Conv2d(c, c, (1, 7), padding=(0, 3)),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        return torch.relu(x + self.core(x))


class Branches(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(c, 8, (1, k), padding=(0, k // 2)) for k in (3, 9, 17)]
        )

    def forward(self, x):
        return torch.cat([torch.relu(b(x)) for b in self.branches], 1)


class Rbf(nn.Module):
    def __init__(self, x, truth, classes):
        super().__init__()
        centers = np.concatenate(
            [
                KMeans(n_clusters=min(3, sum(truth == k)), random_state=31, n_init=5)
                .fit(x[truth == k])
                .cluster_centers_
                for k in range(classes)
            ]
        )
        scale = (
            float(np.median(((centers[:, None] - centers[None]) ** 2).sum(-1))) or 1.0
        )
        phi = np.exp(-((x[:, None] - centers[None]) ** 2).sum(-1) / scale)
        beta = np.linalg.solve(
            phi.T @ phi + 0.01 * np.eye(len(centers)), phi.T @ np.eye(classes)[truth]
        )
        self.register_buffer("centers", torch.from_numpy(centers))
        self.register_buffer("beta", torch.tensor(beta, dtype=torch.float32))
        self.scale = scale

    def forward(self, x):
        delta = x.flatten(1)[:, None] - self.centers
        return torch.exp(-(delta * delta).sum(2) / self.scale) @ self.beta


def builder(kind, c, length, n, x, y):
    if kind in ("lda", "ridge"):
        flat = x.flatten(1).numpy()
        est = (
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            if kind == "lda"
            else RidgeClassifier(alpha=1.0)
        ).fit(flat, y.numpy())
        weights, bias = est.coef_, est.intercept_
        if len(weights) == 1:
            weights, bias = (
                np.vstack([-weights, weights]),
                np.concatenate([-bias, bias]),
            )
        layer = nn.Linear(c * length, n)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(weights))
            layer.bias.copy_(torch.tensor(bias))
        return nn.Sequential(nn.Flatten(), layer).eval(), False
    if kind == "rbf":
        return Rbf(x.flatten(1).numpy(), y.numpy(), n).eval(), False
    if kind == "mlp":
        return nn.Sequential(
            nn.Flatten(), nn.Linear(c * length, 64), nn.ReLU(), nn.Linear(64, n)
        ), True
    layers = [nn.Conv2d(c, 24, (1, 5), padding=(0, 2)), nn.BatchNorm2d(24), nn.ReLU()]
    if kind == "residual":
        layers += [Residual(24), Residual(24)]
    elif kind == "inception":
        layers += [Branches(24), Branches(24)]
    elif kind == "separable":
        for size in (7, 11):
            layers += [
                nn.Conv2d(24, 24, (1, size), groups=24, padding=(0, size // 2)),
                nn.Conv2d(24, 24, 1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
            ]
    else:
        for k, d in ((7, 2), (5, 4)) if kind == "dilated" else ((7, 1), (3, 1)):
            layers += [
                nn.Conv2d(24, 24, (1, k), padding=(0, (k // 2) * d), dilation=(1, d)),
                nn.BatchNorm2d(24),
                nn.ReLU(),
            ]
    return nn.Sequential(
        *layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(24, n)
    ), True


def stratified(labels, limit, rng):
    # Round-robin selection retains minority classes and never duplicates an example.
    groups = [
        rng.permutation(np.flatnonzero(labels == v)).tolist()
        for v in sorted(set(labels))
    ]
    out = []
    while len(out) < limit and any(groups):
        for g in groups:
            if g and len(out) < limit:
                out.append(g.pop())
    return np.asarray(out)


def train(work, name):
    spec = next(s for s in SIGNALS if s[0] == name)
    _, kind, _, c, length = spec
    torch.manual_seed(31)
    torch.set_num_threads(2)
    rng = np.random.default_rng(31)
    archive = work / "data" / (name + ".zip")
    x, y = read_split(archive, name, "TRAIN")
    xt, yt = read_split(archive, name, "TEST")
    if x.shape[1:] != (c, length):
        raise ValueError(f"{name} expected {c, length}, got {x.shape[1:]}")
    if not np.isfinite(x).all() or not np.isfinite(xt).all():
        raise ValueError("Nonfinite data")
    classes = sorted(set(y))
    take = stratified(y, min(1200, len(x)), rng)
    x = x[take]
    y = np.asarray([classes.index(v) for v in y[take]])
    mean = x.mean(axis=(0, 2), keepdims=True)
    std = np.maximum(x.std(axis=(0, 2), keepdims=True), 1e-6)
    x = torch.from_numpy(((x - mean) / std)[:, :, None, :])
    y = torch.from_numpy(y)
    model, learn = builder(kind, c, length, len(classes), x, y)
    if learn:
        opt = torch.optim.Adam(model.parameters(), lr=0.002)
        for _ in range(35):
            for ids in torch.randperm(len(x)).split(48):
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(x[ids]), y[ids])
                loss.backward()
                opt.step()
    stem = "sig-" + name.lower()
    export = work / "sources" / (stem + ".onnx")
    model.eval()
    torch.onnx.export(
        model,
        x[:1],
        export,
        input_names=["signal"],
        output_names=["logits"],
        opset_version=13,
    )
    ids = stratified(yt, min(60, len(xt)), rng)
    folder = work / "fixtures" / stem
    folder.mkdir(exist_ok=True, parents=True)
    records = []
    for i in ids:
        filename = f"{stem}/{int(i)}.npy"
        # Store raw float samples. The board performs standardization in timed preprocessing.
        np.save(work / "fixtures" / filename, xt[i])
        records.append({"file": filename, "label": classes.index(yt[i])})
    provenance = dict(
        mean=mean.ravel().tolist(),
        std=std.ravel().tolist(),
        classes=classes,
        train_count=len(x),
        test_partition=len(xt),
        dataset=name,
        algorithm=kind,
        seed=31,
        source="https://www.timeseriesclassification.com/aeon-toolkit/" + name + ".zip",
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    (work / "data" / (stem + ".json")).write_text(
        json.dumps(dict(records=records, **provenance))
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("dataset")
    args = p.parse_args()
    train(args.work, args.dataset)
