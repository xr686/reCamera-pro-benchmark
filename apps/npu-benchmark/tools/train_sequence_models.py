"""Fit three sequence transforms using only the official TRAIN partition."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

from train_signal_suite import read_split, stratified
from train_wave3_supervised import linear, records, save
from sequence_models import SERIES


class CosineBasis(nn.Module):
    def __init__(self, length):
        super().__init__()
        pos = np.arange(length)[None] + 0.5
        freq = np.arange(1, min(length, 97))[:, None]
        self.project = linear(
            np.cos(np.pi * freq * pos / length) / np.sqrt(length), np.zeros(len(freq))
        )

    def forward(self, x):
        coeff = self.project(x.flatten(1))
        return torch.cat((coeff, coeff.abs()), 1)


class Shapelets(nn.Module):
    def __init__(self, x, y, rng):
        super().__init__()
        self.banks = nn.ModuleList()
        for width in (9, 25, 49):
            templates = []
            for label in np.unique(y):
                for _ in range(4):
                    row = x[rng.choice(np.flatnonzero(y == label)), 0, 0]
                    start = rng.integers(len(row) - width + 1)
                    templates.append(row[start : start + width])
            t = torch.from_numpy(np.stack(templates))
            bank = nn.Conv2d(1, len(t), (1, width))
            with torch.no_grad():
                bank.weight.copy_((-2 * t / width)[:, None, None])
                bank.bias.copy_(t.square().mean(1))
            self.banks.append(bank)

    def forward(self, x):
        energy = x.square()
        values = []
        for bank in self.banks:
            cost = bank(x) + nn.functional.avg_pool2d(
                energy, bank.kernel_size, stride=1
            )
            values.append(cost.amin((2, 3)))
        return torch.cat(values, 1)


class RandomConv(nn.Module):
    def __init__(self, rng):
        super().__init__()
        self.banks = nn.ModuleList()
        for dilation in (1, 3, 7):
            weights = rng.normal(size=(32, 9))
            weights -= weights.mean(1, keepdims=True)
            weights /= np.linalg.norm(weights, axis=1, keepdims=True)
            bank = nn.Conv2d(1, 32, (1, 9), dilation=(1, dilation))
            with torch.no_grad():
                bank.weight.copy_(torch.tensor(weights)[:, None, None, :])
                bank.bias.copy_(torch.tensor(rng.uniform(-0.5, 0.5, size=32)))
            self.banks.append(bank)

    def forward(self, x):
        features = []
        for bank in self.banks:
            y = torch.relu(bank(x))
            features.extend(
                [
                    nn.functional.adaptive_max_pool2d(y, 1),
                    nn.functional.adaptive_avg_pool2d(y, 1),
                ]
            )
        return torch.cat(features, 1).flatten(1)


def fit_head(features, x, y, alpha=2.0):
    with torch.no_grad():
        z = np.concatenate(
            [features(t).numpy() for t in torch.from_numpy(x).split(32)]
        ).astype(np.float64)
    center = z.mean(0)
    scale = z.std(0).clip(0.02)
    estimator = Ridge(alpha=alpha).fit((z - center) / scale, np.eye(int(max(y)) + 1)[y])
    weight = estimator.coef_ / scale
    return nn.Sequential(
        features, linear(weight, estimator.intercept_ - weight @ center)
    )


def train(work, ds):
    archive = work / "data" / (ds + ".zip")
    x, labels = read_split(archive, ds, "TRAIN")
    test, yt = read_split(archive, ds, "TEST")
    classes, y = np.unique(labels, return_inverse=True)
    assert x.shape[1:] == test.shape[1:] == (1, SERIES[ds][1])
    assert np.isfinite(x).all() and np.isfinite(test).all()
    known = {row.tobytes() for row in x}
    keep = []
    for i, row in enumerate(test):
        ident = row.tobytes()
        if ident not in known:
            keep.append(i)
            known.add(ident)
    rng = np.random.default_rng(493)
    keep = np.asarray(keep)
    chosen = keep[stratified(yt[keep], min(len(keep), 60), rng)]
    assert set(yt[chosen]) == set(classes)
    mu = float(x.mean())
    sd = max(float(x.std()), 1e-6)
    z = ((x - mu) / sd)[:, :, None, :]
    meta = dict(
        normalization=dict(mean=[mu], std=[sd]),
        classes=classes.tolist(),
        train=len(x),
        test=len(chosen),
        archive_test=len(test),
        unique_test_after_dedup=len(keep),
        seed=493,
    )
    root = work / "fixtures" / ds
    root.mkdir(exist_ok=True)
    entries = []
    for i in chosen:
        np.save(root / f"{i}.npy", test[i])
        entries.append(dict(file=f"{ds}/{i}.npy", label=classes.tolist().index(yt[i])))
    records(work, ds, entries)
    (work / "data" / (ds + "-split.json")).write_text(
        json.dumps(dict(**meta, test_rows=chosen.tolist()))
    )
    for key, features in [
        ("cosine", CosineBasis(z.shape[-1])),
        ("randomconv", RandomConv(rng)),
        ("shapelets", Shapelets(z, y, rng)),
    ]:
        save(work, ds.lower() + "-" + key, fit_head(features, z, y), z[:1], meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    args = p.parse_args()
    torch.set_num_threads(2)
    for ds in SERIES:
        train(args.work, ds)
