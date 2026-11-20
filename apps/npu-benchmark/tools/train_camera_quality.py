"""Fit artifact classifiers on transformed Imagenette crops, split by original photograph."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from train_sequence_models import fit_head
from train_wave3_supervised import save, records
from sequence_models import CAMERA


class Energy(nn.Module):
    def __init__(self, filters, stride, padding):
        super().__init__()
        weights = np.stack(filters).astype(np.float32)
        self.bank = nn.Conv2d(
            1,
            len(filters),
            weights.shape[-1],
            stride=stride,
            padding=padding,
            bias=False,
        )
        with torch.no_grad():
            self.bank.weight.copy_(torch.from_numpy(weights[:, None]))

    def forward(self, x):
        y = self.bank(x).abs()
        return torch.cat(
            [
                nn.functional.adaptive_avg_pool2d(y, 1),
                nn.functional.adaptive_avg_pool2d(y * y, 1),
            ],
            1,
        ).flatten(1)


def banks():
    i = np.arange(8)
    basis = np.cos(np.pi * i[:, None] * (i[None] + 0.5) / 8) / 2
    dct = [np.outer(basis[u], basis[v]) for u in i for v in i if u + v > 0]
    derivatives = []
    for step in (1, 2, 3):
        for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            for second in (False, True):
                f = np.zeros((7, 7))
                f[3, 3] = -2 if second else -1
                f[3 + dy * step, 3 + dx * step] = 1
                if second:
                    f[3 - dy * step, 3 - dx * step] = 1
                derivatives.append(f)
    return {"dct": Energy(dct, 8, 0), "derivative": Energy(derivatives, 1, 0)}


def degrade(im, kind, label, rng):
    if kind == "jpeg-level":
        return cv2.imdecode(
            cv2.imencode(
                ".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, (10, 30, 60, 90)[label]]
            )[1],
            0,
        )
    if kind == "noise-level":
        return np.clip(
            np.rint(im + rng.normal(0, (0, 6, 14, 28)[label], im.shape)), 0, 255
        ).astype(np.uint8)
    if kind == "focus-level":
        radius = (0, 2, 4, 6)[label]
        if not radius:
            return im
        kernel = cv2.circle(
            np.zeros((2 * radius + 1,) * 2, np.float32), (radius, radius), radius, 1, -1
        )
    else:
        kernel = np.zeros((11, 11), np.float32)
        p, q = [
            ((0, 5), (10, 5)),
            ((1, 1), (9, 9)),
            ((5, 0), (5, 10)),
            ((1, 9), (9, 1)),
        ][label]
        cv2.line(kernel, p, q, 1, 1)
    return cv2.filter2D(
        im, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REFLECT_101
    )


def make_samples(work, base, ds, split, limit):
    rng = np.random.default_rng(822 if split == "train" else 823)
    imgs = []
    labels = []
    rows = []
    sources = []
    names = sorted(
        (base / split).rglob("*.JPEG"),
        key=lambda p: hashlib.sha256(str(p.relative_to(base)).encode()).digest(),
    )[:limit]
    assert len(names) == limit
    for i, p in enumerate(names):
        im = cv2.imread(str(p), 0)
        h, w = im.shape
        y, x = rng.integers(h - 64 + 1), rng.integers(w - 64 + 1)
        crop = im[y : y + 64, x : x + 64]
        sources.append(
            dict(
                file=str(p.relative_to(base)),
                crop_xy=[int(x), int(y)],
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
        )
        for label in range(4):
            sample = degrade(crop, ds, label, rng)
            imgs.append(sample)
            labels.append(label)
            if split == "val":
                name = f"{ds}/{i}-{label}.png"
                cv2.imwrite(str(work / "fixtures" / name), sample)
                rows.append(
                    dict(file=name, label=label, source=str(p.relative_to(base)))
                )
    return (
        np.array(imgs, dtype=np.float32)[:, None] / 255,
        np.array(labels),
        rows,
        sources,
    )


def texture_net(x, y):
    net = nn.Sequential(
        nn.Conv2d(1, 16, 5, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(16, 24, 3, padding=1),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(24, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 4),
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=0.002)
    images = torch.from_numpy(x).to(dev)
    labels = torch.from_numpy(y).to(dev)
    for _ in range(15):
        for ix in torch.randperm(len(y)).split(64):
            opt.zero_grad()
            nn.functional.cross_entropy(net(images[ix]), labels[ix]).backward()
            opt.step()
    return net.cpu().eval()


def main(work, base):
    torch.set_num_threads(2)
    for ds in CAMERA:
        torch.manual_seed(822)
        (work / "fixtures" / ds).mkdir(exist_ok=True)
        x, y, _, training = make_samples(work, base, ds, "train", 320)
        _, _, rows, testing = make_samples(work, base, ds, "val", 24)
        assert not ({p["sha256"] for p in training} & {p["sha256"] for p in testing})
        records(work, ds, rows)
        (work / "data" / (ds + "-split.json")).write_text(
            json.dumps(dict(train=training, test=testing))
        )
        meta = dict(seed=822, training=len(x), test=len(rows), source="Imagenette2-160")
        for key, feature in banks().items():
            save(work, ds + "-" + key, fit_head(feature, x, y, alpha=0.2), x[:1], meta)
        save(work, ds + "-stackcnn", texture_net(x, y), x[:1], meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("imagenette", type=Path)
    args = p.parse_args()
    main(args.work, args.imagenette)
