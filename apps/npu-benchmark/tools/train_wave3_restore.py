"""Train RGB demosaicing and JPEG restoration on disjoint Imagenette source photos."""

import argparse, json
from pathlib import Path
import cv2
import torch
import numpy as np
from torch import nn


def mosaic(im):
    raw = im[:, :, 1].copy()
    raw[::2, ::2] = im[::2, ::2, 0]
    raw[1::2, 1::2] = im[1::2, 1::2, 2]
    return raw


def planes(raw):
    return np.stack([raw[y::2, x::2] for y, x in ((0, 0), (0, 1), (1, 0), (1, 1))])


class Bayer(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.interp = nn.Conv2d(
            4, 12, 3, padding=1, padding_mode="replicate", bias=False
        )
        self.interp.weight.data.zero_()
        color = [[0, 1], [1, 2]]
        for c in range(3):
            for y in range(2):
                for x in range(2):
                    positions = [
                        (j, i)
                        for j in range(y - 1, y + 2)
                        for i in range(x - 1, x + 2)
                        if color[j % 2][i % 2] == c
                    ]
                    d = min((j - y) ** 2 + (i - x) ** 2 for j, i in positions)
                    chosen = [
                        (j, i) for j, i in positions if (j - y) ** 2 + (i - x) ** 2 == d
                    ]
                    for j, i in chosen:
                        self.interp.weight.data[
                            4 * c + 2 * y + x,
                            2 * (j % 2) + i % 2,
                            1 + j // 2,
                            1 + i // 2,
                        ] = 1 / len(chosen)
        self.interp.requires_grad_(False)
        self.refine = (
            nn.Sequential(
                nn.Conv2d(4, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 24, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(24, 12, 3, padding=1),
            )
            if residual
            else None
        )

    def forward(self, x):
        y = self.interp(x)
        if self.refine is not None:
            y = y + self.refine(x)
        return nn.functional.pixel_shuffle(y, 2)


class Jpeg(nn.Module):
    def __init__(self, dilated):
        super().__init__()
        self.residual = dilated
        if dilated:
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 24, 3, dilation=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(24, 24, 3, dilation=3, padding=3),
                nn.ReLU(),
                nn.Conv2d(24, 3, 3, padding=1),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 9, padding=4),
                nn.PReLU(64),
                nn.Conv2d(64, 32, 7, padding=3),
                nn.PReLU(32),
                nn.Conv2d(32, 16, 1),
                nn.PReLU(16),
                nn.Conv2d(16, 3, 5, padding=2),
            )

    def forward(self, x):
        y = self.net(x)
        return x + y if self.residual else y


def jpeg(im):
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 15]
    )
    if not ok:
        raise ValueError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)


def run(work, base):
    torch.manual_seed(32)
    torch.set_num_threads(3)
    rng = np.random.default_rng(32)
    original = json.loads((base / "index.json").read_text())["samples"]
    counts = {}
    train = []
    test = []
    for rec in original:
        n = counts.get(rec["label"], 0)
        counts[rec["label"]] = n + 1
        im = cv2.cvtColor(cv2.imread(str(base / rec["file"])), cv2.COLOR_BGR2RGB)
        if n < 2:
            test.append(
                (cv2.resize(im, (96, 96), interpolation=cv2.INTER_AREA), rec["file"])
            )
        elif n >= 10:
            train.append(
                (cv2.resize(im, (96, 96), interpolation=cv2.INTER_AREA), rec["file"])
            )
    training = np.stack([r[0] for r in train])
    splits = {"train": [r[1] for r in train], "test": [r[1] for r in test]}
    (work / "data/restoration-split.json").write_text(json.dumps(splits))
    for ds in ("jpeg", "bayer"):
        fx = work / "fixtures" / ds
        fx.mkdir(exist_ok=True)
        rows = []
        for j, (im, src) in enumerate(test):
            clean = f"{ds}/{j}-clean.png"
            dirty = f"{ds}/{j}.png"
            cv2.imwrite(str(work / "fixtures" / clean), im[:, :, ::-1])
            cv2.imwrite(
                str(work / "fixtures" / dirty),
                mosaic(im) if ds == "bayer" else jpeg(im)[:, :, ::-1],
            )
            rows.append(dict(file=dirty, target=clean, source=src))
        (work / "data" / (ds + "-records.json")).write_text(json.dumps(rows))
        domain = np.stack(
            [
                planes(mosaic(im)) if ds == "bayer" else jpeg(im).transpose(2, 0, 1)
                for im in training
            ]
        )
        x = torch.from_numpy(domain.copy()).float() / 255
        y = torch.from_numpy(training.transpose(0, 3, 1, 2).copy()).float() / 255
        names = (
            ["bayer-linear", "bayer-residual"]
            if ds == "bayer"
            else ["jpeg-arcnn", "jpeg-dilated"]
        )
        for i, name in enumerate(names):
            net = Bayer(bool(i)) if ds == "bayer" else Jpeg(bool(i))
            steps = 0 if name == "bayer-linear" else 500
            if steps:
                opt = torch.optim.Adam(net.parameters(), lr=0.001)
                for step in range(steps):
                    yy, xx = rng.integers(0, 32, size=2) * 2
                    ids = rng.integers(0, len(x), size=12)
                    scale = 2 if ds == "bayer" else 1
                    inp = x[
                        ids,
                        :,
                        yy // scale : (yy + 32) // scale,
                        xx // scale : (xx + 32) // scale,
                    ]
                    pred = net(inp)
                    loss = (
                        (
                            pred[:, :, 3:-3, 3:-3]
                            - y[ids, :, yy + 3 : yy + 29, xx + 3 : xx + 29]
                        )
                        .abs()
                        .mean()
                    )
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            torch.onnx.export(
                net.eval(),
                x[:1],
                work / "sources" / (name + ".onnx"),
                input_names=["image"],
                opset_version=13,
            )
            (work / "sources" / (name + ".json")).write_text(
                json.dumps(
                    dict(
                        training=len(x) if steps else 0,
                        test=len(test),
                        steps=steps,
                        seed=32,
                    )
                )
            )
            print(name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("imagenette", type=Path)
    args = p.parse_args()
    run(args.work, args.imagenette)
