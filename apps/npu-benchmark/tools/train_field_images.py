"""Train four comparable image classifiers on each labelled domain; keep held-out source IDs."""

import argparse, csv, hashlib, io, json, tarfile, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    squeezenet1_1,
    SqueezeNet1_1_Weights,
)
from sklearn.linear_model import Ridge
from train_wave3_supervised import linear, save, records


def key(s):
    return hashlib.sha256(s.encode()).digest()


def load(work, ds, voc):
    archives = []

    def archive(path):
        z = zipfile.ZipFile(work / "data" / path)
        archives.append(z)
        return z

    def samples(files, train_count, test_count, loader, test_files=None):
        train = []
        test = []
        for c in sorted({c for _, c in files}):
            trainpool = sorted([n for n, k in files if k == c], key=key)
            testpool = (
                sorted([n for n, k in test_files if k == c], key=key)
                if test_files is not None
                else trainpool[train_count:]
            )
            train.extend((p, c) for p in trainpool[:train_count])
            test.extend((p, c) for p in testpool[:test_count])
        return train, test, loader

    if ds == "flowers":
        t = tarfile.open(work / "data/flowers.tgz")
        archives.append(t)
        paths = [
            (m.name, Path(m.name).parent.name)
            for m in t.getmembers()
            if m.name.endswith(".jpg")
        ]
        tr, te, _ = samples(paths, 200, 20, None)
        selected = {f for f, _ in tr + te}
        t.close()
        with tarfile.open(work / "data/flowers.tgz", "r|gz") as stream:
            images = {
                m.name: stream.extractfile(m).read()
                for m in stream
                if m.name in selected
            }
        read = images.__getitem__
    elif ds == "eurosat":
        z = archive("eurosat.zip")
        files = [
            (n, Path(n).parent.name)
            for n in z.namelist()
            if n.lower().endswith(".jpg") and not n.startswith("__")
        ]
        tr, te, read = samples(files, 150, 12, z.read)
    elif ds == "gts":
        train = archive("gts-train.zip")
        test = archive("gts-test.zip")
        labels = archive("gts-labels.zip")
        files = [
            (n, str(int(Path(n).parent.name)))
            for n in train.namelist()
            if n.lower().endswith(".ppm")
        ]
        annot = {
            r["Filename"]: r["ClassId"]
            for r in csv.DictReader(
                io.StringIO(labels.read("GT-final_test.csv").decode()), delimiter=";"
            )
        }
        tests = [
            (n, annot[Path(n).name])
            for n in test.namelist()
            if n.lower().endswith(".ppm")
        ]
        tr, te, read = samples(
            files,
            48,
            4,
            lambda p: (train if p in train.NameToInfo else test).read(p),
            tests,
        )
    else:
        cats = "aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor".split()

        def split(s, per):
            ids = (voc / "ImageSets/Main" / (s + ".txt")).read_text().split()
            quota = np.zeros(len(cats), int)
            data = []
            for f in sorted(ids, key=key):
                yy = np.zeros(len(cats), int)
                for obj in ET.parse(voc / "Annotations" / (f + ".xml")).findall(
                    "object"
                ):
                    yy[cats.index(obj.findtext("name"))] = 1
                if np.any((yy > 0) & (quota < per)):
                    data.append((f, yy.tolist()))
                    quota += yy
                if np.all(quota >= per):
                    break
            return data

        tr, te = split("train", 60), split("val", 7)
        read = lambda p: (voc / "JPEGImages" / (p + ".jpg")).read_bytes()
    cats = cats if ds == "voc-tags" else sorted({c for _, c in tr})

    def process(rows):
        images = []
        targets = []
        for n, label in rows:
            img = cv2.imdecode(np.frombuffer(read(n), np.uint8), cv2.IMREAD_COLOR)
            images.append(
                cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)[:, :, ::-1]
            )
            targets.append(label if ds == "voc-tags" else cats.index(label))
        return np.stack(images), np.asarray(targets)

    x, y = process(tr)
    tx, ty = process(te)
    for z in archives:
        z.close()
    fx = work / "fixtures" / ds
    fx.mkdir(exist_ok=True)
    rows = []
    for i, (n, _) in enumerate(te):
        f = ds + f"/{i}.png"
        cv2.imwrite(str(work / "fixtures" / f), tx[i, :, :, ::-1])
        r = dict(file=f, source=n)
        if ds == "voc-tags":
            r["labels"] = ty[i].tolist()
        else:
            r["label"] = int(ty[i])
        rows.append(r)
    records(work, ds, rows)
    (work / "data" / (ds + "-split.json")).write_text(
        json.dumps(dict(train=[n for n, _ in tr], test=[n for n, _ in te]))
    )
    return (
        torch.tensor(x.transpose(0, 3, 1, 2).copy(), dtype=torch.float32) / 255,
        y,
        cats,
        len(te),
    )


class Features(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "resprobe":
            self.net = nn.Sequential(
                *list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1]
            )
        elif kind == "squeezeprobe":
            self.net = nn.Sequential(
                squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1).features,
                nn.AdaptiveAvgPool2d(1),
            )
        else:
            kernels = []
            for wavelength in (3.0, 5.5):
                for angle in np.arange(6) * np.pi / 6:
                    v = cv2.getGaborKernel(
                        (9, 9), 2.0, angle, wavelength, 0.7, 0, cv2.CV_32F
                    )
                    v -= v.mean()
                    kernels.append(v / max(1e-8, np.linalg.norm(v)))
            self.register_buffer("bank", torch.from_numpy(np.stack(kernels))[:, None])
        self.register_buffer(
            "mu", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "sigma", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, x):
        if self.kind == "gabor":
            gray = x[:, :1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114
            texture = nn.functional.conv2d(gray, self.bank, padding=4).abs()
            x = torch.cat([x, x * x, texture], 1)
            return nn.functional.avg_pool2d(x, 20, 20).flatten(1)
        return self.net((x - self.mu) / self.sigma).flatten(1)


class Branches(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2, 2), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(24, 16, k, padding=k // 2), nn.BatchNorm2d(16), nn.ReLU()
                )
                for k in (1, 3, 5)
            ]
        )
        self.head = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AvgPool2d(5),
            nn.Flatten(),
            nn.Linear(192, n),
        )

    def forward(self, x):
        z = self.stem(x)
        return self.head(torch.cat([m(z) for m in self.branches], 1))


def fit(work, ds, voc):
    torch.set_num_threads(3)
    torch.manual_seed(86)
    x, y, cats, n = load(work, ds, voc)
    meta = dict(classes=cats, training=len(x), test=n, seed=86)
    target = np.eye(len(cats))[y] if y.ndim == 1 else y
    for k in ("gabor", "resprobe", "squeezeprobe"):
        feature = Features(k).eval()
        with torch.no_grad():
            z = torch.cat([feature(t) for t in x.split(48)]).numpy()
        mean = z.mean(0)
        std = z.std(0).clip(0.05)
        model = Ridge(alpha=12.0).fit((z - mean) / std, target)
        w = model.coef_ / std
        b = model.intercept_ - w @ mean
        save(
            work,
            ds + "-" + k,
            nn.Sequential(feature, linear(w, b)),
            x[:1].numpy(),
            meta,
        )
    net = Branches(len(cats))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
    y = torch.from_numpy(y if y.ndim == 1 else y.astype(np.float32))
    for epoch in range(12):
        for ids in torch.randperm(len(x)).split(48):
            est = net(x[ids])
            loss = (
                nn.functional.cross_entropy(est, y[ids])
                if y.ndim == 1
                else nn.functional.binary_cross_entropy_with_logits(est, y[ids])
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    save(work, ds + "-branches", net, x[:1].numpy(), dict(**meta, epochs=12))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("domain")
    p.add_argument("voc", type=Path)
    a = p.parse_args()
    fit(a.work, a.domain, a.voc)
