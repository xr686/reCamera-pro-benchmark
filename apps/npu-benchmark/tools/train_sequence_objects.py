"""Train and evaluate box and mask heads using disjoint VOC2012 Segmentation splits."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

from train_wave3_supervised import save, records
from sequence_models import OBJECTS, OBJECT_HEADS
from sequence_tasks import object_quality


def dataset(work, voc, ds, split, limit):
    names = (voc / "ImageSets/Segmentation" / (split + ".txt")).read_text().split()
    rng = np.random.default_rng(4481 if split == "train" else 4482)
    images = []
    labels = []
    rows = []
    sources = []
    for n in sorted(names, key=lambda v: hashlib.sha256(v.encode()).digest()):
        m = np.array(Image.open(voc / "SegmentationObject" / (n + ".png")))
        ids, size = np.unique(m[(m > 0) & (m < 255)], return_counts=True)
        if not len(ids):
            continue
        obj = m == ids[size.argmax()]
        ys, xs = np.nonzero(obj)
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        if min(x2 - x1, y2 - y1) < 8:
            continue
        h, w = m.shape
        path = voc / "JPEGImages" / (n + ".jpg")
        im = cv2.imread(str(path))
        meta = dict(
            id=n,
            instance=int(ids[size.argmax()]),
            image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        name = f"{ds}/{n}.png"
        rec = dict(file=name)
        if ds == "voc-prompt":
            # Each side expands independently to avoid putting the true boundary at a fixed inset.
            left, top, right, bottom = np.rint(
                rng.uniform(0.05, 0.55, 4) * [x2 - x1, y2 - y1, x2 - x1, y2 - y1]
            ).astype(int)
            l, t, r, b = (
                max(0, x1 - left),
                max(0, y1 - top),
                min(w, x2 + right),
                min(h, y2 + bottom),
            )
            im = im[t:b, l:r]
            tri = np.where(m == 255, 255, obj.astype(np.uint8)).astype(np.uint8)
            target = cv2.resize(
                tri[t:b, l:r], (96, 96), interpolation=cv2.INTER_NEAREST
            )
            meta["prompt_xyxy"] = [int(v) for v in (l, t, r, b)]
            rec["mask"] = f"{ds}/{n}-mask.png"
            if split == "val":
                cv2.imwrite(str(work / "fixtures" / rec["mask"]), target)
        else:
            target = np.array([x1 / w, y1 / h, x2 / w, y2 / h], np.float32)
            rec["bbox"] = target.tolist()
        im = cv2.resize(im, (96, 96), interpolation=cv2.INTER_AREA)
        if split == "val":
            cv2.imwrite(str(work / "fixtures" / name), im)
        images.append(im[:, :, ::-1])
        labels.append(target)
        rows.append(rec)
        sources.append(meta)
        if len(images) >= limit:
            break
    assert len(images) == limit
    return (
        np.stack(images).transpose(0, 3, 1, 2).astype(np.float32) / 255,
        np.stack(labels),
        rows,
        sources,
    )


def stage(cin, cout, kind):
    if kind == "depthwise":
        return nn.Sequential(
            nn.Conv2d(cin, cin, 3, 2, 1, groups=cin),
            nn.BatchNorm2d(cin),
            nn.ReLU(),
            nn.Conv2d(cin, cout, 1),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
        )
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, 2, 1),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, 1, 1),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    )


class Pyramid(nn.Module):
    def __init__(self, kind, mask):
        super().__init__()
        self.mask = mask
        self.pretrained = kind == "fire"
        if self.pretrained:
            fire = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1).features
            self.stages = nn.ModuleList([fire[:2], fire[2:5], fire[5:8]])
            channels = [64, 128, 256]
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
            )
        else:
            channels = [16, 32, 48]
            self.stages = nn.ModuleList(
                [stage(i, o, kind) for i, o in zip([3] + channels, channels)]
            )
        if mask:
            self.lateral = nn.ModuleList([nn.Conv2d(c, 12, 1) for c in channels])
            self.head = nn.Sequential(
                nn.Conv2d(36, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 1, 1)
            )
        else:
            # Fixed bilinear 6x6 grid works for both the stride-8 and SqueezeNet encoders.
            self.project = nn.Conv2d(channels[-1], 16, 1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 6 * 6, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
                nn.Sigmoid(),
            )

    def forward(self, x):
        if self.pretrained:
            x = (x - self.mean) / self.std
        taps = []
        for stage in self.stages:
            x = stage(x)
            taps.append(x)
        if self.mask:
            y = torch.cat(
                [
                    nn.functional.interpolate(
                        proj(v), size=(48, 48), mode="bilinear", align_corners=False
                    )
                    for proj, v in zip(self.lateral, taps)
                ],
                1,
            )
            return nn.functional.interpolate(
                self.head(y), size=(96, 96), mode="bilinear", align_corners=False
            )
        return self.head(
            nn.functional.interpolate(
                self.project(x), size=(6, 6), mode="bilinear", align_corners=False
            )
        )


def train(work, voc):
    torch.set_num_threads(2)
    for ds in OBJECTS:
        (work / "fixtures" / ds).mkdir(exist_ok=True)
        x, y, _, training = dataset(work, voc, ds, "train", 640)
        _, _, rows, testing = dataset(work, voc, ds, "val", 80)
        assert not (
            {p["image_sha256"] for p in training} & {p["image_sha256"] for p in testing}
        )
        records(work, ds, rows)
        mask = ds == "voc-prompt"
        prior = (
            dict(mask=(y == 1).sum(0) > ((y != 255).sum(0) * 0.5))
            if mask
            else dict(box=np.median(y, axis=0))
        )
        baseline = {}
        for rec in rows:
            for key, (value, _) in object_quality(
                work / "fixtures", OBJECTS[ds][1], rec, prior
            ).items():
                baseline.setdefault(key, []).append(value)
        meta = dict(
            training=len(x),
            test=len(rows),
            seed=4481,
            constant_baseline={k: float(np.mean(v)) for k, v in baseline.items()},
        )
        (work / "data" / (ds + "-split.json")).write_text(
            json.dumps(
                dict(
                    training=training,
                    testing=testing,
                    **{"baseline": meta["constant_baseline"]},
                )
            )
        )
        fit_pyramids(work, ds, x, y, mask, meta)


def fit_pyramids(work, ds, x, y, mask, meta):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xx = torch.from_numpy(x).to(dev)
    yy = torch.from_numpy(y).to(dev)
    for kind in OBJECT_HEADS:
        torch.manual_seed(4481)
        net = Pyramid(kind, mask).to(dev)
        opt = torch.optim.AdamW(
            net.parameters(), lr=0.0006 if kind == "fire" else 0.002
        )
        for _ in range(12):
            for ix in torch.randperm(len(x)).split(32):
                pred = net(xx[ix])
                target = yy[ix]
                if mask:
                    valid = target != 255
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        pred[:, 0][valid], target[valid].float()
                    )
                else:
                    loss = nn.functional.smooth_l1_loss(pred, target, beta=0.1)
                opt.zero_grad()
                loss.backward()
                opt.step()
        save(work, ds + "-" + kind, net.eval().cpu(), np.ascontiguousarray(x[:1]), meta)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("voc", type=Path)
    args = p.parse_args()
    train(args.work, args.voc)
