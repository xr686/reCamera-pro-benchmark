"""Train pet-breed probes and unprompted foreground segmenters from Oxford-IIIT Pet."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models
from train_sequence_objects import fit_pyramids
from train_sequence_models import fit_head
from train_wave3_supervised import save, records
from sequence_models import PET_BREEDS


class BreedFeatures(nn.Module):
    def __init__(self, key):
        super().__init__()
        if key == "resnet":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Identity()
        elif key == "shuffle":
            model = models.shufflenet_v2_x1_0(
                weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1
            )
            model.fc = nn.Identity()
        else:
            model = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            )
            model.classifier = nn.Identity()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def unpack(archive, root):
    # Read only regular images, annotations and split lists into known destinations.
    with tarfile.open(archive, "r|gz") as z:
        for member in z:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 2 or any(p.startswith(".") for p in parts):
                continue
            name = parts[-1]
            if parts[-2] == "images" and name.endswith(".jpg"):
                dest = root / "images" / name
            elif parts[-2] == "trimaps" and name.endswith(".png"):
                dest = root / "trimaps" / name
            elif parts[-2] == "annotations" and name in ("trainval.txt", "test.txt"):
                dest = root / name
            else:
                continue
            dest.parent.mkdir(exist_ok=True, parents=True)
            dest.write_bytes(z.extractfile(member).read())


def split(work, base, ds, subset, count):
    listing = [
        r.split()[:2]
        for r in (base / (subset + ".txt")).read_text().splitlines()
        if r and not r.startswith("#")
    ]
    quota = np.zeros(37, int)
    imgs = []
    labels = []
    recs = []
    sources = []
    side = 160 if ds == "pet-breed" else 96
    for name, c in sorted(
        listing, key=lambda r: hashlib.sha256(r[0].encode()).digest()
    ):
        label = int(c) - 1
        if quota[label] >= count:
            continue
        path = base / "images" / (name + ".jpg")
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError(name)
        im = cv2.resize(im, (side, side), interpolation=cv2.INTER_AREA)
        rec = dict(file=f"{ds}/{name}.png")
        if ds == "pet-mask":
            # Oxford labels: 1 foreground, 2 background, 3 uncertain boundary.
            m = cv2.imread(str(base / "trimaps" / (name + ".png")), 0)
            assert set(np.unique(m)).issubset({1, 2, 3}) and m.max() >= 2
            m = cv2.resize(m, (side, side), interpolation=cv2.INTER_NEAREST)
            target = np.where(m == 3, 255, m == 1).astype(np.uint8)
            if not (target == 1).any():
                continue
            rec["mask"] = f"{ds}/{name}-mask.png"
            if subset == "test":
                cv2.imwrite(str(work / "fixtures" / rec["mask"]), target)
        else:
            target = label
            rec["label"] = label
        if subset == "test":
            cv2.imwrite(str(work / "fixtures" / rec["file"]), im)
        quota[label] += 1
        imgs.append(im[:, :, ::-1])
        labels.append(target)
        recs.append(rec)
        sources.append(
            dict(
                id=name,
                breed=label,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    assert min(quota) == max(quota) == count
    return (
        np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32) / 255,
        np.array(labels),
        recs,
        sources,
    )


def train(work, archive):
    base = work / "data/pet-files"
    unpack(archive, base)
    torch.set_num_threads(3)
    for ds in ("pet-breed", "pet-mask"):
        (work / "fixtures" / ds).mkdir(exist_ok=True)
        x, y, _, training = split(work, base, ds, "trainval", 32)
        _, _, recs, testing = split(
            work, base, ds, "test", 4 if ds == "pet-breed" else 2
        )
        assert not ({p["sha256"] for p in training} & {p["sha256"] for p in testing})
        records(work, ds, recs)
        meta = dict(training=len(training), test=len(testing))
        (work / "data" / (ds + "-split.json")).write_text(
            json.dumps(dict(train=training, test=testing))
        )
        if ds == "pet-mask":
            fit_pyramids(work, ds, x, y, True, meta)
        else:
            for key in PET_BREEDS:
                features = BreedFeatures(key).eval()
                save(
                    work,
                    ds + "-" + key,
                    fit_head(features, x, y, alpha=20),
                    np.ascontiguousarray(x[:1]),
                    meta,
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("archive", type=Path)
    args = p.parse_args()
    train(args.work, args.archive)
