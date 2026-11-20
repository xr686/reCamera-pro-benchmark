"""Train three normal-only industrial anomaly baselines on MVTec AD."""

import argparse, json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class Features(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
        self.first = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.second = m.layer2
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, x):
        z = self.first((x - self.mean) / self.scale)
        # Eight fixed channels across two ResNet scales form a 16x16 feature grid.
        a = nn.functional.avg_pool2d(z, 2)[:, 3:64:16]
        b = self.second(z)[:, 7:128:32]
        return torch.cat([a, b], 1)


class PaDiM(nn.Module):
    def __init__(self, teacher, emb):
        super().__init__()
        self.teacher = teacher
        center = emb.mean(0, keepdim=True)
        self.register_buffer("center", center)
        z = (emb - center).permute(2, 3, 0, 1)
        cov = (z.transpose(2, 3) @ z) / (len(emb) - 1) + torch.eye(8) * 0.05
        whiten = torch.linalg.inv(torch.linalg.cholesky(cov)).permute(2, 3, 0, 1)
        self.register_buffer("whiten", whiten)

    def forward(self, x):
        z = self.teacher(x) - self.center
        terms = [
            (z * self.whiten[i : i + 1]).sum(1, keepdim=True).square() for i in range(8)
        ]
        return sum(terms)


class Student(nn.Module):
    def __init__(self, teacher, channel_std):
        super().__init__()
        self.teacher = teacher
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1),
        )
        self.register_buffer("std", channel_std.clamp_min(0.1))

    def forward(self, x):
        return ((self.teacher(x) / self.std - self.net(x)) ** 2).mean(1, keepdim=True)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in [(3, 16), (16, 24), (24, 8)]:
            layers.extend([nn.Conv2d(a, b, 4, stride=2, padding=1), nn.ReLU()])
        for a, b in [(8, 24), (24, 16), (16, 3)]:
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(a, b, 3, padding=1),
                    nn.ReLU() if b != 3 else nn.Sigmoid(),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return nn.functional.avg_pool2d(
            (self.net(x) - x).abs().mean(1, keepdim=True), 4
        )


def train(work):
    torch.set_num_threads(3)
    torch.manual_seed(59)
    samples = json.loads((work / "data/mvtec-sources.json").read_text())["images"]
    for cat in ("bottle", "grid"):
        coll = []
        rows = []
        group = "mvtec-" + cat
        fx = work / "fixtures" / group
        fx.mkdir(exist_ok=True)
        for rec in (s for s in samples if s["category"] == cat):
            img = cv2.resize(
                cv2.imread(str(work / "data/mvtec" / rec["path"])),
                (128, 128),
                interpolation=cv2.INTER_AREA,
            )
            if rec["split"] == "train":
                assert rec["defect"] == "good"
                coll.append(img[:, :, ::-1])
            else:
                file = f"{group}/{len(rows)}.png"
                cv2.imwrite(str(work / "fixtures" / file), img)
                rows.append(
                    dict(
                        file=file,
                        source=rec["path"],
                        defect=rec["defect"],
                        label=int(rec["defect"] != "good"),
                    )
                )
        x = (
            torch.tensor(
                np.stack(coll).transpose(0, 3, 1, 2).copy(), dtype=torch.float32
            )
            / 255
        )
        teacher = Features().eval().requires_grad_(False)
        with torch.no_grad():
            emb = torch.cat([teacher(t) for t in x.split(8)])
        (work / "data" / (group + "-records.json")).write_text(json.dumps(rows))
        for kind, net in [
            ("padim", PaDiM(teacher, emb)),
            ("student", Student(teacher, emb.std((0, 2, 3), keepdim=True))),
            ("ae", Autoencoder()),
        ]:
            steps = 0 if kind == "padim" else 240
            if steps:
                opt = torch.optim.Adam(net.parameters(), lr=0.002)
                for step in range(steps):
                    ids = torch.randint(len(x), (8,))
                    if kind == "student":
                        loss = ((net.net(x[ids]) - emb[ids] / net.std) ** 2).mean()
                    else:
                        loss = net(x[ids]).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            key = group + "-" + kind
            torch.onnx.export(
                net.eval(),
                x[:1],
                work / "sources" / (key + ".onnx"),
                input_names=["image"],
                opset_version=13,
            )
            (work / "sources" / (key + ".json")).write_text(
                json.dumps(dict(training=len(x), test=len(rows), steps=steps, seed=59))
            )
            print(key, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    train(p.parse_args().work)
