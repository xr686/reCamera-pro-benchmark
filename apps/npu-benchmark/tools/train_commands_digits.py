"""Train DS-CNN/TC-ResNet with disjoint speakers, and a LeNet digit baseline."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import cv2
import numpy as np
import torch
from torch import nn
from scipy.io import wavfile
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_tasks import log_mel
from train_signal_suite import Residual


class TCResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(40, 32, (1, 5), padding=(0, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            Residual(32),
            Residual(32),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 8),
        )

    def forward(self, x):
        return self.layers(x.transpose(1, 2))


def ds_cnn():
    layers = [
        nn.Conv2d(1, 48, (7, 5), stride=2, padding=(3, 2)),
        nn.BatchNorm2d(48),
        nn.ReLU(),
    ]
    for _ in range(4):
        layers.extend(
            [
                nn.Conv2d(48, 48, 3, padding=1, groups=48),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.Conv2d(48, 48, 1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
            ]
        )
    return nn.Sequential(
        *layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(48, 8)
    )


def fit(w, name, model, values, labels, epochs):
    torch.manual_seed(77)
    torch.set_num_threads(3)
    x = torch.from_numpy(values.astype(np.float32))
    y = torch.tensor(labels)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    for _ in range(epochs):
        for ids in torch.randperm(len(y)).split(64):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x[ids]), y[ids])
            loss.backward()
            opt.step()
    torch.onnx.export(
        model.eval(),
        x[:1],
        w / "sources" / (name + ".onnx"),
        opset_version=13,
        input_names=["features"],
    )


def run(w):
    torch.manual_seed(77)
    torch.set_num_threads(3)
    ds = load_digits()
    ids = np.arange(len(ds.target))
    tr, te = train_test_split(ids, test_size=0.25, stratify=ds.target, random_state=77)
    imgs = np.asarray(
        [
            cv2.resize(
                np.uint8(np.rint(z * 255 / 16)),
                (32, 32),
                interpolation=cv2.INTER_LINEAR,
            )
            for z in ds.images
        ]
    )
    records = []
    (w / "fixtures/digits").mkdir(parents=True, exist_ok=True)
    for label in range(10):
        for i in te[ds.target[te] == label][:8]:
            f = f"digits/{i}.png"
            cv2.imwrite(str(w / "fixtures" / f), imgs[i])
            records.append(dict(file=f, label=label))
    (w / "data/digits.json").write_text(
        json.dumps(dict(records=records, train_count=len(tr), seed=77))
    )
    net = nn.Sequential(
        nn.Conv2d(1, 6, 5),
        nn.Tanh(),
        nn.AvgPool2d(2),
        nn.Conv2d(6, 16, 5),
        nn.Tanh(),
        nn.AvgPool2d(2),
        nn.Flatten(),
        nn.Linear(400, 120),
        nn.Tanh(),
        nn.Linear(120, 84),
        nn.Tanh(),
        nn.Linear(84, 10),
    )
    fit(w, "lenet-digits", net, imgs[tr, None] / 255, ds.target[tr], 30)
    (w / "fixtures/commands").mkdir(parents=True, exist_ok=True)
    records = []
    features = []
    labels = []
    classes = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
    train_speakers = set()
    test_speakers = set()
    with zipfile.ZipFile(w / "data/mini_speech_commands.zip") as archive:
        for label, word in enumerate(classes):
            train = test = 0
            names = sorted(
                n
                for n in archive.namelist()
                if n.startswith("mini_speech_commands/" + word + "/")
                and n.endswith(".wav")
            )
            np.random.default_rng(53).shuffle(names)
            for n in names:
                speaker = Path(n).stem.split("_nohash_")[0]
                held = (
                    int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) % 5 == 0
                )
                if (held and test >= 12) or (not held and train >= 200):
                    continue
                rate, pcm = wavfile.read(io.BytesIO(archive.read(n)))
                assert rate == 16000
                wave = pcm.astype(np.float32) / 32768
                if held:
                    f = f"commands/{word}-{Path(n).stem}.npy"
                    np.save(w / "fixtures" / f, wave)
                    records.append(dict(file=f, label=label, speaker=speaker))
                    test += 1
                    test_speakers.add(speaker)
                else:
                    labels.append(label)
                    features.append(log_mel(wave))
                    train += 1
                    train_speakers.add(speaker)
                if train == 200 and test == 12:
                    break
    assert not train_speakers & test_speakers
    (w / "data/commands.json").write_text(
        json.dumps(
            dict(
                records=records,
                classes=classes,
                train_count=len(features),
                train_speakers=len(train_speakers),
                test_speakers=len(test_speakers),
                seed=77,
            )
        )
    )
    features = np.asarray(features)[:, None]
    for name, model in (("ds-kws", ds_cnn()), ("tc-kws", TCResNet())):
        fit(w, name, model, features, labels, 22)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    run(p.parse_args().work)
