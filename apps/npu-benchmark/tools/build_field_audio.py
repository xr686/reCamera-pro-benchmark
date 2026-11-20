"""Benchmark tone identification and fundamental frequency estimation on labelled synthetic waveforms."""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_wave3_supervised import records, save

LOW = np.array([697, 770, 852, 941])
HIGH = np.array([1209, 1336, 1477, 1633])
TIME = np.arange(1024) / 8000


def dtmf(rng, label):
    a, b = LOW[label // 4], HIGH[label % 4]
    w = np.zeros(1024)
    for f in (a, b):
        w += rng.uniform(0.18, 0.48) * np.sin(
            2 * np.pi * f * (1 + rng.uniform(-0.012, 0.012)) * TIME
            + rng.uniform(0, 2 * np.pi)
        )
    return (w + rng.normal(0, rng.uniform(0.01, 0.15), len(w))).astype(np.float32)


class ToneBank(nn.Module):
    def __init__(self, learned):
        super().__init__()
        weights = []
        for f in np.r_[LOW, HIGH]:
            for ratio in (0.99, 1.0, 1.01):
                theta = 2 * np.pi * f * ratio * TIME
                weights.extend(
                    [np.hanning(1024) * y / 20 for y in (np.cos(theta), np.sin(theta))]
                )
        self.proj = nn.Linear(1024, len(weights), bias=False)
        self.proj.weight.data.copy_(torch.tensor(np.stack(weights)).float())
        self.proj.requires_grad_(False)
        self.head = (
            nn.Sequential(nn.Linear(8, 40), nn.ReLU(), nn.Linear(40, 16))
            if learned
            else None
        )

    def forward(self, x):
        z = (
            self.proj(x.flatten(1)).reshape(1, 8, 3, 2)
            if not self.training
            else self.proj(x.flatten(1)).reshape(-1, 8, 3, 2)
        )
        energy = (z * z).sum(-1).max(-1).values
        if self.head is not None:
            return self.head(energy)
        return (energy[:, :4, None] + energy[:, None, 4:]).flatten(1)


class Pitch(nn.Module):
    def __init__(self, yin):
        super().__init__()
        self.yin = yin
        self.register_buffer("delays", torch.eye(101).reshape(101, 1, 1, 101))
        self.register_buffer(
            "average", torch.tril(torch.ones(100, 100)) / torch.arange(1, 101)[:, None]
        )
        self.register_buffer("penalty", torch.linspace(0, 0.055, 83))

    def forward(self, x):
        taps = F.conv2d(x, self.delays)
        a = taps[:, :1]
        shift = taps[:, 1:]
        pool = lambda y: F.adaptive_avg_pool2d(y, 1).flatten(1)
        if self.yin:
            diff = pool((shift - a) ** 2)
            cumulative = F.linear(diff, self.average)
            values = -diff / (cumulative + 0.0001)
        else:
            values = (
                pool(a * shift) / (pool(a * a) * pool(shift * shift) + 0.00001).sqrt()
            )
        return values[:, 17:] - self.penalty


def run(w):
    torch.manual_seed(217)
    torch.set_num_threads(2)
    rng = np.random.default_rng(217)
    net = ToneBank(True)
    labels = np.tile(np.arange(16), 100)
    x = torch.from_numpy(np.stack([dtmf(rng, i) for i in labels]))[:, None, None]
    y = torch.tensor(labels)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    for _ in range(24):
        for ix in torch.randperm(len(x)).split(64):
            loss = F.cross_entropy(net(x[ix]), y[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
    rng = np.random.default_rng(2137)
    for ds in ("dtmf", "pitch"):
        path = w / "fixtures" / ds
        path.mkdir(exist_ok=True)
        rows = []
        for i in range(96):
            if ds == "dtmf":
                label = i % 16
                wave = dtmf(rng, label)
                meta = dict(label=int(label))
            else:
                hz = rng.uniform(90, 410)
                harm = rng.uniform(0.25, 1.2, 5) / np.sqrt(np.arange(1, 6))
                if i % 3 == 0:
                    harm[0] = 0
                wave = sum(
                    amp * np.sin(2 * np.pi * k * hz * TIME + rng.uniform(0, 2 * np.pi))
                    for k, amp in enumerate(harm, 1)
                )
                wave = (
                    wave / max(abs(wave)) * 0.7
                    + rng.normal(0, rng.uniform(0.015, 0.10), 1024)
                ).astype(np.float32)
                meta = dict(hz=hz, missing_fundamental=bool(i % 3 == 0))
            np.save(path / f"{i}.npy", wave)
            rows.append(dict(file=f"{ds}/{i}.npy", **meta))
        records(w, ds, rows)
    for id, model in [
        ("dtmf-filter", ToneBank(False)),
        ("dtmf-mlp", net),
        ("pitch-yin", Pitch(True)),
        ("pitch-acf", Pitch(False)),
    ]:
        save(
            w,
            id,
            model,
            np.zeros((1, 1, 1, 1024), np.float32),
            dict(test=96, rate=8000, training=1600 if id == "dtmf-mlp" else 0),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    run(p.parse_args().work)
