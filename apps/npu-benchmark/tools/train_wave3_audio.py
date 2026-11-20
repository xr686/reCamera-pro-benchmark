"""Fit speaker retrieval and clip-level speech-presence baselines with source-separated tests."""

from collections import defaultdict
import argparse, hashlib, io, json, sys, zipfile
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_tasks import log_mel
from train_wave3_supervised import linear, save, records


class ResidualSpeaker(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(1, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU()
        )
        self.res = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
        )
        self.pool = nn.Sequential(
            nn.Conv2d(24, 24, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 25)),
            nn.Flatten(),
            nn.Linear(240, 48),
        )

    def forward(self, x):
        h = self.embed(x / 12)
        return self.pool(torch.relu(h + self.res(h)))


class StatsTDNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(40, 48, (1, 5)),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, (1, 3), dilation=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.embed = nn.Linear(128, 48)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2) / 12)
        mean = h.mean((2, 3), keepdim=True)
        std = torch.sqrt(((h - mean) ** 2).mean((2, 3)) + 1e-4)
        return self.embed(torch.cat((mean.flatten(1), std), 1))


class PoolLDA(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = linear(
            np.vstack((-model.coef_, model.coef_)),
            np.r_[-model.intercept_, model.intercept_],
        )

    def forward(self, x):
        return self.head(x.mean(-1).flatten(1))


def run(work, commands):
    torch.set_num_threads(3)
    torch.manual_seed(17)

    def hash_order(v):
        return hashlib.sha256(v.encode()).digest()

    def features(wave):
        return log_mel(wave)[None]

    archive = zipfile.ZipFile(commands)
    by = defaultdict(list)
    for n in archive.namelist():
        if n.endswith(".wav") and "_nohash_" in n and "__MACOSX" not in n:
            by[Path(n).name.split("_nohash_")[0]].append(n)

    def wave(n):
        sr, x = wavfile.read(io.BytesIO(archive.read(n)))
        assert sr == 16000
        pcm = np.zeros(16000, np.float32)
        pcm[: min(len(x), 16000)] = x[:16000].astype(np.float32) / 32768
        return pcm

    test_spk = sorted(
        (s for s in by if len({Path(n).parent.name for n in by[s]}) >= 4),
        key=hash_order,
    )[:24]
    train_spk = sorted(
        (s for s in by if s not in test_spk and len(by[s]) >= 5),
        key=lambda s: (-len(by[s]), hash_order(s)),
    )[:120]

    def fixture(ds, rows, pcm, **labels):
        target = work / "fixtures" / ds
        target.mkdir(exist_ok=True)
        p = target / f"{len(rows)}.npy"
        np.save(p, pcm)
        rows.append(dict(file=ds + "/" + p.name, **labels))

    sx = []
    sy = []
    rows = []
    for label, s in enumerate(train_spk):
        for n in by[s]:
            sx.append(features(wave(n)))
            sy.append(label)
    for s in test_spk:
        chosen = {}
        for n in sorted(by[s], key=hash_order):
            chosen.setdefault(Path(n).parent.name, n)
        for j, n in enumerate(list(chosen.values())[:4]):
            fixture(
                "speaker",
                rows,
                wave(n),
                subject=s,
                role="gallery" if j == 0 else "query",
                source=n,
            )
    records(work, "speaker", rows)

    def fit(key, encoder, ncls, x, y, epochs, meta, embedding=False):
        x = torch.tensor(np.asarray(x))
        y = torch.tensor(y, dtype=torch.long)
        net = nn.Sequential(encoder, nn.Linear(48, ncls))
        opt = torch.optim.Adam(net.parameters(), lr=0.002)
        for ep in range(epochs):
            for ix in torch.randperm(len(x)).split(40):
                loss = nn.functional.cross_entropy(net(x[ix]), y[ix])
                opt.zero_grad()
                loss.backward()
                opt.step()
        save(
            work,
            key,
            encoder if embedding else net,
            x[:1].numpy(),
            dict(**meta, training=len(x), epochs=epochs),
        )

    meta = dict(
        train_speakers=len(train_spk),
        test_speakers=len(test_spk),
        test=len(rows),
        seed=17,
    )
    fit("speaker-tdnn", StatsTDNN(), len(train_spk), sx, sy, 18, meta, True)
    fit("speaker-rescnn", ResidualSpeaker(), len(train_spk), sx, sy, 18, meta, True)
    sx = []
    sy = []
    rows = []
    speech_train = [p for s in train_spk for p in by[s]]
    for n in sorted(speech_train, key=hash_order)[:384]:
        sx.append(features(wave(n)))
        sy.append(1)
    for n in sorted([n for s in test_spk for n in by[s]], key=hash_order)[:64]:
        fixture("presence", rows, wave(n), label=1, source=n)
    for r in json.loads((work / "data/esc-sources.json").read_text()):
        sr, x = wavfile.read(work / "data/esc" / r["filename"])
        pcm = resample_poly(x.astype(np.float32) / 32768, 16000, np.int64(sr)).astype(
            np.float32
        )
        for i in range(4) if int(r["fold"]) < 5 else range(2):
            clip = pcm[16000 * i : 16000 * (i + 1)]
            if int(r["fold"]) < 5:
                sx.append(features(clip))
                sy.append(0)
            else:
                fixture("presence", rows, clip, label=0, source=r["filename"], second=i)
    records(work, "presence", rows)
    sx = np.asarray(sx)
    meta = dict(test=len(rows), seed=17)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        sx.mean(-1)[:, 0], sy
    )
    save(work, "presence-lda", PoolLDA(lda), sx[:1], dict(training=len(sx), **meta))
    fit("presence-cnn", ResidualSpeaker(), 2, sx, sy, 10, meta)
    archive.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("commands", type=Path)
    args = p.parse_args()
    run(args.work, args.commands)
