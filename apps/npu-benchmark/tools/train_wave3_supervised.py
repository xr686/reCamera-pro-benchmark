"""Fit supervised and analytic baselines using held-out text, regression and Fashion-MNIST samples."""

import argparse, gzip, io, json, sys, zipfile, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extended_tasks import text_vector
from extended_models import REGRESSION, REGRESSORS, APPAREL


def save(work, key, model, example, metadata):
    model.eval()
    torch.onnx.export(
        model,
        torch.from_numpy(example.astype(np.float32)),
        work / "sources" / (key + ".onnx"),
        input_names=["input"],
        opset_version=13,
    )
    (work / "sources" / (key + ".json")).write_text(
        json.dumps(metadata, ensure_ascii=False)
    )
    print(key, flush=True)


def linear(coef, bias):
    w = torch.tensor(np.atleast_2d(coef), dtype=torch.float32)
    result = nn.Linear(w.shape[1], w.shape[0])
    with torch.no_grad():
        result.weight.copy_(w)
        result.bias.copy_(torch.tensor(np.atleast_1d(bias), dtype=torch.float32))
    return result


def records(work, ds, rows):
    (work / "data" / (ds + "-records.json")).write_text(
        json.dumps(rows, ensure_ascii=False)
    )


def regression_data(root, ds):
    with zipfile.ZipFile(root / (ds + ".zip")) as z:
        if ds == "energy":
            with zipfile.ZipFile(io.BytesIO(z.read("ENB2012_data.xlsx"))) as book:
                sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
                data = []
                for row in sheet.findall(".//{*}row")[1:]:
                    vals = [
                        c.findtext("{*}v")
                        for c in row.findall("{*}c")
                        if c.get("t") != "s"
                    ]
                    if len(vals) >= 10 and all(v is not None for v in vals[:10]):
                        data.append([float(v) for v in vals[:10]])
                arr = np.array(data)
                return arr[:, :8], arr[:, 8]
        if ds == "wine":
            arr = np.loadtxt(z.open("winequality-red.csv"), delimiter=";", skiprows=1)
        else:
            arr = np.loadtxt(z.open("airfoil_self_noise.dat"))
        return arr[:, :-1], arr[:, -1]


class RandomFeatures(nn.Module):
    def __init__(self, projection, weights, bias, kind):
        super().__init__()
        self.project = projection
        self.output = linear(weights, bias)
        self.kind = kind

    def forward(self, x):
        z = self.project(x)
        return self.output(torch.cos(z) if self.kind == "rff" else torch.tanh(z))


def tabular(work):
    rng = np.random.default_rng(74)
    for ds in REGRESSION:
        x, y = regression_data(work / "data", ds)
        # Keep duplicate input vectors in the same partition.
        _, first, inv = np.unique(x, axis=0, return_index=True, return_inverse=True)
        tr, te = train_test_split(
            np.arange(len(first)), test_size=0.25, random_state=39
        )
        train = np.flatnonzero(np.isin(inv, tr))
        test = first[te][:96]
        mu = x[train].mean(0)
        std = x[train].std(0).clip(1e-6)
        z = (x - mu) / std
        base = work / "fixtures" / ds
        base.mkdir(exist_ok=True)
        rows = []
        for n, i in enumerate(test):
            np.save(base / f"{n}.npy", x[i].astype(np.float32))
            rows.append(dict(file=f"{ds}/{n}.npy", value=float(y[i])))
        records(work, ds, rows)
        for alg in REGRESSORS:
            meta = dict(
                mean=mu.tolist(),
                std=std.tolist(),
                training=len(train),
                test=len(test),
                split="grouped unique predictor tuples, seed 39",
            )
            if alg == "pls":
                model = PLSRegression(n_components=min(4, x.shape[1]), scale=False).fit(
                    z[train], y[train]
                )
                origin = np.zeros((1, x.shape[1]))
                intercept = model.predict(origin).ravel()
                coef = (model.predict(np.eye(x.shape[1])).ravel() - intercept)[None]
                net = linear(coef, intercept)
            else:
                w = rng.normal(0, 0.55, (96, x.shape[1]))
                bias = rng.uniform(-np.pi, np.pi, 96)
                func = np.cos if alg == "rff" else np.tanh
                ridge = Ridge(alpha=0.12).fit(func(z[train] @ w.T + bias), y[train])
                net = RandomFeatures(
                    linear(w, bias), ridge.coef_, ridge.intercept_, alg
                )
            save(work, ds + "-" + alg, net, z[:1], meta)


def spam(work):
    with zipfile.ZipFile(work / "data/sms.zip") as z:
        lines = z.read("SMSSpamCollection").decode().splitlines()
    samples = {}
    for line in lines:
        if "\t" in line:
            label, msg = line.split("\t", 1)
            samples.setdefault(" ".join(msg.lower().split()), int(label == "spam"))
    msgs = list(samples)
    y = np.array(list(samples.values()))
    train, test = train_test_split(
        np.arange(len(y)), test_size=0.25, stratify=y, random_state=41
    )
    test = np.concatenate([test[y[test] == i][:64] for i in (0, 1)])
    x = np.stack([text_vector(t) for t in msgs])
    base = work / "fixtures/sms"
    base.mkdir(exist_ok=True)
    rows = []
    for j, i in enumerate(test):
        (base / f"{j}.txt").write_text(msgs[i])
        rows.append(dict(file=f"sms/{j}.txt", label=int(y[i])))
    records(work, "sms", rows)
    nb = MultinomialNB().fit(x[train], y[train])
    svm = LinearSVC(C=0.15, dual="auto", random_state=41).fit(x[train], y[train])
    mlp = MLPClassifier((40,), max_iter=70, random_state=41).fit(x[train], y[train])

    # sklearn binary estimators produce one margin; two logits preserve the sign decision.
    def logits(w, b):
        return linear(np.vstack([-w, w]), np.r_[-b, b])

    models = {
        "sms-nb": linear(nb.feature_log_prob_, nb.class_log_prior_),
        "sms-svm": logits(svm.coef_, svm.intercept_),
        "sms-mlp": nn.Sequential(
            linear(mlp.coefs_[0].T, mlp.intercepts_[0]),
            nn.ReLU(),
            logits(mlp.coefs_[1].T, mlp.intercepts_[1]),
        ),
    }
    for key, m in models.items():
        save(
            work,
            key,
            m,
            x[:1],
            dict(training=len(train), test=len(test), features=512, seed=41),
        )


class Highway(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(784, 96)
        self.hidden = nn.Linear(96, 96)
        self.gate = nn.Linear(96, 96)
        self.head = nn.Linear(96, 10)

    def forward(self, x):
        h = self.embed(x.flatten(1)).relu()
        g = self.gate(h).sigmoid()
        return self.head(h * (1 - g) + g * self.hidden(h).relu())


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(1, 32, 4, 4)
        self.tokens = nn.Sequential(nn.Linear(49, 64), nn.ReLU(), nn.Linear(64, 49))
        self.channels = nn.Sequential(nn.Linear(32, 48), nn.ReLU(), nn.Linear(48, 32))
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        h = self.patch(x).flatten(2)
        h = (h + self.tokens(h)).transpose(1, 2)
        h = h + self.channels(h)
        return self.head(h.mean(1))


def fashion(work):
    def arrays(prefix):
        x = np.frombuffer(
            gzip.decompress(
                (work / "data" / (prefix + "-images-idx3-ubyte.gz")).read_bytes()
            ),
            np.uint8,
            offset=16,
        ).reshape(-1, 28, 28)
        y = np.frombuffer(
            gzip.decompress(
                (work / "data" / (prefix + "-labels-idx1-ubyte.gz")).read_bytes()
            ),
            np.uint8,
            offset=8,
        )
        return x, y

    x, y = arrays("train")
    tx, ty = arrays("t10k")
    tr = np.concatenate([np.where(y == i)[0][:1000] for i in range(10)])
    test = np.concatenate([np.where(ty == i)[0][:12] for i in range(10)])
    train_x = torch.from_numpy(x[tr].copy()).float()[:, None] / 255
    train_y = torch.from_numpy(y[tr].copy()).long()
    torch.manual_seed(41)
    base = work / "fixtures/fashion"
    base.mkdir(exist_ok=True)
    rows = []
    for idx in test:
        cv2.imwrite(str(base / f"{idx}.png"), tx[idx])
        rows.append(dict(file=f"fashion/{idx}.png", label=int(ty[idx])))
    records(work, "fashion", rows)
    networks = [
        nn.Sequential(
            nn.Conv2d(1, 24, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 24, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 10, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        ),
        Highway(),
        Mixer(),
    ]
    for key, net in zip(APPAREL, networks):
        optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
        for epoch in range(8):
            for ids in torch.randperm(len(tr)).split(96):
                loss = nn.functional.cross_entropy(net(train_x[ids]), train_y[ids])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        save(
            work,
            key,
            net,
            train_x[:1].numpy(),
            dict(training=len(tr), test=len(test), epochs=8, seed=41),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("group", choices=["tabular", "spam", "fashion"])
    a = p.parse_args()
    torch.set_num_threads(3)
    {"tabular": tabular, "spam": spam, "fashion": fashion}[a.group](a.work)
