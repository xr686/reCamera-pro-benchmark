"""Fit distinct supervised classifiers on held-out UCI measurements and labelled text."""

import argparse, csv, io, sys, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extended_tasks import text_vector
from train_wave3_supervised import linear, save, records


def table(work, ds):
    archive = {"gridstab": "grid"}.get(ds, ds)
    with zipfile.ZipFile(work / "data" / (archive + ".zip")) as z:
        raw = z.read(z.namelist()[0])
    if ds == "raisins":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            raw = z.read("Raisin_Dataset/Raisin_Dataset.arff")
        raw = raw.decode().lower().split("@data")[1]
        rows = [
            l.strip().split(",")
            for l in raw.splitlines()
            if l.strip() and not l.startswith("%")
        ]
    elif ds == "gridstab":
        rows = [
            v[:12] + [v[-1]] for v in list(csv.reader(io.StringIO(raw.decode())))[1:]
        ]
    else:
        rows = [v.split() for v in raw.decode().splitlines() if v.strip()]
    x = np.asarray([r[:-1] for r in rows], np.float32)
    labels, y = np.unique([r[-1] for r in rows], return_inverse=True)
    _, ix = np.unique(x, axis=0, return_index=True)
    x = x[ix]
    y = y[ix]
    tr, te = train_test_split(
        np.arange(len(y)), test_size=0.28, stratify=y, random_state=86
    )
    te = np.concatenate([te[y[te] == i][:24] for i in range(len(labels))])
    return x, y, tr, te, labels.tolist()


class Quadratic(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        mu = np.stack([x[y == i].mean(0) for i in np.unique(y)])
        var = np.stack([x[y == i].var(0) + 0.08 for i in np.unique(y)])
        self.quad = linear(
            -0.5 / var,
            np.log(np.bincount(y) / len(y))
            - 0.5 * (mu * mu / var + np.log(var)).sum(1),
        )
        self.lin = linear(mu / var, np.zeros(len(mu)))

    def forward(self, x):
        return self.quad(x * x) + self.lin(x)


class Proto(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        centers = np.concatenate(
            [
                KMeans(n_clusters=4, n_init=5, random_state=86)
                .fit(x[y == i])
                .cluster_centers_
                for i in np.unique(y)
            ]
        )
        self.score = linear(2 * centers, -(centers**2).sum(1))
        self.n = len(np.unique(y))

    def forward(self, x):
        return self.score(x).reshape(1, self.n, 4).max(-1).values


def vectors(work, ds):
    x, y, tr, te, classes = table(work, ds)
    mu = x[tr].mean(0)
    sd = x[tr].std(0).clip(0.0001)
    z = (x - mu) / sd
    rng = np.random.default_rng(86)
    w = rng.normal(0, 1 / np.sqrt(x.shape[1]), (96, x.shape[1]))
    b = rng.normal(0, 0.4, 96)
    ridge = Ridge(alpha=0.15).fit(
        np.maximum(z[tr] @ w.T + b, 0), np.eye(len(classes))[y[tr]]
    )
    nets = {
        "qda": Quadratic(z[tr], y[tr]),
        "lvq": Proto(z[tr], y[tr]),
        "randomnet": nn.Sequential(
            linear(w, b), nn.ReLU(), linear(ridge.coef_, ridge.intercept_)
        ),
    }
    fx = work / "fixtures" / ds
    fx.mkdir(exist_ok=True)
    rows = []
    for i in te:
        np.save(fx / f"{i}.npy", x[i])
        rows.append(dict(file=f"{ds}/{i}.npy", label=int(y[i])))
    meta = dict(
        mean=mu.tolist(),
        std=sd.tolist(),
        training=len(tr),
        classes=classes,
        test=len(te),
        seed=86,
    )
    for k, m in nets.items():
        save(work, ds + "-" + k, m, z[:1], meta)
    records(work, ds, rows)


class BinaryTerms(nn.Module):
    def __init__(self, coef, bias):
        super().__init__()
        self.fc = linear(coef, bias)

    def forward(self, x):
        return self.fc(x.clamp(0, 1))


def texts(work, ds):
    if ds == "trec":

        def load(split):
            lines = (
                (work / "data" / ("trec-" + split + ".txt"))
                .read_text(encoding="latin1")
                .splitlines()
            )
            return [
                (l.split(" ", 1)[1], l.split(":", 1)[0]) for l in lines if l.strip()
            ]

        training = load("train")
        known = {s.casefold().strip() for s, _ in training}
        testing = [r for r in load("test") if r[0].casefold().strip() not in known]
    else:
        pairs = {}
        with zipfile.ZipFile(work / "data/sentiment.zip") as z:
            for p in z.namelist():
                if p.endswith("_labelled.txt") and not p.startswith("__"):
                    for l in z.read(p).decode().splitlines():
                        if "\t" in l:
                            t, y = l.rsplit("\t", 1)
                            pairs.setdefault(t.lower().strip(), y.strip())
        training, testing = train_test_split(
            list(pairs.items()),
            stratify=list(pairs.values()),
            random_state=86,
            test_size=0.25,
        )
    labels = sorted({y for _, y in training})
    testing = [r for y in labels for r in [p for p in testing if p[1] == y][:30]]
    x = np.stack([text_vector(t) for t, _ in training])
    y = np.array([labels.index(l) for _, l in training])
    truth = np.eye(len(labels))[y]
    clf = BernoulliNB(alpha=0.8).fit(x, y)
    logp = clf.feature_log_prob_
    neg = np.log1p(-np.exp(logp))
    ridge = Ridge(alpha=5).fit(x, truth)
    clf2 = SGDClassifier(
        loss="perceptron", average=True, max_iter=120, random_state=86
    ).fit(x, y)
    weights, bias = clf2.coef_, clf2.intercept_
    if len(labels) == 2:
        weights = np.vstack((-weights, weights))
        bias = np.r_[-bias, bias]
    models = {
        "bnb": BinaryTerms(logp - neg, clf.class_log_prior_ + neg.sum(1)),
        "ridge": linear(ridge.coef_, ridge.intercept_),
        "perceptron": linear(weights, bias),
    }
    fx = work / "fixtures" / ds
    fx.mkdir(exist_ok=True)
    rows = []
    for i, (t, y) in enumerate(testing):
        (fx / f"{i}.txt").write_text(t)
        rows.append(dict(file=f"{ds}/{i}.txt", label=labels.index(y)))
    records(work, ds, rows)
    for k, m in models.items():
        save(
            work,
            ds + "-" + k,
            m,
            x[:1],
            dict(training=len(training), classes=labels, test=len(testing), seed=86),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("--domain", choices=("wifi", "gridstab", "seeds", "raisins", "sent", "trec"))
    args = p.parse_args()
    torch.set_num_threads(2)
    domains = (args.domain,) if args.domain else ("wifi", "gridstab", "seeds", "raisins", "sent", "trec")
    for domain in domains:
        trainer = texts if domain in ("sent", "trec") else vectors
        trainer(args.work, domain)
