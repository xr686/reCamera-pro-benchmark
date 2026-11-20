"""Cross-check the public CSV against board records. Uses only the Python standard library."""

import csv
import json
import math
from pathlib import Path
from statistics import mean


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def check(value, message):
    if not value:
        raise ValueError(message)


def verify(root):
    measured = read_csv(root / "measurements.csv")
    ids = {r["id"]: r for r in measured}
    check(len(ids) == len(measured), "Duplicate model IDs")
    seen = set()
    samples = 0
    runs = sorted((root / "runs").glob("*/*.json"))
    for path in runs:
        run = json.loads(path.read_text())
        check(len(run["idle"]) >= 3 and all(s.split()[-1] == "0%" for s in run["idle"][-3:]), path)
        for model in run["models"]:
            key = model["id"]
            row = ids[key]
            check(key not in seen, key)
            seen.add(key)
            check((row["series"], row["run"]) == (path.parent.name, path.name), key)
            check(row["sha256"] == model["sha256"] and len(model["sha256"]) == 64, key)
            check(model["quality"] == json.loads(row["quality"]), key)
            check(model["checks"] and all(c["owners"] in (["python"], ["python", "rkipc"]) for c in model["checks"]), key)
            times = model["times_ms"]
            check(len(times) == int(row["calls"]) and all(len(t) == 3 for t in times), key)
            samples += len(times)
            pre, infer, post = (mean(column) for column in zip(*times))
            values = sorted(t[1] for t in times)
            pos = (len(values) - 1) * 0.95
            lo, hi = math.floor(pos), math.ceil(pos)
            p95 = values[lo] + (pos - lo) * (values[hi] - values[lo])
            metrics = dict(pre_ms=pre, infer_ms=infer, post_ms=post, p95_ms=p95, fps=1000 / (pre + infer + post))
            if row["with_gc"]:
                check(len(model["cleanup_ms"]) == len(times), key)
                metrics["with_gc"] = 1000 / (pre + infer + post + mean(model["cleanup_ms"]))
            for field, value in metrics.items():
                check(math.isclose(float(row[field]), value, rel_tol=1e-8, abs_tol=1e-8), f"{key} {field}")
    check(seen == ids.keys(), "Missing board measurements")
    families = read_csv(root / "RV1126B-ImageNet-model-index.csv")
    members = [key for row in families for key in [row["id"], *filter(None, row["merged_ids"].split(","))]]
    checkpoints = read_csv(root / "classification-sources.csv")
    check(len(members) == len(set(members)) and set(members) <= ids.keys(), "Overlapping or unknown family members")
    check(set(members) == {r["id"] for r in checkpoints}, "Missing checkpoint sources")
    hashes = read_csv(root / "model-hashes.csv")
    check({r["id"]: r["rknn_sha256"] for r in hashes} == {k: r["sha256"] for k, r in ids.items()}, "Artifact manifest mismatch")
    print(f"{len(ids)} configurations, {len(families)} representative classification models, {len(runs)} runs, {samples} timed calls")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parent)
