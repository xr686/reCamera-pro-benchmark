"""Select fixed, labelled train/test images from the Voxel51 mirror of MVTec AD."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib, json, subprocess
from pathlib import Path
import requests


def fetch(work):
    repo = "Voxel51/mvtec-ad"
    info = requests.get("https://huggingface.co/api/datasets/" + repo, timeout=30)
    info.raise_for_status()
    base = f"https://huggingface.co/datasets/{repo}/resolve/" + info.json()["sha"] + "/"
    request = requests.get(base + "samples.json", timeout=60)
    request.raise_for_status()
    groups = defaultdict(list)
    for row in request.json()["samples"]:
        cat = row["category"]["label"]
        defect = row["defect"]["label"]
        split = row["split"]
        if cat in ("bottle", "grid"):
            groups[cat, split, defect].append(row["filepath"])
    todo = []
    for (cat, split, defect), paths in sorted(groups.items()):
        count = 72 if split == "train" else (20 if defect == "good" else 8)
        for path in sorted(paths, key=lambda v: hashlib.sha256(v.encode()).digest())[
            :count
        ]:
            todo.append((path, cat, split, defect))
    assert {t[2] for t in todo} == {"train", "test"}

    def one(t):
        name, cat, split, defect = t
        path = work / "data" / "mvtec" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = path.with_suffix(".part")
            subprocess.run(
                [
                    "curl",
                    "-fsSL",
                    "--retry",
                    "3",
                    "--max-time",
                    "70",
                    base + name,
                    "-o",
                    str(tmp),
                ],
                check=True,
            )
            tmp.replace(path)
        return dict(
            path=name,
            category=cat,
            split=split,
            defect=defect,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    with ThreadPoolExecutor(12) as pool:
        images = list(pool.map(one, todo))
    (work / "data/mvtec-sources.json").write_text(
        json.dumps(dict(base=base, images=images), indent=2)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    fetch(p.parse_args().work)
