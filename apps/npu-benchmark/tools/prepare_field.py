"""Assemble field fixtures and compute original ONNX measurements on the matching inputs."""

import argparse, json, shutil, sys, tarfile, zipfile
from pathlib import Path
from prepare_wave3 import reference_suite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_models import Model
from field_models import contracts, TABLES, LANG


def prepare(work, previous, first=False):
    source = work
    catalog = contracts(Model)
    if first:
        work = work / "first"
        work.mkdir(exist_ok=True)
        for d in ("sources", "compiled", "data"):
            if not (work / d).exists():
                (work / d).symlink_to((source / d).resolve(), target_is_directory=True)
        catalog = [
            m for m in catalog if m.corpus in {*TABLES, *LANG, "imagenette-extra"}
        ]
    root = work / "fixtures"
    root.mkdir(exist_ok=True)
    g0 = json.loads((previous / "groups.json").read_text())
    groups = {"imagenette-extra": g0["imagenette-extra"]}
    for r in groups["imagenette-extra"]:
        path = root / r["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous / r["file"], path)
    for ds in sorted({m.corpus for m in catalog} - {"imagenette-extra"}):
        groups[ds] = json.loads((source / "data" / (ds + "-records.json")).read_text())
        if first:
            shutil.copytree(source / "fixtures" / ds, root / ds, dirs_exist_ok=True)
    for m in catalog:
        if not (work / "compiled" / (m.file + ".rknn")).is_file():
            raise ValueError(m.file)
    if not first:
        notices = work / "source-notices"
        notices.mkdir(exist_ok=True)
        with tarfile.open(work / "data/flowers.tgz") as z:
            (notices / "flowers-LICENSE.txt").write_bytes(
                z.extractfile("flower_photos/LICENSE.txt").read()
            )
        for archive, member, name in [
            ("gts-train.zip", "GTSRB/Training/Readme.txt", "gtsrb-Readme.txt"),
            (
                "sentiment.zip",
                "sentiment labelled sentences/readme.txt",
                "sentiment-readme.txt",
            ),
        ]:
            with zipfile.ZipFile(work / "data" / archive) as z:
                (notices / name).write_bytes(z.read(member))
    reference_suite(work, groups, catalog)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("previous", type=Path)
    p.add_argument("--first", action="store_true")
    args = p.parse_args()
    prepare(args.work, args.previous, args.first)
