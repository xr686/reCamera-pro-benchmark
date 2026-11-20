"""Assemble a complete pack (or an initial subset) and compute same-input ONNX references."""

import argparse
import json
import shutil
from pathlib import Path

from prepare_wave3 import reference_suite
from task_models import Model
from sequence_models import contracts


def prepare(work, previous, subset):
    root = work
    models = contracts(Model)
    if subset:
        work = root / "initial"
        work.mkdir(exist_ok=True)
        for folder in ("sources", "compiled", "data"):
            if not (work / folder).exists():
                (work / folder).symlink_to(
                    (root / folder).resolve(), target_is_directory=True
                )
        models = [m for m in models if m.corpus in subset]
    fixtures = work / "fixtures"
    fixtures.mkdir(exist_ok=True)
    groups = {}
    for corpus in sorted({m.corpus for m in models}):
        if corpus == "imagenette-extra":
            groups[corpus] = json.loads((previous / "groups.json").read_text())[corpus]
            for record in groups[corpus]:
                dest = fixtures / record["file"]
                dest.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(previous / record["file"], dest)
        else:
            groups[corpus] = json.loads(
                (root / "data" / (corpus + "-records.json")).read_text()
            )
            if subset:
                shutil.copytree(
                    root / "fixtures" / corpus, fixtures / corpus, dirs_exist_ok=True
                )
    for m in models:
        assert (work / "compiled" / (m.file + ".rknn")).is_file(), m.file
    reference_suite(work, groups, models)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("previous", type=Path)
    p.add_argument("--subset", nargs="+")
    a = p.parse_args()
    prepare(a.work, a.previous, a.subset)
