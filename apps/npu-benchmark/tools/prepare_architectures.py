"""Prepare shared Imagenette fixtures and run original ONNX references for the selected architecture families."""

import argparse
import json
from pathlib import Path
import shutil

from prepare_wave3 import reference_suite
from architecture_models import contracts
from task_models import Model


def prepare(work, previous, ids):
    models = [m for m in contracts(Model) if m.file in ids]
    if len(models) != len(ids):
        raise ValueError("Unknown or duplicate architecture ID")
    for m in models:
        if not (work / "compiled" / (m.file + ".rknn")).is_file():
            raise FileNotFoundError(m.file)
    group = "imagenette-extra"
    rows = json.loads((previous / "groups.json").read_text())[group]
    assert len(rows) == len({r["file"] for r in rows}) == 100
    for row in rows:
        name = row["file"]
        dest = work / "fixtures" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous / name, dest)
    reference_suite(work, {group: rows}, models)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("previous", type=Path)
    p.add_argument("ids", nargs="+")
    a = p.parse_args()
    prepare(a.work, a.previous, a.ids)
