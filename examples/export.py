"""Fetch source weights or run the original export/training recipe for a single demo."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen

from build import HERE, TOOLS, index


def fetch(asset, dest):
    def digest(path):
        with path.open("rb") as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
            return h.hexdigest()

    if dest.is_file() and digest(dest) == asset["sha256"]:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        with urlopen(asset["url"], timeout=120) as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out)
        if digest(tmp) != asset["sha256"]:
            raise ValueError(f"Upstream artifact checksum mismatch for {dest.name}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=index("measurements.csv"), metavar="MODEL_ID")
    parser.add_argument("work", type=Path)
    for name in ("imagenette", "voc", "pet-archive", "commands", "esc-index", "previous-fixtures"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    spec = json.loads((HERE / args.model / "model.json").read_text())
    recipe = spec.get("export")
    if not recipe:
        parser.error(f"Use the precompiled weights or compatible graph listed in examples/{args.model}/README.md")
    paths = {name: str(v.resolve()) for name, v in vars(args).items() if isinstance(v, Path)}
    missing = [flag for flag in recipe.get("requires", []) if flag not in paths]
    if missing:
        parser.error("Required data: " + ", ".join("--" + k.replace("_","-") for k in missing))
    work = args.work.resolve()
    for sub in ("data","sources","compiled","fixtures","logs","ready"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    for name, asset in recipe.get("assets", {}).items():
        fetch(asset, work / name)
    env = dict(os.environ)
    module_paths = [str(TOOLS.parent), str(TOOLS)]
    if env.get("RECAMERA_SDK_ROOT"):
        module_paths.append(str(Path(env["RECAMERA_SDK_ROOT"]).resolve()))
    if env.get("PYTHONPATH"):
        module_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(module_paths)
    for name, *params in recipe.get("steps", []):
        subprocess.run([sys.executable, (name if name == "-m" else str(TOOLS / name)), *(value.format(**paths) for value in params)],
                       check=True, cwd=work, env=env)
    output = work / "ready" / (args.model + ".onnx")
    shutil.copyfile(work / recipe["graph"], output)
    print(f"Conversion-ready ONNX: {output}")


if __name__ == "__main__":
    main()
