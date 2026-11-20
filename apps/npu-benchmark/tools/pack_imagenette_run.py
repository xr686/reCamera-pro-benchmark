"""Build a device resource pack from measured RKNNs and the matching Imagenette validation files."""

import argparse
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from pack_resources import pack


def build(manifests, models, validation, output, ids):
    suite = json.loads((manifests / "suite.json").read_text())
    selected = ids or [k for k in suite["models"] if (models / (k + ".rknn")).is_file()]
    if not selected or set(selected) - suite["models"].keys():
        raise ValueError("Choose at least one measured architecture ID from suite.json")
    suite["models"] = {key: suite["models"][key] for key in selected}
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "compiled").mkdir()
        for key in selected:
            # pack_resources checks that each input resolves inside its staging directory.
            shutil.copyfile(models / (key + ".rknn"), work / "compiled" / (key + ".rknn"))
        for name in suite["files"]:
            dest = work / "fixtures" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = manifests / name if name == "groups.json" else validation / Path(name).relative_to("imagenette-extra")
            shutil.copyfile(source, dest)
        (work / "suite.json").write_text(json.dumps(suite, ensure_ascii=False))
        pack(work, work, output, architectures=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", type=Path, help="benchmark_results/app/imagenette-100")
    parser.add_argument("models", type=Path, help="directory containing measured .rknn files")
    parser.add_argument("validation", type=Path, help="extracted imagenette2-160/val")
    parser.add_argument("output", type=Path)
    parser.add_argument("ids", nargs="*", help="optional model IDs to include")
    args = parser.parse_args()
    build(args.manifests, args.models, args.validation, args.output, args.ids)
