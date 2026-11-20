"""Select a reproducible, balanced subset of the Imagenette2-160 val directory."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

LABELS = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
          "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
          "n03445777": 574, "n03888257": 701}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("val", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-class", type=int, default=30)
    args = parser.parse_args()
    groups = []
    for synset, label in LABELS.items():
        def path_key(path):
            return hashlib.sha256(path.relative_to(args.val).as_posix().encode()).digest()

        images = sorted((args.val / synset).glob("*.JPEG"), key=path_key)
        if not 1 <= args.per_class <= len(images):
            parser.error(f"{synset} has {len(images)} images, requested {args.per_class}")
        samples = []
        for image in images[:args.per_class]:
            name = image.relative_to(args.val).as_posix()
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, target)
            samples.append(dict(file=name, label=label, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        groups.append(samples)
    rows = [sample for group in zip(*groups) for sample in group]
    manifest = {"dataset": "Imagenette2-160 val", "samples": rows}
    (args.output / "index.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(rows)} labeled images.")
