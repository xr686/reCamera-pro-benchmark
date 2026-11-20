"""Package exactly the benchmark inputs and models pinned by the two device manifests."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def pack(work, classification, target, extra=False, extended=False, field=False, sequence=False, architectures=False):
    folder = "multitask-extended" if extended else ("multitask-extra" if extra else "multitask")
    if field:
        folder = "multitask-field"
    if sequence:
        folder = "multitask-sequence"
    if architectures:
        folder = "architectures"
    suite = json.loads((work / "suite.json").read_text())
    files = [(work, "suite.json", folder + "/suite.json", None)]
    if (field or sequence) and (work / "source-notices").is_dir():
        for note in sorted((work / "source-notices").glob("*.txt")):
            files.append((note.parent,note.name,folder + "/source-notices/" + note.name,None))
    if not (extra or extended or field or sequence or architectures):
        index = json.loads((classification / "index.json").read_text())
        files.append((classification, "index.json", "fixtures/index.json", None))
        for row in index["samples"]:
            name = row["file"]
            files.append((classification, name, "fixtures/" + name, row["sha256"]))
    for name, digest in suite["files"].items():
        files.append((work / "fixtures", name, folder + "/fixtures/" + name, digest))
    for model, metadata in suite["models"].items():
        name = model + ".rknn"
        files.append(
            (work / "compiled", name, folder + "/compiled/" + name, metadata["sha256"])
        )
    temp = target.with_suffix(".partial")
    try:
        with tarfile.open(temp, "w:gz", compresslevel=3) as archive:
            for root, name, arcname, digest in files:
                path = (root / name).resolve()
                if (
                    not path.is_relative_to(root.resolve())
                    or ".." in Path(name).parts
                    or Path(name).is_absolute()
                ):
                    raise ValueError(f"Invalid resource path {name}")
                if digest:
                    with path.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                            raise ValueError(f"SHA-256 mismatch for {name}")
                archive.add(path, arcname, recursive=False)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("work", "classification", "target"):
        parser.add_argument(key, type=Path)
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Build the additional resource pack; classification is unused",
    )
    parser.add_argument("--extended", action="store_true", help="Build the third resource pack")
    parser.add_argument("--field", action="store_true", help="Build the fourth resource pack")
    parser.add_argument("--sequence", action="store_true", help="Build the fifth resource pack")
    parser.add_argument("--architectures", action="store_true", help="Pack one representative per new backbone family")
    pack(**vars(parser.parse_args()))
