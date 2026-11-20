"""Fetch the public model sources and datasets used in the additional suite; verify SHA-256."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import shutil


def fetch(work):
    manifest = json.loads(Path(__file__).with_name("extra-sources.json").read_text())
    for name, record in manifest.items():
        dest = work / name
        if dest.is_file():
            with dest.open("rb") as stream:
                if (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    == record["sha256"]
                ):
                    continue
            raise ValueError(f"An existing input has a different SHA-256: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        try:
            with urlopen(record["url"], timeout=120) as source, tmp.open("wb") as out:
                shutil.copyfileobj(source, out)
            with tmp.open("rb") as stream:
                if (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    != record["sha256"]
                ):
                    raise ValueError(name)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    fetch(p.parse_args().work)
