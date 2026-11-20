"""Fetch the fixed public UCR/UEA archives and retain their SHA-256 provenance."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sequence_models import SERIES


def fetch(work, ds):
    target = work / "data" / (ds + ".zip")
    url = f"https://www.timeseriesclassification.com/aeon-toolkit/{ds}.zip"
    if not target.exists():
        partial = target.with_suffix(".part")
        subprocess.run(
            ["curl", "-fsSL", "--retry", "3", "--max-time", "120", "-o", partial, url],
            check=True,
        )
        with zipfile.ZipFile(partial) as archive:
            assert archive.testzip() is None
        partial.replace(target)
    with target.open("rb") as stream:
        return ds, dict(url=url, sha256=hashlib.sha256(stream.read()).hexdigest())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    work = p.parse_args().work
    with ThreadPoolExecutor(4) as pool:
        sources = dict(pool.map(lambda ds: fetch(work, ds), SERIES))
    (work / "source-assets.json").write_text(json.dumps(sources, indent=2))
