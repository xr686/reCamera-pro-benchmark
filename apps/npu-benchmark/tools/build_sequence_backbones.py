"""Export the sequence pack's trained ImageNet classifiers and build their RKNN graphs."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import os
import subprocess
import sys

from build_field_models import build
from export_wave3_backbone import export
from sequence_models import BACKBONES

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("models", nargs="*")
    p.add_argument("--export", choices=BACKBONES)
    args = p.parse_args()
    work = args.work.resolve()
    if args.export:
        export(work, args.export, BACKBONES[args.export][1])
    else:

        def one(key):
            with (work / "logs" / (key + "-export.log")).open("w") as log:
                subprocess.run(
                    ["python", __file__, work, "--export", key],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env={
                        **os.environ,
                        "PATH": str(Path(sys.executable).parent)
                        + os.pathsep
                        + os.environ["PATH"],
                    },
                )
            build(work, [key])

        with ThreadPoolExecutor(3) as pool:
            list(pool.map(one, args.models or BACKBONES))
