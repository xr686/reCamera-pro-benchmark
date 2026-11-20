"""Export fourth-pack timm models and compile selected static ONNX graphs to RV1126B."""

import argparse, concurrent.futures, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from field_models import BACKBONES


def build(work, ids):
    base = Path(__file__).resolve().parent

    def one(key):
        with (work / "logs" / (key + ".log")).open("w") as log:
            env = dict(
                os.environ,
                PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            )
            if key in BACKBONES:
                subprocess.run(
                    ["python", Path(__file__).resolve(), str(work), "--export", key],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    cwd=work,
                    env=env,
                )
            subprocess.run(
                [
                    "python",
                    base / "compile_onnx.py",
                    work / "sources" / (key + ".onnx"),
                    work / "compiled" / (key + ".rknn"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=work,
                env=env,
            )
            print(key, flush=True)

    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        jobs = {pool.submit(one, k): k for k in ids}
        failed = []
        for f in concurrent.futures.as_completed(jobs):
            try:
                f.result()
            except Exception as e:
                failed.append(jobs[f])
                print(f"Failed {jobs[f]}: {e}", flush=True)
        if failed:
            raise RuntimeError(f"Compilation failed: {failed}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("models", nargs="*")
    p.add_argument("--export", choices=BACKBONES)
    args = p.parse_args()
    work = args.work.resolve()
    if args.export:
        from export_wave3_backbone import export

        export(work, args.export, BACKBONES[args.export])
    else:
        build(work, args.models or list(BACKBONES))
