"""Export one representative per backbone family, compile and reference it independently."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import os
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from architecture_models import FAMILIES


def compile_one(work, key):
    env = {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    }
    script = Path(__file__).resolve()
    with (work / "logs" / (key + ".log")).open("w") as log:
        source = work / "sources" / (key + ".onnx")
        if key in ("swiftformer", "volo"):
            source = work / "data" / (key + "-rv.onnx")
        flags = (
            ["--disable-rule", "convert_rs_add_rs_to_rs_gather_elements"]
            if key == "maxvit"
            else []
        )
        for arguments in (
            [str(script), str(work), "--export", key],
            [
                str(script.with_name("compile_onnx.py")),
                str(source),
                str(work / "compiled" / (key + ".rknn")),
                *flags,
            ],
        ):
            status = subprocess.run(
                ["python", *arguments],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=work,
            )
            if status.returncode:
                print(key, "failed", flush=True)
                return False
    print(key, "compiled", flush=True)
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("models", nargs="*", help="Family IDs (default: all)")
    p.add_argument("--export", choices=FAMILIES)
    args = p.parse_args()
    work = args.work.resolve()
    if args.export:
        from timm.layers import set_fused_attn
        from export_wave3_backbone import export

        set_fused_attn(False)
        key = args.export
        _, name, side, _ = FAMILIES[key]
        if key == "volo":
            from unittest.mock import patch
            from torch.nn import functional as F

            original_fold = F.fold

            def static_fold(x, output_size, *args, **kwargs):
                return original_fold(x, tuple(map(int, output_size)), *args, **kwargs)

            with patch.object(F, "fold", static_fold):
                export(work, key, name, opset=18)
        else:
            export(work, key, name)
        if key == "swiftformer":
            from lower_speaker_pool import lower

            lower(work / "sources/swiftformer.onnx", work / "data/swiftformer-rv.onnx")
        if key == "volo":
            import timm
            from lower_outlook import lower

            lower(
                timm.create_model(name, pretrained=True).eval(),
                work / "data/volo-rv.onnx",
            )
        assert (
            json.loads((work / "sources" / (key + ".json")).read_text())["size"] == side
        )
    else:
        keys = args.models or list(FAMILIES)
        if set(keys) - FAMILIES.keys():
            p.error("Unknown family ID")
        with ThreadPoolExecutor(4) as pool:
            statuses = list(pool.map(lambda key: compile_one(work, key), keys))
        raise SystemExit(0 if all(statuses) else 1)
