"""Compile the exported ONNX models and refresh bundled artifact identities."""
import hashlib
import json
import sys
from pathlib import Path

from rknn.api import RKNN

root = Path(__file__).resolve().parents[1]
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text())
for item in manifest["artifacts"]:
    slug = item["id"]
    output = root / item["file"]
    output.parent.mkdir(exist_ok=True)
    compiler = RKNN()
    try:
        compiler.config(target_platform="rv1126b", mean_values=[[123.675, 116.28, 103.53]],
                        std_values=[[58.395, 57.12, 57.375]])
        if compiler.load_onnx(model=str(Path(sys.argv[1]) / f"{slug}.onnx")):
            raise RuntimeError(f"Failed to import {slug}")
        if compiler.build(do_quantization=False) or compiler.export_rknn(str(output)):
            raise RuntimeError(f"Failed to compile {slug}")
    finally:
        compiler.release()
    item.update(size=output.stat().st_size, sha256=hashlib.sha256(output.read_bytes()).hexdigest())
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
