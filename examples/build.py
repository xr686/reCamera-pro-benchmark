"""Build a one-model RV1126B App Center archive using the official Extension SDK."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "apps/npu-benchmark"
TOOLS = BENCH / "tools"
RESULTS = HERE.parent / "benchmark_results/app"


def index(filename):
    with (RESULTS / filename).open(encoding="utf-8", newline="") as handle:
        return {r["id"]: r for r in csv.DictReader(handle)}


def extract_weights(archive, key, dest):
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if member.isfile() and Path(member.name).name == key + ".rknn":
                with tar.extractfile(member) as source, dest.open("wb") as target:
                    shutil.copyfileobj(source, target)
                return
    raise ValueError(f"{archive} contains no {key}.rknn")


def compile_graph(key, spec, graph, target):
    if key in ("resnet18", "mobilenet-v2", "shufflenet-v2"):
        from rknn.api import RKNN
        compiler = RKNN()
        try:
            compiler.config(target_platform="rv1126b", mean_values=[[123.675,116.28,103.53]],
                            std_values=[[58.395,57.12,57.375]])
            if compiler.load_onnx(model=str(graph)) or compiler.build(do_quantization=False) or compiler.export_rknn(str(target)):
                raise RuntimeError("Base classifier conversion failed")
        finally:
            compiler.release()
        return
    cmd = [sys.executable, str(TOOLS / "compile_onnx.py"), str(graph), str(target)]
    for tensor in spec["tensors"]:
        cmd.extend(["--shape", ",".join(map(str, tensor["shape"]))])
    if key == "maxvit":
        cmd.extend(["--disable-rule", "convert_rs_add_rs_to_rs_gather_elements"])
    subprocess.run(cmd, check=True, cwd=target.parent)


def build(args, row, spec):
    sdk = Path(args.sdk).resolve()
    builder = sdk / "market/packaging/build.py"
    if not builder.is_file():
        raise ValueError("Pass --sdk with the root of a reCamera Pro Extension API checkout")
    key = row["id"]
    app_id = "demo-" + re.sub(r"[^a-z0-9-]", "-", key.lower())
    spec.update(app_id=app_id, inputs=[])
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmp:
        stage = Path(tmp) / app_id
        stage.mkdir()
        target = stage / "model.rknn"
        if args.from_pack:
            extract_weights(args.from_pack, key, target)
        elif args.onnx:
            if not index("model-hashes.csv")[key]["onnx_sha256"]:
                raise ValueError("This demo uses an upstream precompiled RKNN; use --weights")
            compile_graph(key, spec, args.onnx.resolve(), target)
        else:
            source = args.weights or HERE.parent / "models/app-architectures" / (key + ".rknn")
            if not source.is_file():
                raise ValueError(f"Supply the RKNN, a resource pack, or a conversion-ready graph. See examples/{key}/README.md")
            shutil.copyfile(source, target)
        with target.open("rb") as f:
            if f.read(4) != b"RKNN":
                raise ValueError("Expected a binary RKNN; fetch Git LFS content if this is a pointer")
            f.seek(0)
            hasher = hashlib.sha256()
            for block in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(block)
            digest = hasher.hexdigest()
        if digest != row["sha256"] and not (args.onnx or args.rebuilt):
            raise ValueError("RKNN differs from the benchmark. Use --rebuilt only for your own compatible conversion")
        spec["weights_sha256"] = digest
        if args.second and not args.input:
            raise ValueError("--second requires --input")
        paired = spec["input_kind"] in ("pair", "mask")
        if bool(args.second) != (paired and bool(args.input)):
            raise ValueError("Paired examples require both --input and --second")
        for path in (args.input, args.second):
            if path:
                name = f"input{len(spec['inputs'])}{path.suffix.lower()}"
                shutil.copyfile(path, stage / name)
                spec["inputs"].append(name)
        if args.gallery:
            if key not in ("clip-text", "clip-image"):
                raise ValueError("--gallery applies to the two CLIP encoders")
            shutil.copyfile(args.gallery, stage / "gallery.npz")
        modules = ("imagenet_inputs", "task_inputs", "task_decode", "task_models", "extra_tasks",
                   "extended_tasks", "field_tasks", "sequence_tasks", "clip_benchmark", "signal_catalog",
                   "architecture_models", "extended_models", "field_models", "sequence_models")
        for module in modules:
            shutil.copyfile(BENCH / (module + ".py"), stage / (module + ".py"))
        for name in ("app.py", "predict.py"):
            shutil.copyfile(HERE / "_shared" / name, stage / name)
        for name in ("LICENSE", "NOTICE"):
            shutil.copyfile(BENCH / name, stage / name)
        (stage / "model.json").write_text(json.dumps(spec, ensure_ascii=False))
        manifest = json.loads((BENCH / "manifest.json").read_text())
        manifest.update(id=app_id, name="Demo "+row["name"], name_zh=row["name"]+" 示例",
                        description="Run one input and save its decoded prediction.",
                        description_zh="运行一次模型推理，保存解码结果。", version=f"1.0.{args.sequence}",
                        config_schema={"groups":[]})
        manifest["release"]["sequence"] = args.sequence
        manifest["resources"]["limits"]["storage_mb"] = max(192, target.stat().st_size // 2**20 + 128)
        manifest["artifacts"] = [dict(id="weights", kind="rknn", source="bundled", file="model.rknn",
                                      mount="model.rknn", sha256=digest, size=target.stat().st_size,
                                      required=True, share_scope="content")]
        manifest["output"].update(schema="results[]{model,inference_ms,file,prediction}", fields=[
            dict(name="model", description="Measured model ID", type="string", **{"from":"results[].model"}),
            dict(name="inference_ms", description="Synchronous inference milliseconds", type="number", **{"from":"results[].inference_ms"})])
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        subprocess.run([sys.executable, str(builder), str(stage), "--out", str(out)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="model ID from the examples index")
    parser.add_argument("--sdk", default=os.environ.get("RECAMERA_SDK_ROOT"), help="reCamera Pro Extension API checkout")
    parser.add_argument("--sequence", type=int, default=1, help="increment for each changed release of this app ID")
    parser.add_argument("--out", type=Path, default=Path("dist"))
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--weights", type=Path)
    weights.add_argument("--from-pack", type=Path, help="measured model resource archive or baseline benchmark App archive")
    weights.add_argument("--onnx", type=Path, help="conversion-ready, fixed-shape ONNX matching this model")
    parser.add_argument("--rebuilt", action="store_true", help="accept a compatible RKNN with a different SHA-256")
    parser.add_argument("--input", type=Path, help="input file in the format described by the model demo")
    parser.add_argument("--second", type=Path, help="second input image for masks or template/search pairs")
    parser.add_argument("--gallery", type=Path, help="optional CLIP gallery NPZ (vectors and labels)")
    args = parser.parse_args()
    if args.sequence < 1:
        parser.error("--sequence must be a positive integer")
    rows = index("measurements.csv")
    if args.model not in rows or not args.sdk:
        parser.error("Use a measured model ID and set RECAMERA_SDK_ROOT or --sdk")
    spec = json.loads((HERE / args.model / "model.json").read_text())
    build(args, rows[args.model], spec)


if __name__ == "__main__":
    main()
