"""Check the published model demos; optionally exercise their input transforms on the host."""

import argparse
import json
import string
import sys

from build import BENCH, HERE, TOOLS, index


def verify(prepare_inputs):
    rows = index("measurements.csv")
    hashes = index("model-hashes.csv")
    specs = {p.parent.name: p for p in HERE.glob("*/model.json")}
    if specs.keys() != rows.keys():
        raise ValueError(f"Demo/measurement ID mismatch: {specs.keys() ^ rows.keys()}")
    if prepare_inputs:
        sys.path[:0] = [str(HERE / "_shared"), str(BENCH)]
        from predict import model_inputs
    for key, path in specs.items():
        spec = json.loads(path.read_text())
        if spec["id"] != key or not (path.parent / "demo.sh").samefile(HERE / "demo.sh"):
            raise ValueError(f"Invalid demo entry for {key}")
        if not (path.parent / "README.md").is_file():
            raise ValueError(f"No model instructions for {key}")
        recipe = spec.get("export")
        if bool(recipe) != bool(hashes[key]["onnx_sha256"]):
            raise ValueError(f"Missing conversion recipe or unexpected source graph for {key}")
        if recipe:
            available = {"work", *recipe["requires"]}
            for name, *args in recipe["steps"]:
                if name != "-m" and not (TOOLS / name).is_file():
                    raise ValueError(f"{key}: missing export script {name}")
                for arg in args:
                    for _, field, _, _ in string.Formatter().parse(arg):
                        if field and field not in available:
                            raise ValueError(f"{key}: undeclared dataset {field}")
        elif not spec.get("supplier_app"):
            raise ValueError(f"No RKNN supplier for {key}")
        if prepare_inputs:
            model_inputs(spec)
    print(f"Verified {len(specs)} model demos" + (" and their generated inputs" if prepare_inputs else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-inputs", action="store_true", help="requires NumPy, OpenCV and the SDK kit on PYTHONPATH")
    verify(parser.parse_args().prepare_inputs)
