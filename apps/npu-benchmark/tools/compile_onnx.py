"""Compile a named ONNX graph with fixed RV1126B inputs. Keep numeric preprocessing in the app."""

import argparse
from pathlib import Path

import onnx
from rknn.api import RKNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument(
        "--shape",
        action="append",
        help="Override an input shape in graph order, e.g. 1,3,320,320",
    )
    parser.add_argument("--disable-rule", action="append", help="Disable a specific Toolkit2 optimizer rule for this graph")
    args = parser.parse_args()
    graph = onnx.load(str(args.source)).graph
    weights = {x.name for x in graph.initializer}
    inputs = [x for x in graph.input if x.name not in weights]
    shapes = (
        [list(map(int, s.split(","))) for s in args.shape]
        if args.shape
        else [[d.dim_value or 1 for d in x.type.tensor_type.shape.dim] for x in inputs]
    )
    if len(shapes) != len(inputs):
        raise ValueError("One static shape is required per graph input")
    compiler = RKNN()
    try:
        compiler.config(target_platform="rv1126b", **({"disable_rules": args.disable_rule} if args.disable_rule else {}))
        if compiler.load_onnx(
            model=str(args.source),
            inputs=[i.name for i in inputs],
            input_size_list=shapes,
        ):
            raise RuntimeError("ONNX import failed")
        if compiler.build(do_quantization=False) or compiler.export_rknn(
            str(args.target)
        ):
            raise RuntimeError("RKNN compilation failed")
    finally:
        compiler.release()


if __name__ == "__main__":
    main()
