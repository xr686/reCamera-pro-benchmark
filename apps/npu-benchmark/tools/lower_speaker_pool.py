"""Specialize spatial means to global pooling for RKNN-Toolkit2's TDNN rank handling."""

import argparse
import onnx
from onnx import helper
from pathlib import Path


def lower(src, dst):
    m = onnx.load(src)
    nodes = []
    for n in m.graph.node:
        attrs = {a.name: helper.get_attribute_value(a) for a in n.attribute}
        if n.op_type != "ReduceMean" or list(attrs.get("axes", [])) != [2, 3]:
            nodes.append(n)
            continue
        target = n.output[0]
        flat = not attrs.get("keepdims", 1)
        p = target + "_4d" if flat else target
        nodes.append(helper.make_node("GlobalAveragePool", [n.input[0]], [p]))
        if flat:
            nodes.append(helper.make_node("Flatten", [p], [target], axis=1))
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("target", type=Path)
    args = p.parse_args()
    lower(args.source, args.target)
