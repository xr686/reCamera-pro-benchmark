"""Express the three noninteger half-pixel resizes in a trained Fire pyramid as fixed linear maps."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def coefficients(source, target):
    coordinate = np.clip(
        (np.arange(target) + 0.5) * source / target - 0.5, 0, source - 1
    )
    low = np.floor(coordinate).astype(int)
    high = np.minimum(low + 1, source - 1)
    fraction = coordinate - low
    weights = np.zeros((target, source), np.float32)
    weights[np.arange(target), low] = 1 - fraction
    weights[np.arange(target), high] += fraction
    return weights[:, :, None, None]


def lower(src, dst):
    model = onnx.load(src)
    g = onnx.shape_inference.infer_shapes(model).graph
    shapes = {
        v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
        for v in g.value_info
    }
    nodes = []
    found = []
    for node in model.graph.node:
        if node.op_type != "Resize" or shapes.get(node.input[0], [])[-2:] not in (
            [47, 47],
            [23, 23],
            [11, 11],
        ):
            nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        assert (
            attrs["coordinate_transformation_mode"] == b"half_pixel"
            and attrs["mode"] == b"linear"
        )
        side = shapes[node.input[0]][-1]
        found.append(side)
        prefix = "fixed_interp_" + str(side)
        weight = prefix + "_weight"
        model.graph.initializer.append(
            numpy_helper.from_array(coefficients(side, 48), weight)
        )
        nodes.extend(
            [
                helper.make_node(
                    "Transpose", [node.input[0]], [prefix + "_x"], perm=[0, 3, 1, 2]
                ),
                helper.make_node(
                    "Conv",
                    [prefix + "_x", weight],
                    [prefix + "_width"],
                    kernel_shape=[1, 1],
                ),
                helper.make_node(
                    "Transpose", [prefix + "_width"], [prefix + "_y"], perm=[0, 3, 2, 1]
                ),
                helper.make_node(
                    "Conv",
                    [prefix + "_y", weight],
                    [prefix + "_height"],
                    kernel_shape=[1, 1],
                ),
                helper.make_node(
                    "Transpose",
                    [prefix + "_height"],
                    [node.output[0]],
                    perm=[0, 2, 1, 3],
                ),
            ]
        )
    assert sorted(found) == [11, 23, 47]
    # Discard orphaned dynamic size calculations after replacing the Resize nodes.
    used = {o.name for o in model.graph.output}
    keep = []
    for node in reversed(nodes):
        if any(k in used for k in node.output):
            keep.append(node)
            used.update(node.input)
    del model.graph.node[:]
    model.graph.node.extend(reversed(keep))
    onnx.checker.check_model(model)
    onnx.save(model, dst)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("target", type=Path)
    args = p.parse_args()
    lower(args.source, args.target)
