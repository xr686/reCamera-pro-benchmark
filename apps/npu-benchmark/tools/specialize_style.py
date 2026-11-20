"""Lower a fixed-size ONNX style graph to explicit reflection and spatial normalization."""

import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper
from onnxsim import simplify


def lower(source, target):
    model, ok = simplify(onnx.load(source))
    if not ok:
        raise ValueError("Static simplification failed")
    model = onnx.shape_inference.infer_shapes(model)
    g = model.graph
    shapes = {
        r.name: [d.dim_value for d in r.type.tensor_type.shape.dim]
        for r in [*g.input, *g.value_info]
    }
    weights = {r.name: numpy_helper.to_array(r) for r in g.initializer}
    nodes = []
    for i, n in enumerate(g.node):
        tag = f"style_{i}_"

        def const(key, array):
            name = tag + key
            g.initializer.append(numpy_helper.from_array(np.asarray(array), name))
            return name

        def op(kind, args, key, **attrs):
            name = tag + key
            nodes.append(helper.make_node(kind, args, [name], **attrs))
            return name

        x = n.input[0]
        if n.op_type == "InstanceNormalization":
            attr = {a.name: helper.get_attribute_value(a) for a in n.attribute}
            scale = const("gain", weights[n.input[1]].reshape(1, -1, 1, 1))
            bias = const("bias", weights[n.input[2]].reshape(1, -1, 1, 1))
            mean = op("GlobalAveragePool", [x], "mean")
            delta = op("Sub", [x, mean], "delta")
            var = op(
                "GlobalAveragePool", [op("Mul", [delta, delta], "square")], "variance"
            )
            den = op(
                "Sqrt",
                [
                    op(
                        "Add",
                        [var, const("epsilon", np.float32(attr.get("epsilon", 1e-5)))],
                        "stable",
                    )
                ],
                "den",
            )
            scaled = op("Mul", [op("Div", [delta, den], "z"), scale], "scaled")
            nodes.append(helper.make_node("Add", [scaled, bias], list(n.output)))
        elif n.op_type == "Pad" and any(
            a.name == "mode" and a.s == b"reflect" for a in n.attribute
        ):
            pads = next(
                helper.get_attribute_value(a) for a in n.attribute if a.name == "pads"
            )
            for axis in (2, 3):
                length = shapes[n.input[0]][axis]
                ids = np.pad(
                    np.arange(length, dtype=np.int64),
                    (pads[axis], pads[axis + 4]),
                    mode="reflect",
                )
                x = op(
                    "Gather", [x, const(f"reflect{axis}", ids)], f"pad{axis}", axis=axis
                )
            nodes.append(helper.make_node("Identity", [x], list(n.output)))
        else:
            nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    model, ok = simplify(model)
    if not ok:
        raise ValueError("Specialized graph failed comparison")
    onnx.checker.check_model(model)
    onnx.save(model, target)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("target")
    a = p.parse_args()
    lower(a.source, a.target)
