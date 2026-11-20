"""Keep SuperPoint's channelwise L2 normalization numerically stable on RV1126B FP16."""

import argparse
import numpy as np
import onnx
from onnx import numpy_helper, helper


def lower(src, dst):
    model = onnx.load(src)
    g = model.graph
    norm = next(n for n in g.node if n.op_type == "ReduceL2")
    raw = norm.input[0]
    out = g.output[1].name
    ch = g.output[1].type.tensor_type.shape.dim[1].dim_value
    assert ch == 256
    nodes = list(g.node)
    cut = next(i for i, x in enumerate(nodes) if x == norm)
    del g.node[:]
    g.node.extend(nodes[:cut])
    g.initializer.extend(
        [
            numpy_helper.from_array(np.float32(1 / 32), "desc_scale"),
            numpy_helper.from_array(
                np.ones((1, ch, 1, 1), np.float32), "desc_channel_sum"
            ),
        ]
    )
    g.node.extend(
        [
            helper.make_node("Mul", [raw, "desc_scale"], ["desc_scaled"]),
            helper.make_node("Mul", ["desc_scaled", "desc_scaled"], ["desc_square"]),
            helper.make_node(
                "Conv",
                ["desc_square", "desc_channel_sum"],
                ["desc_norm2"],
                kernel_shape=[1, 1],
            ),
            helper.make_node("Sqrt", ["desc_norm2"], ["desc_norm"]),
            helper.make_node("Div", ["desc_scaled", "desc_norm"], [out]),
        ]
    )
    onnx.checker.check_model(model)
    onnx.save(model, dst)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("target")
    a = p.parse_args()
    lower(a.source, a.target)
