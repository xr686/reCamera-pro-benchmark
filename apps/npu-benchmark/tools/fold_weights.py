"""Materialize constant blocked DequantizeLinear weights before down-converting ONNX opset 21."""

import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper


def fold(src, dst):
    model = onnx.load(src)
    graph = model.graph
    initializers = {x.name: x for x in graph.initializer}
    keep = []
    for node in graph.node:
        if node.op_type == "ConstantOfShape":
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            data = (
                numpy_helper.to_array(attrs["value"])
                if "value" in attrs
                else np.array(0, np.float32)
            )
            name = node.output[0] + ".scalar"
            graph.initializer.append(numpy_helper.from_array(data.reshape(()), name))
            keep.append(
                helper.make_node("Expand", [name, node.input[0]], list(node.output))
            )
            continue
        if node.op_type != "DequantizeLinear" or not all(
            i in initializers for i in node.input
        ):
            keep.append(node)
            continue
        q, scale, zp = [numpy_helper.to_array(initializers[i]) for i in node.input]
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        block = attrs.get("block_size", 0)
        if block:
            axis = attrs.get("axis", 1) % q.ndim
            stop = [slice(None)] * q.ndim
            stop[axis] = slice(0, q.shape[axis])
            scale = np.repeat(scale, block, axis=axis)[tuple(stop)]
            zp = np.repeat(zp, block, axis=axis)[tuple(stop)]
        elif scale.ndim == 1:
            shape = [1] * q.ndim
            shape[attrs.get("axis", 1)] = -1
            scale, zp = scale.reshape(shape), zp.reshape(shape)
        value = (q.astype(np.float32) - zp.astype(np.float32)) * scale
        graph.initializer.append(numpy_helper.from_array(value, node.output[0]))
    del graph.node[:]
    graph.node.extend(keep)
    used = {s for n in graph.node for s in n.input} | {x.name for x in graph.output}
    weights = [x for x in graph.initializer if x.name in used]
    del graph.initializer[:]
    graph.initializer.extend(weights)
    model = onnx.version_converter.convert_version(model, 19)
    onnx.checker.check_model(model)
    onnx.save(model, dst)


if __name__ == "__main__":
    fold(Path(sys.argv[1]), Path(sys.argv[2]))
