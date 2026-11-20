"""Expose NCHW on an NHWC image ONNX graph for Toolkit2's image importer."""

import sys
import onnx
from onnx import TensorProto, helper

m = onnx.load(sys.argv[1])
g = m.graph
old = g.input[0]
dims = [v.dim_value for v in old.type.tensor_type.shape.dim]
name = old.name
if len(dims) != 4 or dims[-1] not in (3, 4):
    raise ValueError(dims)
g.input.remove(old)
g.input.append(
    helper.make_tensor_value_info(
        "rgb_nchw", TensorProto.FLOAT, [dims[0], dims[-1], *dims[1:3]]
    )
)
g.node.insert(0, helper.make_node("Transpose", ["rgb_nchw"], [name], perm=[0, 2, 3, 1]))
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
