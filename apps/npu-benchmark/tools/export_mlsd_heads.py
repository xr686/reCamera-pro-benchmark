"""Move M-LSD's final TopK coordinate selection to host postprocessing."""

import sys
import onnx
from onnx import helper, TensorProto

g = onnx.load(sys.argv[1])
head = next(n for n in g.graph.node if n.op_type == "TopK")
prob = head.input[0]
line = g.graph.output[2].name
keep = {prob, line}
live = []
for n in reversed(g.graph.node):
    if keep.intersection(n.output):
        live.append(n)
        keep.update(n.input)
weights = [v for v in g.graph.initializer if v.name in keep]
del g.graph.node[:]
g.graph.node.extend(reversed(live))
del g.graph.initializer[:]
g.graph.initializer.extend(weights)
del g.graph.output[:]
g.graph.output.extend(
    [
        helper.make_tensor_value_info(prob, TensorProto.FLOAT, [1, 25600]),
        helper.make_tensor_value_info(line, TensorProto.FLOAT, [1, 160, 160, 4]),
    ]
)
onnx.checker.check_model(g)
onnx.save(g, sys.argv[2])
