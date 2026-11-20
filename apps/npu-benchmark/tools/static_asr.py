"""Specialize the NeMo CTC network to a 6 second utterance for RV1126B."""

import sys
import numpy as np
import onnx
from onnx import numpy_helper
from onnxsim import simplify

m = onnx.load(sys.argv[1])
del m.metadata_props[:]
length = next(i for i in m.graph.input if i.name == "length")
m.graph.input.remove(length)
m.graph.initializer.append(numpy_helper.from_array(np.array([600], np.int64), "length"))
m, ok = simplify(m, overwrite_input_shapes={"audio_signal": [1, 80, 600]})
if not ok:
    raise ValueError("Static specialization failed")
inferred = onnx.shape_inference.infer_shapes(m)
ranks = {
    t.name: len(t.type.tensor_type.shape.dim)
    for t in [*inferred.graph.value_info, *inferred.graph.input]
}
for n in m.graph.node:
    if n.op_type in ("Softmax", "LogSoftmax"):
        for a in n.attribute:
            if a.name == "axis" and a.i < 0:
                a.i += ranks[n.input[0]]
onnx.save(m, sys.argv[2])
