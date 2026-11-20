"""Bound BlazePose classifier logits before FP16 output; probability error is < 1.2e-7."""

import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

m = onnx.load(sys.argv[1])
g = m.graph
out = g.output[0]
name = out.name
out.name = name + "_bounded"
g.initializer.extend(
    [
        numpy_helper.from_array(np.float32(-16), "logit_min"),
        numpy_helper.from_array(np.float32(16), "logit_max"),
    ]
)
g.node.append(helper.make_node("Clip", [name, "logit_min", "logit_max"], [out.name]))
onnx.checker.check_model(m)
onnx.save(m, sys.argv[2])
