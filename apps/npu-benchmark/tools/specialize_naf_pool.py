"""Replace full-frame integral-image averages with global pooling on a fixed NAFNet tile."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def specialize(source, target):
    model = onnx.load(source)
    graph = model.graph
    dims = {
        v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
        for v in graph.value_info
    }
    weights = {v.name: v for v in graph.initializer}
    nodes = {n.name: n for n in graph.node}
    count = 0
    for n in graph.node:
        if not n.name.endswith("/sca/sca.0/Pad_1"):
            continue
        prefix = n.name.rsplit("/", 1)[0]
        feature = nodes[prefix + "/CumSum"].input[0]
        divisor = nodes[prefix + "/Div"].input[1]
        h, w = dims[feature][-2:]
        if (
            dims[n.input[0]][-2:] != (1, 1)
            or dims[n.output[0]][-2:] != (h, w)
            or float(np.asarray(numpy_helper.to_array(weights[divisor])).item())
            != h * w
            or not any(a.name == "mode" and a.s == b"edge" for a in n.attribute)
        ):
            raise ValueError(
                f"Expected a full-window mean extended with edge padding at {prefix}"
            )
        # A following pointwise convolution and broadcast multiplication consume the mean.
        n.CopyFrom(
            helper.make_node(
                "GlobalAveragePool",
                [feature],
                list(n.output),
                name=prefix + "/GlobalAveragePool",
            )
        )
        count += 1
    if count != 36:
        raise ValueError(f"Expected 36 full-window attention pools, got {count}")
    keep = set(v.name for v in graph.output)
    live = []
    for n in reversed(graph.node):
        if any(k in keep for k in n.output):
            live.append(n)
            keep.update(n.input)
    del graph.node[:]
    graph.node.extend(reversed(live))
    used = [v for v in graph.initializer if v.name in keep]
    del graph.initializer[:]
    graph.initializer.extend(used)
    del graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    specialize(**vars(parser.parse_args()))
