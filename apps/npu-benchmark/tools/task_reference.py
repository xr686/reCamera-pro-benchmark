"""Run the original graph on the first three fixtures of each task with CPU ONNX Runtime."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_models import MODELS
from task_inputs import prepare, read_sample, fingerprint


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(base):
    fixtures = base / "fixtures"
    groups = json.loads((fixtures / "groups.json").read_text())
    asset_keys = (
        "file",
        "template",
        "target",
        "mask",
        "disparity",
        "edges",
        "restore",
        "hole",
        "citymask",
        "objectmask",
    )
    assets = {"groups.json"} | {
        rec[k]
        for records in groups.values()
        for rec in records
        for k in asset_keys
        if k in rec
    }
    manifest = {
        "models": {},
        "files": {name: digest(fixtures / name) for name in sorted(assets)},
    }
    options = ort.SessionOptions()
    options.intra_op_num_threads = 3
    options.inter_op_num_threads = 1
    for m in MODELS:
        if not (base / "compiled" / (m.file + ".rknn")).exists():
            continue
        item = manifest["models"][m.file] = {
            "sha256": digest(base / "compiled" / (m.file + ".rknn"))
        }
        if m.decoder == "subword":
            token_file = (
                "nemo-tokens.txt"
                if m.file.startswith("nemo")
                else "citrinet-tokens.txt"
            )
            vocab = {}
            for line in (base / "aux" / token_file).read_text().splitlines():
                word, idx = line.rsplit(maxsplit=1)
                vocab[int(idx)] = word
            item["tokens"] = [vocab[i] for i in range(len(vocab))]
        graph = base / "sources" / (m.file + ".onnx")
        if not graph.exists():
            continue
        item["onnx_sha256"] = digest(graph)
        try:
            # ORT does not implement all blocked weight layouts in the Zoo exports.
            # Fold weights with their ONNX definition for CPU comparison.
            ref = graph
            if m.decoder in ("edge", "restore", "inpaint", "person"):
                ref = base / "folded" / graph.name
            if ref != graph:
                item["reference_graph_sha256"] = digest(ref)
            session = ort.InferenceSession(
                str(ref), options, providers=["CPUExecutionProvider"]
            )
            item["reference"] = {}
            for rec in groups[m.corpus][:3]:
                tensors = prepare(m, read_sample(fixtures, rec))
                outputs = session.run(
                    None, {v.name: x for v, x in zip(session.get_inputs(), tensors)}
                )
                item["reference"][rec["file"]] = fingerprint(outputs)
            del session
        except Exception as exc:
            item["reference_error"] = str(exc)
            print(m.file, str(exc)[:180], flush=True)
    (base / "suite.json").write_text(
        json.dumps(manifest, ensure_ascii=False, allow_nan=False)
    )
    print("Model and fixture digests written.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
