"""Evaluate the same unique fixtures using original ONNX graphs on the host CPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import cv2
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_tasks import contracts
from task_decode import decode
from task_quality import quality, summarize
from task_inputs import prepare, read_sample
from task_models import Model


def score(work):
    cv2.setNumThreads(1)
    suite = json.loads((work / "suite.json").read_text())
    fx = work / "fixtures"
    groups = json.loads((fx / "groups.json").read_text())

    def evaluate(m):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        s = ort.InferenceSession(
            str(work / "sources" / (m.file + ".onnx")),
            opts,
            providers=["CPUExecutionProvider"],
        )
        from clip_benchmark import gallery

        ref = gallery(fx, m) if m.decoder == "clip-rank" else None
        rows = []
        for rec in groups[m.corpus]:
            inputs = prepare(m, read_sample(fx, rec), suite["models"][m.file])
            out = s.run(None, dict(zip([v.name for v in s.get_inputs()], inputs)))
            rows.append(quality(fx, m, rec, decode(m, out, gallery=ref)))
        return m.file, dict(samples=len(rows), metrics=summarize(rows))

    with ThreadPoolExecutor(4) as pool:
        scores = dict(
            pool.map(
                evaluate, [m for m in contracts(Model) if m.file in suite["models"]]
            )
        )
    (work / "onnx-quality.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    score(p.parse_args().work)
