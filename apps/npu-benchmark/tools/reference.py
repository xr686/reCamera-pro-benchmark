"""Evaluate the original FP32 weights against the same input pipeline on the host."""
import json
import sys
from pathlib import Path

import cv2
import torch

from export_models import CHECKPOINTS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import load_samples, preprocess

if __name__ == "__main__":
    data = Path(sys.argv[1])
    _, digest, samples = load_samples(data)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    report = {"index_sha256": digest, "predictions": {}}
    with torch.inference_mode():
        for slug, (build, weights) in CHECKPOINTS.items():
            model = build(weights=weights).eval()
            predictions = []
            for sample in samples:
                image = cv2.imread(str(data / sample["file"]))
                tensor = torch.from_numpy(preprocess(image)).permute(0, 3, 1, 2)
                output = model((tensor.float() - mean) / std)
                predictions.append(output.argsort(descending=True, stable=True)[0, :5].tolist())
            report["predictions"][slug] = predictions
    Path(sys.argv[2]).write_text(json.dumps(report))
