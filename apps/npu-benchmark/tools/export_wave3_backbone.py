"""Export a pretrained timm classifier at its benchmark crop size."""

from pathlib import Path
import argparse
import json
import sys
import torch
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extended_models import BACKBONES


def export(work, key, model_name=None, opset=14):
    torch.set_num_threads(2)
    name = model_name or BACKBONES[key][1]
    model = timm.create_model(name, pretrained=True).eval()
    cfg = model.pretrained_cfg
    side = cfg["input_size"][-1]
    torch.onnx.export(
        model,
        torch.zeros(1, 3, side, side),
        work / "sources" / (key + ".onnx"),
        input_names=["image"],
        opset_version=opset,
    )
    transform = dict(
        weights=name,
        url=("https://huggingface.co/" + cfg["hf_hub_id"] if cfg.get("hf_hub_id") else cfg["url"]),
        size=side,
        resize=[int(side / cfg.get("crop_pct", 0.875))],
        interpolation=cfg["interpolation"],
        mean=cfg["mean"],
        std=cfg["std"],
    )
    (work / "sources" / (key + ".json")).write_text(json.dumps(transform))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("model", choices=BACKBONES)
    args = p.parse_args()
    export(args.work, args.model)
