"""Export selected pretrained timm networks to a common 224-square evaluation crop."""

from pathlib import Path
import argparse, json
import timm
import torch

NAMES = {
    "ghostnet": "ghostnet_100.in1k",
    "mobileone-s0": "mobileone_s0.apple_in1k",
    "edgenext-xxs": "edgenext_xx_small.in1k",
    "repvgg-a0": "repvgg_a0.rvgg_in1k",
}


def export(work, key):
    torch.set_num_threads(2)
    name = NAMES[key]
    model = timm.create_model(name, pretrained=True).eval()
    cfg = model.pretrained_cfg
    if key in ("mobileone-s0", "repvgg-a0"):
        from timm.utils.model import reparameterize_model

        model = reparameterize_model(model).eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 224, 224),
        work / "sources" / (key + ".onnx"),
        input_names=["image"],
        opset_version=14,
    )
    metadata = dict(
        weights=name,
        url="https://huggingface.co/" + cfg["hf_hub_id"],
        size=224,
        resize=[int(224 / cfg.get("crop_pct", 0.875))],
        interpolation=cfg["interpolation"],
        mean=cfg["mean"],
        std=cfg["std"],
        timm=timm.__version__,
    )
    (work / "sources" / (key + ".json")).write_text(json.dumps(metadata))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("model", choices=NAMES)
    a = p.parse_args()
    export(a.work, a.model)
