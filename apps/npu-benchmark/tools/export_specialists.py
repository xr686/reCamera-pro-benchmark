"""Export pretrained feature, denoising and exposure networks from their original state dictionaries."""

import argparse
import importlib.util
from pathlib import Path
import shutil
import torch
from torch import nn


def source(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Denoiser(nn.Module):
    def __init__(self, state, color):
        super().__init__()
        self.color = color
        layers = []
        for i in range(0, len(state), 2):
            weight = state[f"model.{i}.weight"]
            layers.append(nn.Conv2d(weight.shape[1], weight.shape[0], 3, padding=1))
            if i + 2 < len(state):
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)
        self.load_state_dict(state)

    def forward(self, x):
        if self.color:
            packed = nn.functional.pixel_unshuffle(x, 2)
            noise = torch.full_like(packed[:, :1], 25 / 255)
            return nn.functional.pixel_shuffle(
                self.model(torch.cat([packed, noise], 1)), 2
            )
        return x - self.model(x)


class Exposure(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)[1]


def export(w, k):
    data = w / "data"
    out = w / "sources" / (k + ".onnx")
    if k in ("u2netp", "mosaic"):
        shutil.copyfile(data / (k + ".onnx"), out)
        return
    load = lambda name: torch.load(data / name, map_location="cpu", weights_only=True)
    if k in ("dncnn25", "ffdnet25"):
        color = k == "ffdnet25"
        net = Denoiser(
            load("ffdnet_color_clip.pth" if color else "dncnn_25.pth"), color
        )
        c, h, b = (3 if color else 1), 128, 128
    elif k == "zero-dce":
        model = source(data / "zero.py").enhance_net_nopool()
        model.load_state_dict(load("zero.pth"))
        net = Exposure(model)
        c, h, b = 3, 192, 192
    elif k == "superpoint":
        net = source(data / "superpoint.py").SuperPointNet()
        net.load_state_dict(load("superpoint.pth"))
        c, h, b = 1, 192, 256
    else:
        net = source(data / "xnet.py").XFeatModel()
        net.load_state_dict(load("xfeat.pt"))
        net._unfold2d = lambda x, ws: nn.functional.pixel_unshuffle(x, ws)
        c, h, b = 1, 192, 256
    torch.set_num_threads(2)
    torch.onnx.export(
        net.eval(),
        torch.zeros(1, c, h, b),
        out,
        opset_version=13,
        input_names=["image"],
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("model")
    a = p.parse_args()
    export(a.work, a.model)
