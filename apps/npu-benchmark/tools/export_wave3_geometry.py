"""Build disparity cost volumes and export pretrained Liberty patch descriptors."""

from pathlib import Path
import argparse, sys
import torch
from torch import nn
from torch.nn import functional as F
from kornia.feature import HardNet, SOSNet, TFeat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extended_models import PATCH, STEREO


class Stereo(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        # One convolution materializes R(x-d,y) for each disparity d.
        kernel = torch.zeros(32, 1, 1, 32)
        for d in range(32):
            kernel[d, 0, 0, 31 - d] = 1
        self.register_buffer("shifts", kernel)
        self.register_buffer(
            "sobel",
            torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]).reshape(1, 1, 3, 3)
            / 8,
        )

    def avg(self, x):
        return F.avg_pool2d(x, 7, 1, 3)

    def forward(self, x):
        l, r = x[:, :1], x[:, 1:2]
        if self.kind == "stereo-gradient":
            l = F.conv2d(l, self.sobel, padding=1)
            r = F.conv2d(r, self.sobel, padding=1)
        if self.kind == "stereo-zsad":
            l = l - self.avg(l)
            r = r - self.avg(r)
        r = F.conv2d(F.pad(r, (31, 0, 0, 0)), self.shifts)
        if self.kind == "stereo-ncc":
            # Rescale differences before covariance accumulation in FP16.
            l = (l - 0.5) * 8
            r = (r - 0.5) * 8
            a, b = self.avg(l), self.avg(r)
            cov = self.avg(l * r) - a * b
            v1 = torch.clamp(self.avg(l * l) - a * a, min=0.0064)
            v2 = torch.clamp(self.avg(r * r) - b * b, min=0.0064)
            return -cov / (torch.sqrt(v1) * torch.sqrt(v2))
        return self.avg(torch.abs(l - r))


def export(work, selected=None):
    torch.set_num_threads(2)
    for key in selected or [*PATCH, *STEREO]:
        if key in PATCH:
            mod = {"tfeat": TFeat, "sosnet": SOSNet, "hardnet": HardNet}[key]().eval()
            weights = torch.load(
                work / "data" / (key + ".pth"), map_location="cpu", weights_only=True
            )
            mod.load_state_dict(weights.get("state_dict", weights))
            # The app performs per-patch input normalization and output L2 normalization.
            if key == "tfeat":
                net = nn.Sequential(
                    *list(mod.features.children())[1:], nn.Flatten(), mod.descr
                )
            elif key == "hardnet":
                net = nn.Sequential(mod.features, nn.Flatten())
            else:
                net = nn.Sequential(*list(mod.layers.children())[1:], nn.Flatten())
            sample = torch.zeros(1, 1, 32, 32)
        else:
            net = Stereo(key)
            sample = torch.zeros(1, 2, 96, 160)
        torch.onnx.export(
            net.eval(),
            sample,
            work / "sources" / (key + ".onnx"),
            input_names=["input"],
            opset_version=13,
        )
        print(key, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("--model", choices=[*PATCH, *STEREO])
    args = p.parse_args()
    export(args.work, [args.model] if args.model else None)
