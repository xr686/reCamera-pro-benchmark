"""Export official TorchVision checkpoints with static RV1126B image inputs."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torchvision
from torchvision import models

CLASSIFIERS = (
    "squeezenet1_1",
    "densenet121",
    "efficientnet_b0",
    "mnasnet0_5",
    "mobilenet_v3_small",
    "regnet_x_400mf",
    "googlenet",
    "resnext50_32x4d",
    "convnext_tiny",
    "swin_t",
    "vit_b_32",
    "shufflenet_v2_x0_5",
)
SEGMENTERS = (
    "lraspp_mobilenet_v3_large",
    "deeplabv3_mobilenet_v3_large",
    "fcn_resnet50",
)


class SegmentationOutput(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, image):
        return self.network(image)["out"]


def export(work, name):
    torch.set_num_threads(2)
    enum = models.get_model_weights(name)
    weights = enum.DEFAULT
    network = models.get_model(name, weights=weights).eval()
    transform = weights.transforms()
    size = 256 if name in SEGMENTERS else transform.crop_size[0]
    if name == "mnasnet0_5":
        network = nn.Sequential(
            network.layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(1), network.classifier
        ).eval()
    if name == "shufflenet_v2_x0_5":
        network = nn.Sequential(
            network.conv1,
            network.maxpool,
            network.stage2,
            network.stage3,
            network.stage4,
            network.conv5,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            network.fc,
        ).eval()
    if name in SEGMENTERS:
        network = SegmentationOutput(network).eval()
    source = work / "sources" / (name + ".onnx")
    torch.onnx.export(
        network,
        torch.zeros(1, 3, size, size),
        source,
        input_names=["image"],
        output_names=["scores"],
        opset_version=14,
    )
    (work / "sources" / (name + ".json")).write_text(
        json.dumps(
            {
                "weights": str(weights),
                "url": weights.url,
                "torchvision": torchvision.__version__,
                "size": size,
                "resize": getattr(transform, "resize_size", None),
                "interpolation": str(transform.interpolation.value),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("model", choices=CLASSIFIERS + SEGMENTERS)
    a = p.parse_args()
    export(a.work, a.model)
