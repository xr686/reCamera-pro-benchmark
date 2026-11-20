"""Export the three TorchVision V1 checkpoints to static ONNX."""
import sys
from pathlib import Path

import torch
from torchvision import models

CHECKPOINTS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "shufflenet-v2": (models.shufflenet_v2_x1_0, models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
    "mobilenet-v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
}

if __name__ == "__main__":
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    for slug, (build, weight) in CHECKPOINTS.items():
        network = build(weights=weight).eval()
        if slug == "shufflenet-v2":
            # Toolkit 2.3.2 mis-infers the rank after ReduceMean over H,W.
            # GlobalAveragePool + Flatten gives the same 1024 channel means.
            layers = [network.conv1, network.maxpool, network.stage2, network.stage3,
                      network.stage4, network.conv5, torch.nn.AdaptiveAvgPool2d(1),
                      torch.nn.Flatten(1), network.fc]
            exportable = torch.nn.Sequential(*layers).eval()
            image = torch.rand(1, 3, 224, 224)
            with torch.no_grad():
                torch.testing.assert_close(network(image), exportable(image))
            network = exportable
        torch.onnx.export(network, torch.zeros(1, 3, 224, 224), target / f"{slug}.onnx",
                          input_names=["image"], output_names=["logits"], opset_version=12)
        print(slug, weight.url, flush=True)
