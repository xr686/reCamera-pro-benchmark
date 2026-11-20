"""Express VOLO's fixed-stride local patch extraction and overlap-add as static slices and sums."""

from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F


def unfold(x, kernel_size, dilation=1, padding=0, stride=1):
    assert (kernel_size, dilation, padding, stride) == (3, 1, 1, 2)
    h, w = map(int, x.shape[-2:])
    v = F.pad(x, (1, 1, 1, 1))
    taps = [v[:, :, i : i + h : 2, j : j + w : 2] for i in range(3) for j in range(3)]
    return torch.stack(taps, dim=2).flatten(1, 2).flatten(2)


def fold(x, output_size, kernel_size, dilation=1, padding=0, stride=1):
    assert (kernel_size, dilation, padding, stride) == (3, 1, 1, 2)
    h, w = map(int, output_size)
    assert h % 2 == w % 2 == 0
    rows, cols = h // 2, w // 2
    v = x.reshape(1, -1, 9, rows, cols)
    # Each output parity receives up to four overlapping patches. A top or left
    # tap comes from the next patch on that axis; zero-padding handles the boundary.
    parities = []
    for dy in range(2):
        for dx in range(2):
            terms = []
            for i in [1] if dy == 0 else [0, 2]:
                for j in [1] if dx == 0 else [0, 2]:
                    sy, sx = int(i == 0), int(j == 0)
                    t = v[:, :, 3 * i + j]
                    if sy or sx:
                        t = F.pad(t, (0, sx, 0, sy))[
                            :, :, sy : sy + rows, sx : sx + cols
                        ]
                    terms.append(t)
            parities.append(sum(terms[1:], terms[0]))
    blocks = torch.stack(parities, dim=2).flatten(1, 2)
    return F.pixel_shuffle(blocks, 2)


def lower(model, dest: Path):
    with patch.object(F, "fold", fold), patch.object(F, "unfold", unfold):
        torch.onnx.export(
            model,
            torch.zeros(1, 3, 224, 224),
            dest,
            input_names=["image"],
            opset_version=14,
        )
