"""Export analytical optical-flow, illuminant and near-duplicate retrieval graphs."""

import argparse, json, sys, zipfile
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from field_models import FLOW, COLOR, COPIES


class Flow(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        deriv = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        self.register_buffer("grad", torch.stack([deriv, deriv.T])[:, None])
        self.register_buffer("shift", torch.eye(49).reshape(49, 1, 7, 7))

    def forward(self, x):
        a, b = x[:, :1], x[:, 1:]
        if self.kind == "flow-sad":
            shifted = F.conv2d(b, self.shift, padding=3)
            return F.avg_pool2d((shifted - a).abs(), 7, 1, 3)
        grad = F.conv2d(16 * (a + b), self.grad, padding=1)
        ix, iy = grad[:, :1], grad[:, 1:]
        dt = 32 * (b - a)
        if self.kind == "flow-lk":
            avg = lambda v: F.avg_pool2d(v, 7, 1, 3)
            xx, xy, yy = avg(ix * ix) + 0.15, avg(ix * iy), avg(iy * iy) + 0.15
            xt, yt = avg(ix * dt), avg(iy * dt)
            det = (xx * yy - xy * xy).clamp_min(0.021)
            return torch.cat([(xy * yt - yy * xt) / det, (xy * xt - xx * yt) / det], 1)
        u, v = ix * 0, iy * 0
        for i in range(12):
            um, vm = F.avg_pool2d(u, 3, 1, 1), F.avg_pool2d(v, 3, 1, 1)
            step = (ix * um + iy * vm + dt) / (ix * ix + iy * iy + 1.0)
            u, v = um - ix * step, vm - iy * step
        return torch.cat([u, v], 1)


class Illuminant(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        if self.kind == "wb-edge":
            dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
            dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
            return F.adaptive_avg_pool2d(dx, 1) + F.adaptive_avg_pool2d(dy, 1)
        if self.kind == "wb-shades":
            return (
                F.adaptive_avg_pool2d((x * x) ** 2, 1).clamp_min(0.000001).sqrt().sqrt()
            )
        return F.adaptive_avg_pool2d(x, 1)


class CopyHash(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        n = 32
        basis = np.cos(np.pi * np.arange(8)[:, None] * (np.arange(n) + 0.5) / n)
        basis[0] *= 1 / np.sqrt(2)
        basis *= np.sqrt(2 / n)
        filt = np.einsum("ki,lj->klij", basis, basis).reshape(64, 1, 32, 32)[1:]
        self.register_buffer("dct", torch.from_numpy(filt).float())

    def forward(self, x):
        if self.kind == "hash-difference":
            rows = F.avg_pool2d(x, (4, 3), (4, 3))[:, :, :, :9]
            z = (rows[:, :, :, 1:] - rows[:, :, :, :-1]).flatten(1)
        else:
            z = F.conv2d(x, self.dct).flatten(1)
            if self.kind == "hash-spectrum":
                return z
            z = z - z.mean(1, keepdim=True)
        return z  # Convert signed margins to binary bits in the app after RKNN inference.


def fixtures(work, previous):
    root = work / "fixtures"
    for name in ["flow", "wb", "copies"]:
        (root / name).mkdir(exist_ok=True)

    def save(ds, recs):
        (work / "data" / (ds + "-records.json")).write_text(json.dumps(recs))

    recs = []
    with (
        zipfile.ZipFile(work / "data/flow.zip") as images,
        zipfile.ZipFile(work / "data/flow-gt.zip") as targets,
    ):
        for f in sorted(n for n in targets.namelist() if n.endswith("/flow10.flo")):
            scene = Path(f).parent.name
            raw = targets.read(f)
            assert raw[:4] == b"PIEH"
            width, height = np.frombuffer(raw[4:12], "<i4")
            truth = np.frombuffer(raw[12:], "<f4").reshape(height, width, 2).copy()
            old_valid = np.isfinite(truth).all(-1) & (np.abs(truth).max(-1) < 1e8)
            pair = [
                cv2.imdecode(
                    np.frombuffer(
                        images.read(f"other-data/{scene}/frame{i}.png"), np.uint8
                    ),
                    0,
                )
                for i in (10, 11)
            ]
            reduced = [
                cv2.resize(t, (160, 112), interpolation=cv2.INTER_AREA) for t in pair
            ]
            valid = (
                cv2.resize(
                    old_valid.astype(np.uint8),
                    (160, 112),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
            gt = cv2.resize(
                truth, (160, 112), interpolation=cv2.INTER_NEAREST
            ) * np.array([160 / width, 112 / height], np.float32)
            valid[:3] = False
            valid[-3:] = False
            valid[:, :3] = False
            valid[:, -3:] = False
            gt[~valid] = np.nan
            path = f"flow/{scene}"
            np.save(root / (path + ".npy"), np.stack(reduced))
            np.save(root / (path + "-gt.npy"), gt)
            recs.append(
                dict(
                    file=path + ".npy",
                    flow_target=path + "-gt.npy",
                    scene=scene,
                    valid_pixels=int(valid.sum()),
                )
            )
    save("flow", recs)
    rng = np.random.default_rng(205)
    recs = []
    for i in range(48):
        # Matte surfaces with a randomly biased reflectance distribution and a shared RGB illuminant.
        bias = rng.uniform(0.3, 0.95, (1, 1, 3))
        refl = np.clip(rng.uniform(0, 1, (8, 8, 3)) * 0.7 + bias * 0.22, 0, 1)
        shading = cv2.resize(
            rng.uniform(0.55, 1.0, (6, 6)).astype(np.float32),
            (96, 96),
            interpolation=cv2.INTER_CUBIC,
        )
        gain = rng.uniform(0.45, 1.40, 3)
        im = np.clip(
            cv2.resize(refl, (96, 96), interpolation=cv2.INTER_NEAREST)
            * shading[:, :, None]
            * gain,
            0,
            1,
        )
        f = f"wb/{i}.png"
        cv2.imwrite(str(root / f), (im[:, :, ::-1] * 255).round().astype(np.uint8))
        recs.append(dict(file=f, gains=gain.tolist()))
    save("wb", recs)
    recs = []
    g = json.loads((previous / "groups.json").read_text())["imagenette-extra"]
    for n, r in enumerate(g[:60]):
        im = cv2.resize(
            cv2.imread(str(previous / r["file"])),
            (120, 120),
            interpolation=cv2.INTER_AREA,
        )
        jpeg = cv2.imdecode(
            cv2.imencode(
                ".jpg",
                np.clip(im.astype(float) * 0.75 + 20, 0, 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 22],
            )[1],
            1,
        )
        crop = cv2.resize(im[10:110, 7:109], (120, 120))
        for idx, img in enumerate((im, jpeg, crop)):
            f = f"copies/{n}-{idx}.png"
            cv2.imwrite(str(root / f), img)
            recs.append(
                dict(
                    file=f,
                    subject=n,
                    role="gallery" if idx == 0 else "query",
                    transform=["original", "jpeg-exposure", "crop"][idx],
                )
            )
    save("copies", recs)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("previous", type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    fixtures(a.work, a.previous)
    for group, cls, shape in [
        (FLOW, Flow, (1, 2, 112, 160)),
        (COLOR, Illuminant, (1, 3, 96, 96)),
        (COPIES, CopyHash, (1, 1, 32, 32)),
    ]:
        for key in group:
            torch.onnx.export(
                cls(key).eval(),
                torch.zeros(shape),
                a.work / "sources" / (key + ".onnx"),
                input_names=["image"],
                opset_version=13,
            )
