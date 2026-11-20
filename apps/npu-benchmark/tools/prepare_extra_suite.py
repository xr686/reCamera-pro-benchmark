"""Prepare the incremental model pack, labelled fixtures and original-ONNX numerical references."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extra_tasks import contracts
from task_models import Model
from task_inputs import prepare, read_sample, fingerprint
from task_decode import decode


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def create(w, imagenette, voc, patches):
    root = w / "fixtures"
    root.mkdir(exist_ok=True)
    groups = {}

    def save(group, name, image, **labels):
        f = f"{group}/{name}.png"
        (root / group).mkdir(exist_ok=True)
        if not cv2.imwrite(str(root / f), image):
            raise OSError(f)
        groups.setdefault(group, []).append(dict(file=f, **labels))
        return f

    records = json.loads((imagenette / "index.json").read_text())["samples"]
    counts = {}
    for row in records:
        label = row["label"]
        counts[label] = counts.get(label, 0) + 1
        if counts[label] > 10:
            continue
        f = "imagenette-extra/" + row["file"]
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(imagenette / row["file"], root / f)
        groups.setdefault("imagenette-extra", []).append(dict(file=f, label=label))
    val = sorted(
        (voc / "ImageSets/Segmentation/val.txt").read_text().split(),
        key=lambda n: hashlib.sha256(n.encode()).digest(),
    )[:20]
    for name in val:
        im = cv2.imread(str(voc / "JPEGImages" / (name + ".jpg")))
        label = np.array(Image.open(voc / "SegmentationClass" / (name + ".png")))
        target = f"voc-extra/{name}-mask.png"
        save("voc-extra", name, im, mask=target)
        cv2.imwrite(str(root / target), label)
    rng = np.random.default_rng(307)
    for i, name in enumerate(val[:10]):
        original = cv2.imread(str(voc / "JPEGImages" / (name + ".jpg")))
        for g, side in (
            ("exposure", 192),
            ("noisy-gray", 128),
            ("noisy-rgb", 128),
            ("style", 224),
        ):
            image = cv2.resize(original, (side, side))
            if g == "noisy-gray":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if g == "exposure":
                input_image = np.rint((image / 255) ** 2.2 * 255).astype(np.uint8)
            elif g.startswith("noisy"):
                input_image = np.clip(
                    np.rint(image + rng.normal(0, 25, image.shape)), 0, 255
                ).astype(np.uint8)
            else:
                input_image = image
            target = f"{g}/{i}-target.png"
            save(g, str(i), input_image, target=target)
            if g != "style":
                cv2.imwrite(str(root / target), image)
    for i in range(12):
        folder = patches / f"frame_{1 + 17 * i}"
        a, b = [cv2.imread(str(folder / f"{n}.ppm")) for n in (1, 2)]

        def scale(img):
            return np.diag([256 / img.shape[1], 192 / img.shape[0], 1])

        h = scale(b) @ np.loadtxt(folder / "H_1_2") @ np.linalg.inv(scale(a))
        for j, img in enumerate((a, b)):
            save(
                "homography",
                f"{i}-{j}",
                cv2.resize(img, (256, 192)),
                pair_id=i,
                side=j,
                homography=h.tolist(),
            )
    for i in range(16):
        image = np.full((320, 320, 3), 225, np.uint8)
        ls = []
        for j in range(2 + i % 3):
            # Parallel separated lines prevent ambiguity at intersections.
            a, b = (
                (int(rng.integers(12, 55)), 35 + j * 62),
                (int(rng.integers(225, 300)), int(30 + j * 62 + rng.integers(-18, 20))),
            )
            if i % 2:
                a, b = a[::-1], b[::-1]
            cv2.line(image, a, b, (25, 25, 25), 2, cv2.LINE_AA)
            ls.append([a, b])
        save("line-scenes", str(i), image, segments=ls)
    meta = {}
    for m in contracts(Model):
        src = w / "data" / (m.corpus + ".json")
        if m.pixels in ("signal", "logmel") or m.corpus == "digits":
            if src.exists():
                raw = json.loads(src.read_text())
                groups[m.corpus] = raw.pop("records")
                meta[m.file] = raw
    return groups, meta


def build(args):
    w = args.work
    groups, training = create(w, args.classification, args.voc, args.patches)
    fx = w / "fixtures"
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    suite = {"models": {}, "files": {}}
    from prepare_clip import build_galleries

    clip_models = [m for m in contracts(Model) if m.decoder == "clip-rank"]
    with_clip = all(
        (w / "compiled" / (m.file + ".rknn")).is_file() for m in clip_models
    )
    if with_clip:
        build_galleries(w, groups, clip_models, opts)
    for m in contracts(Model):
        model = w / "compiled" / (m.file + ".rknn")
        onnx = w / "sources" / (m.file + ".onnx")
        if not model.is_file():
            continue
        item = {"sha256": digest(model), "onnx_sha256": digest(onnx), "reference": {}}
        optimized = w / "data" / (m.file + "-rv.onnx")
        if optimized.is_file():
            item["compiled_graph_sha256"] = digest(optimized)
        if m.pixels == "cropnet":
            item["transform"] = json.loads(
                (w / "sources" / (m.file + ".json")).read_text()
            )
        if m.file in training:
            item["training"] = training[m.file]
            if m.pixels == "signal":
                item["normalization"] = {
                    k: training[m.file][k] for k in ("mean", "std")
                }
        session = ort.InferenceSession(
            str(onnx), opts, providers=["CPUExecutionProvider"]
        )
        for i, rec in enumerate(
            groups[m.corpus] if m.decoder == "stylize" else groups[m.corpus][:3]
        ):
            inp = prepare(m, read_sample(fx, rec), item)
            out = session.run(
                None, dict(zip([x.name for x in session.get_inputs()], inp))
            )
            if not all(np.isfinite(x).all() for x in out):
                raise ValueError(m.file)
            if m.decoder == "stylize":
                cv2.imwrite(str(fx / rec["target"]), decode(m, out)["image"])
            if i < 3:
                item["reference"][rec["file"]] = fingerprint(out)
        suite["models"][m.file] = item
        print(m.file, flush=True)
    (fx / "groups.json").write_text(json.dumps(groups, ensure_ascii=False))
    files = {"groups.json"} | {
        r[key]
        for group in groups.values()
        for r in group
        for key in ("file", "mask", "target")
        if key in r
    }
    if with_clip:
        files.update(
            [
                "clip-image-gallery.npz",
                "clip-text-gallery.npz",
                "clip-tokenizer/vocab.json",
                "clip-tokenizer/merges.txt",
            ]
        )
    suite["files"] = {name: digest(fx / name) for name in sorted(files)}
    (w / "suite.json").write_text(
        json.dumps(suite, ensure_ascii=False, allow_nan=False)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("work", "classification", "voc", "patches"):
        p.add_argument(arg, type=Path)
    build(p.parse_args())
