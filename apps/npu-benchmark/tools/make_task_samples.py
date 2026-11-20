"""Prepare the evaluation media, annotations and provenance for the additional tasks."""

import argparse
import json
from more_labels import extend
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image


def build(args):
    root, data, zoo = args.output, args.datasets, args.zoo / "examples"
    root.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)

    def save(group, name, image, **labels):
        folder = root / group
        folder.mkdir(exist_ok=True)
        rel = f"{group}/{name}.png"
        if not cv2.imwrite(str(root / rel), image):
            raise ValueError(rel)
        groups[group].append(dict(file=rel, **labels))

    for group, folder in [("coco", "coco8"), ("pose", "coco8-pose")]:
        base = args.coco / folder
        for image in sorted((base / "images/val").glob("*.jpg")):
            bgr = cv2.imread(str(image))
            h, w = bgr.shape[:2]
            rows = np.loadtxt(base / "labels/val" / (image.stem + ".txt"), ndmin=2)
            boxes = []
            for j, row in enumerate(rows):
                cls, cx, cy, bw, bh = row[:5]
                box = np.array(
                    [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
                ).clip(0, 1)
                boxes.append([int(cls), *box])
                if group == "pose" and j < 3:
                    x, y, r, b = (box * np.array([w, h, w, h])).astype(int)
                    side = max(1, int(1.25 * max(r - x, b - y)))
                    x, y = (x + r - side) // 2, (y + b - side) // 2
                    r, b = x + side, y + side
                    roi = cv2.copyMakeBorder(
                        bgr,
                        max(0, -y),
                        max(0, b - h),
                        max(0, -x),
                        max(0, r - w),
                        cv2.BORDER_CONSTANT,
                    )
                    roi = roi[
                        max(0, y) : max(0, y) + side, max(0, x) : max(0, x) + side
                    ]
                    pts = row[5:].reshape(17, 3).copy()
                    if min(r - x, b - y) < 12 or (pts[:, 2] > 0).sum() < 3:
                        continue
                    pts[:, :2] = (pts[:, :2] * [w, h] - [x, y]) / [r - x, b - y]
                    save(group, f"{image.stem}-{j}", roi, points=pts.tolist())
            if group == "coco":
                save(group, image.stem, bgr, boxes=boxes)

    voc = data / "voc/VOCdevkit/VOC2012"
    for name in (voc / "ImageSets/Segmentation/val.txt").read_text().split():
        label = np.array(Image.open(voc / "SegmentationClass" / f"{name}.png"))
        if np.count_nonzero(label == 15) < 500:
            continue
        image = cv2.imread(str(voc / "JPEGImages" / f"{name}.jpg"))
        save(
            "voc",
            name,
            image,
            mask=f"voc/{name}-mask.png",
            objectmask=f"voc/{name}-objects.png",
        )
        objects = np.array(Image.open(voc / "SegmentationObject" / f"{name}.png"))
        cv2.imwrite(str(root / "voc" / f"{name}-objects.png"), objects)
        cv2.imwrite(
            str(root / "voc" / f"{name}-mask.png"),
            np.where(label == 255, 255, label == 15).astype(np.uint8),
        )
        if len(groups["voc"]) >= 12:
            break
    base = data / "hrscvoc"
    for name in (base / "ImageSets/test.txt").read_text().split()[:8]:
        xml = ET.parse(base / "FullDataSet/Annotations" / f"{name}.xml")
        img = cv2.imread(str(base / "FullDataSet/AllImages" / f"{name}.bmp"))
        h, w = img.shape[:2]
        quads = []
        for obj in xml.findall(".//HRSC_Object"):
            x, y, bw, bh, angle = [
                float(obj.findtext("mbox_" + k)) for k in ("cx", "cy", "w", "h", "ang")
            ]
            quads.append(
                (cv2.boxPoints(((x, y), (bw, bh), np.degrees(angle))) / [w, h]).tolist()
            )
        save("ships", name, cv2.resize(img, (640, 480)), quads=quads)
    base = data / "lzx-face/valid"
    coco = json.loads((base / "_annotations.coco.json").read_text())
    for im in sorted(coco["images"], key=lambda it: it["id"])[:16]:
        w, h = im["width"], im["height"]
        boxes = []
        for ann in coco["annotations"]:
            if (
                ann["image_id"] in (im["id"], Path(im["file_name"]).stem)
                and ann["category_id"] == 1
            ):
                x, y, bw, bh = ann["bbox"]
                boxes.append([0, x / w, y / h, (x + bw) / w, (y + bh) / h])
        if not boxes:
            raise ValueError(f"No ground truth for {im}")
        save(
            "faces", str(im["id"]), cv2.imread(str(base / im["file_name"])), boxes=boxes
        )
    ids = {"anger": 0, "disgust": 1, "fear": 2, "happy": 3, "sadness": 5, "surprise": 6}
    for label, idx in ids.items():
        for image in sorted((data / "emotion/test" / label).glob("*.jpg"))[:12]:
            save(
                "portrait",
                image.name.split(".")[0],
                cv2.imread(str(image)),
                expression=idx,
                subject=image.name.split("_")[0],
            )
    for label in ("rock", "paper", "scissors"):
        for image in sorted((data / "hands/rps-cv-images" / label).glob("*.png"))[:5]:
            save("hands", image.stem, cv2.imread(str(image)), gesture=label)
    for i, text in enumerate(
        (
            "camera",
            "rock",
            "testing",
            "model",
            "speed",
            "224",
            "hello",
            "world",
            "frame",
            "pixel",
            "input",
            "signal",
        )
    ):
        word = np.full((48, 180, 3), 245, np.uint8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.putText(
            word,
            text,
            ((180 - tw) // 2, (48 + th) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        save("words", str(i), word, text=text)
        scene = np.full((320, 320, 3), 245, np.uint8)
        cv2.putText(
            scene,
            text,
            (40, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        mat = cv2.getRotationMatrix2D((160, 160), (-1) ** i * (i % 5) * 4, 1)
        scene = cv2.warpAffine(scene, mat, (320, 320), borderValue=(245, 245, 245))
        quad = (
            cv2.transform(
                np.float32(
                    [[[40, 170 - th], [40 + tw, 170 - th], [40 + tw, 173], [40, 173]]]
                ),
                mat,
            )[0]
            / 320
        )
        save("textline", str(i), scene, quads=[quad.tolist()])
    save(
        "plate",
        "sample",
        cv2.imread(str(zoo / "LPRNet/model/test.jpg")),
        text="湘F6CL03",
    )
    save("street", "city", cv2.imread(str(zoo / "ppseg/model/test.png")))
    save("vehicle", "bus", cv2.imread(str(zoo / "ppyoloe/model/bus.jpg")))
    img = cv2.imread(str(args.aux / "motorcycle.png"))
    disp = np.load(args.aux / "disp.npz")["arr_0"]
    save("depth", "motorcycle", img, disparity="depth/disparity.npy")
    np.save(root / "depth/disparity.npy", disp)
    for i, record in enumerate(groups["coco"] + groups["voc"][:4]):
        image = cv2.imread(str(root / record["file"]))
        y = cv2.cvtColor(cv2.resize(image, (672, 672)), cv2.COLOR_BGR2YCrCb)[:, :, 0]
        low = cv2.resize(y, (224, 224), interpolation=cv2.INTER_AREA)
        save("sr", str(i), low, target=f"sr/{i}-hr.png")
        cv2.imwrite(str(root / f"sr/{i}-hr.png"), y)
    for i, rec in enumerate(groups["coco"]):
        image = cv2.imread(str(root / rec["file"]))
        h, w = image.shape[:2]
        cls, x, y, r, b = max(
            rec["boxes"], key=lambda box: (box[3] - box[1]) * (box[4] - box[2])
        )
        target = image[
            int(y * h) : max(int(y * h) + 1, int(b * h)),
            int(x * w) : max(int(x * w) + 1, int(r * w)),
        ]
        for j, (dx, dy) in enumerate([(-16, 9), (12, -8)]):
            search = np.full((256, 256, 3), 114, np.uint8)
            template = np.full((128, 128, 3), 114, np.uint8)
            template[32:96, 32:96] = cv2.resize(target, (64, 64))
            search[96 + dy : 160 + dy, 96 + dx : 160 + dx] = cv2.resize(
                target, (64, 64)
            )
            name = f"{i}-{j}"
            save(
                "track",
                name,
                search,
                template=f"track/{name}-template.png",
                box=(np.array([96 + dx, 96 + dy, 160 + dx, 160 + dy]) / 256).tolist(),
            )
            cv2.imwrite(str(root / f"track/{name}-template.png"), template)
    for group, source in [("sound", "yamnet"), ("speech", "wav2vec2")]:
        import scipy.io.wavfile
        from scipy.signal import resample_poly

        rate, pcm = scipy.io.wavfile.read(zoo / source / "model/test.wav")
        pcm = (
            pcm.astype(np.float32) / 32768
            if pcm.dtype == np.int16
            else pcm.astype(np.float32)
        )
        if pcm.ndim == 2:
            pcm = pcm.mean(1)
        if rate != 16000:
            pcm = resample_poly(pcm, 16000, rate)
        np.save(root / (group + ".npy"), pcm.astype(np.float32))
        labels = (
            {"label": 67}
            if group == "sound"
            else {
                "text": "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"
            }
        )
        groups[group].append(
            dict(file=group + ".npy", seconds=len(pcm) / 16000, **labels)
        )
    extend(args, groups, save)
    (root / "groups.json").write_text(json.dumps(groups, ensure_ascii=False))
    print({k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "datasets", "coco", "zoo", "aux"):
        parser.add_argument("--" + key, required=True, type=Path)
    build(parser.parse_args())
