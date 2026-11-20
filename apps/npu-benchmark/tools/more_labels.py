"""Small supervised sets for cross-camera retrieval, spoofing, city labels and restoration."""

from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat


def extend(args, groups, save):
    root, aux = args.output, args.aux
    import pyarrow.parquet as parquet
    import subprocess
    import xml.etree.ElementTree as ET

    count = 0
    for row in parquet.read_table(aux / "librispeech.parquet").to_pylist():
        audio = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "pipe:1",
            ],
            input=row["audio"]["bytes"],
            capture_output=True,
            check=True,
        )
        pcm = np.frombuffer(audio.stdout, np.float32)
        rate = 16000
        if not 1.25 <= len(pcm) / rate <= 5.8 or not rate == 16000:
            continue
        name = "speech-" + row["id"] + ".npy"
        np.save(root / name, pcm)
        groups["speech"].append(
            dict(file=name, seconds=len(pcm) / rate, text=row["text"].upper())
        )
        count += 1
        if count >= 8:
            break
    # Source RKNN Model Zoo test.wav may duplicate the first LibriSpeech utterance.
    seen = set()
    groups["speech"] = [
        r for r in groups["speech"] if r["text"] not in seen and not seen.add(r["text"])
    ]
    for im in ET.parse(aux / "ibug/testing.xml").findall(".//image"):
        image = cv2.imread(str(aux / "ibug" / im.get("file")))
        for i, box in enumerate(im.findall("box")):
            x, y, w, h = [int(box.get(k)) for k in ("left", "top", "width", "height")]
            margin = int(max(w, h) * 0.18)
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1, y1 = (
                min(image.shape[1], x + w + margin),
                min(image.shape[0], y + h + margin),
            )
            pts = {
                int(p.get("name")): [int(p.get("x")), int(p.get("y"))]
                for p in box.findall("part")
            }
            if len(pts) != 68:
                continue
            gt = (
                np.array([pts[k] for k in (36, 39, 42, 45, 30, 48, 54)]) - [x0, y0]
            ) / [x1 - x0, y1 - y0]
            save(
                "mesh",
                Path(im.get("file")).stem + "-" + str(i),
                image[y0:y1, x0:x1],
                facepoints=gt.tolist(),
            )
    for i, rec in enumerate(groups["portrait"][:12]):
        image = cv2.imread(str(root / rec["file"]))
        save("quality", str(i) + "-original", image, pair=i, blurred=False)
        save(
            "quality",
            str(i) + "-blur",
            cv2.GaussianBlur(image, (21, 21), 6),
            pair=i,
            blurred=True,
        )
    groups["vehicle"].clear()
    # Four corners in source image pixels (720 × 1160), manually read on CCPD example photos.
    plates = [
        ([[300, 508], [474, 502], [469, 567], [295, 580]], "皖A195K9"),
        ([[214, 508], [396, 509], [390, 586], [208, 590]], "皖AUJ299"),
        ([[312, 450], [487, 472], [494, 538], [319, 517]], "皖AT022C"),
        ([[241, 489], [419, 516], [417, 574], [238, 549]], "皖AMK620"),
        ([[209, 489], [388, 511], [395, 570], [216, 547]], "皖A256R2"),
    ]
    for i, (corners, text) in enumerate(plates):
        image = cv2.imread(str(aux / f"ccpd-{i}.jpg"))
        xy = np.float32(corners)
        save("vehicle", f"ccpd-{i}", image, quads=[(xy / [720, 1160]).tolist()])
        transform = cv2.getPerspectiveTransform(
            xy, np.float32([[0, 0], [187, 0], [187, 47], [0, 47]])
        )
        save(
            "plate",
            f"ccpd-{i}",
            cv2.warpPerspective(image, transform, (188, 48)),
            text=text,
        )
    # Face boxes manually located in the original Silent-Face sample images.
    for name, (x, y, r, b) in [
        ("F1", (186, 113, 425, 399)),
        ("F2", (134, 189, 386, 506)),
        ("T1", (90, 96, 294, 371)),
    ]:
        image = cv2.imread(str(aux / f"spoof-{name}.jpg"))
        for scale, group in [(2.7, "spoof27"), (4.0, "spoof40")]:
            h, w = image.shape[:2]
            factor = min(scale, (w - 1) / (r - x), (h - 1) / (b - y))
            bw, bh = (r - x) * factor, (b - y) * factor
            left = int(np.clip((x + r - bw) / 2, 0, w - bw))
            top = int(np.clip((y + b - bh) / 2, 0, h - bh))
            save(
                group,
                name,
                image[top : top + int(bh), left : left + int(bw)],
                live=int(name == "T1"),
            )
    # Select one query and one gallery crop from a different camera for each identity.
    with ZipFile(aux / "market1501.zip") as z:
        gallery = defaultdict(list)
        for f in sorted(z.namelist()):
            if "/bounding_box_test/" in f and f.endswith(".jpg"):
                gallery[Path(f).name.split("_")[0]].append(f)
        ids = set()
        for f in sorted(z.namelist()):
            if "/query/" not in f or not f.endswith(".jpg"):
                continue
            subject, cam = Path(f).name.split("_")[:2]
            if subject in ids or int(subject) <= 0:
                continue
            match = next(
                (
                    p
                    for p in gallery[subject]
                    if Path(p).name.split("_")[1][1] != cam[1]
                ),
                None,
            )
            if not match:
                continue
            ids.add(subject)
            for role, path in [("query", f), ("gallery", match)]:
                image = cv2.imdecode(
                    np.frombuffer(z.read(path), np.uint8), cv2.IMREAD_COLOR
                )
                save(
                    "reid",
                    role + "-" + Path(path).stem,
                    image,
                    subject=subject,
                    role=role,
                )
            if len(ids) == 20:
                break
    groups["street"].clear()
    for p in sorted(aux.glob("cam-*.png")):
        name = p.stem[4:]
        save("street", name, cv2.imread(str(p)), citymask=f"street/{name}-mask.png")
        label = np.array(Image.open(aux / ("gt-" + p.name[4:])))
        cv2.imwrite(str(root / f"street/{name}-mask.png"), label)
    for p in sorted(aux.glob("edge-*.jpg")):
        image = cv2.imread(str(p))
        annot = loadmat(p.with_suffix(".mat"), squeeze_me=True, struct_as_record=False)[
            "groundTruth"
        ]
        mask = np.stack([a.Boundaries for a in annot]).mean(0) >= 0.4
        save("edges", p.stem, image, edges=f"edges/{p.stem}-label.png")
        cv2.imwrite(str(root / f"edges/{p.stem}-label.png"), mask.astype(np.uint8))
    for i, rec in enumerate(groups["coco"] + groups["voc"][:4]):
        image = cv2.resize(cv2.imread(str(root / rec["file"])), (512, 512))
        kernel = np.zeros((9, 9))
        kernel[4] = 1 / 9
        blur = cv2.filter2D(image, -1, kernel)
        save("deblur", str(i), blur, restore=f"deblur/{i}-sharp.png")
        cv2.imwrite(str(root / f"deblur/{i}-sharp.png"), image)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[196:280, 148:256] = 1
        damaged = image.copy()
        damaged[mask > 0] = 0
        save(
            "inpaint",
            str(i),
            damaged,
            hole=f"inpaint/{i}-hole.png",
            restore=f"inpaint/{i}-clean.png",
        )
        cv2.imwrite(str(root / f"inpaint/{i}-hole.png"), mask)
        cv2.imwrite(str(root / f"inpaint/{i}-clean.png"), image)
