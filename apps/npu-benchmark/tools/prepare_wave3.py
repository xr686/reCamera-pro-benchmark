"""Create labelled fixtures and run the same preprocessing/decoding on CPU ONNX for comparison."""

import argparse, concurrent.futures, hashlib, json, shutil, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extended_models import contracts
from task_models import Model
from task_inputs import read_sample, prepare, fingerprint
from task_decode import decode
from task_quality import quality, summarize


def digest(path):
    with path.open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
        return h.hexdigest()


def geometry(work, prev, middlebury):
    root = work / "fixtures"
    groups = {}
    groups0 = json.loads((prev / "groups.json").read_text())
    groups["imagenette-extra"] = groups0["imagenette-extra"]
    for row in groups["imagenette-extra"]:
        p = root / row["file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prev / row["file"], p)
    (root / "patch").mkdir(exist_ok=True)
    rows = []
    # Warp the second exposure onto the first plane for paired descriptor evaluation.
    for left, right in zip(groups0["homography"][::2], groups0["homography"][1::2]):
        im1 = cv2.imread(str(prev / left["file"]), 0)
        im2 = cv2.imread(str(prev / right["file"]), 0)
        h, w = im1.shape
        H = np.array(left["homography"])
        warp = cv2.warpPerspective(
            im2, H, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        )
        mask = cv2.warpPerspective(
            np.ones_like(im2), H, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        )
        mask = cv2.erode(
            mask,
            np.ones((35, 35), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        points = cv2.goodFeaturesToTrack(im1, 8, 0.01, 22, mask=mask)
        for i, (x, y) in enumerate(points[:, 0].astype(int)):
            label = left["pair_id"] * 8 + i
            for img, role in [(im1, "gallery"), (warp, "query")]:
                f = f"patch/{label}-{role}.png"
                cv2.imwrite(str(root / f), img[y - 16 : y + 16, x - 16 : x + 16])
                rows.append(dict(file=f, subject=label, role=role))
    groups["patch"] = rows
    (root / "stereo").mkdir(exist_ok=True)
    rows = []
    for scene, factor in [("motorcycle", 0.4), ("cones", 0.5), ("teddy", 0.5)]:
        if scene == "motorcycle":
            l, r = [
                cv2.imread(str(middlebury / f"motorcycle_{side}.png"), 0)
                for side in ("left", "right")
            ]
            disp = np.load(middlebury / "motorcycle_disp.npz")["arr_0"]
        else:
            l, r = [
                cv2.imread(str(work / "data" / f"{scene}-{side}.png"), 0)
                for side in ("im2", "im6")
            ]
            disp = (
                cv2.imread(str(work / "data" / f"{scene}-disp2.png"), 0).astype(
                    np.float32
                )
                / 4
            )
        size = (round(l.shape[1] * factor), round(l.shape[0] * factor))
        l, r = [cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in (l, r)]
        d = (
            cv2.resize(disp, size, interpolation=cv2.INTER_NEAREST)
            * size[0]
            / disp.shape[1]
        )
        for i, y in enumerate(np.linspace(0, size[1] - 96, 2, dtype=int)):
            for j, x in enumerate(np.linspace(0, size[0] - 160, 3, dtype=int)):
                gt = d[y : y + 96, x : x + 160].copy()
                valid = (
                    np.isfinite(gt)
                    & (gt > 0)
                    & (gt <= 31)
                    & (np.arange(160)[None] >= gt + 4)
                )
                valid[:4] = False
                valid[-4:] = False
                valid[:, -4:] = False
                gt[~valid] = np.nan
                assert valid.sum() > 100
                stem = f"stereo/{scene}-{i}-{j}"
                img = np.stack([t[y : y + 96, x : x + 160] for t in (l, r)])
                np.save(root / (stem + ".npy"), img)
                np.save(root / (stem + "-disp.npy"), gt)
                rows.append(
                    dict(
                        file=stem + ".npy",
                        disparity=stem + "-disp.npy",
                        scene=scene,
                        valid_pixels=int(valid.sum()),
                    )
                )
    groups["stereo"] = rows
    return groups


def voc_fixtures(work, voc):
    order = "aeroplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor".split()
    ids = dict(
        zip(
            order,
            [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62],
        )
    )
    names = sorted(
        (voc / "ImageSets/Main/val.txt").read_text().split(),
        key=lambda v: hashlib.sha256(v.encode()).digest(),
    )[:32]
    out = work / "fixtures/voc-objects"
    out.mkdir(exist_ok=True)
    recs = []
    for n in names:
        tree = ET.parse(voc / "Annotations" / (n + ".xml"))
        w, h = [float(tree.findtext("size/" + s)) for s in ("width", "height")]
        boxes = []
        for obj in tree.findall("object"):
            vals = [
                float(obj.findtext("bndbox/" + s))
                for s in ("xmin", "ymin", "xmax", "ymax")
            ]
            boxes.append(
                [
                    ids[obj.findtext("name")],
                    (vals[0] - 1) / w,
                    (vals[1] - 1) / h,
                    vals[2] / w,
                    vals[3] / h,
                ]
            )
        f = f"voc-objects/{n}.jpg"
        shutil.copy2(voc / "JPEGImages" / (n + ".jpg"), work / "fixtures" / f)
        recs.append(dict(file=f, classes=sorted(ids.values()), boxes=boxes))
    return recs


def build(args):
    work = args.work
    groups = geometry(work, args.previous, args.middlebury)
    groups["voc-objects"] = voc_fixtures(work, args.voc)
    for p in sorted((work / "data").glob("*-records.json")):
        groups[p.name.removesuffix("-records.json")] = json.loads(p.read_text())
    reference_suite(work, groups, contracts(Model))


def reference_suite(work, groups, catalog):
    root = work / "fixtures"
    (root / "groups.json").write_text(json.dumps(groups, ensure_ascii=False))
    selected = [
        m
        for m in catalog
        if (work / "compiled" / (m.file + ".rknn")).is_file() and m.corpus in groups
    ]

    def reference(model):
        k = model.file
        path = work / "sources" / (k + ".onnx")
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        lowered = work / "data" / (k + "-rv.onnx")
        if lowered.exists():
            meta["compiled_graph_sha256"] = digest(lowered)
        if model.pixels == "cropnet":
            meta = {"transform": meta}
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        refs = {}
        scores = []
        for i, rec in enumerate(groups[model.corpus]):
            vals = prepare(model, read_sample(root, rec), meta)
            out = sess.run(None, {v.name: z for v, z in zip(sess.get_inputs(), vals)})
            if not all(np.isfinite(z).all() for z in out):
                raise ValueError(k)
            decoded = decode(model, out)
            scores.append(quality(root, model, rec, decoded))
            if i < 3:
                refs[rec["file"]] = fingerprint(out)
        print(k, len(scores), flush=True)
        return (
            k,
            dict(
                **meta,
                sha256=digest(work / "compiled" / (k + ".rknn")),
                onnx_sha256=digest(path),
                reference=refs,
            ),
            summarize(scores),
        )

    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        measured = list(pool.map(reference, selected))
    files = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    (work / "suite.json").write_text(
        json.dumps(
            dict(models={k: m for k, m, _ in measured}, files=files), ensure_ascii=False
        )
    )
    (work / "onnx-quality.json").write_text(
        json.dumps({k: q for k, _, q in measured}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("work", "previous", "middlebury", "voc"):
        p.add_argument(name, type=Path)
    build(p.parse_args())
