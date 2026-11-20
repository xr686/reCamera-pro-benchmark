"""Create CLIP English prompts and two opposite-modality galleries from CPU ONNX."""

import numpy as np
import onnxruntime as ort
from clip_benchmark import prepare_clip, unit
from task_inputs import read_sample

CATEGORIES = {
    0: "tench",
    217: "English springer spaniel",
    482: "cassette player",
    491: "chain saw",
    497: "church",
    566: "French horn",
    569: "garbage truck",
    571: "gas pump",
    574: "golf ball",
    701: "parachute",
}
TEMPLATES = ("a photo of a {}.", "a picture of the {}.", "an image of a {}.")


def build_galleries(work, groups, models, options):
    root = work / "fixtures"
    (root / "clip-texts").mkdir(exist_ok=True)
    groups["clip-images"] = groups["imagenette-extra"]
    texts = []
    for label, noun in CATEGORIES.items():
        for i, template in enumerate(TEMPLATES):
            name = f"clip-texts/{label}-{i}.txt"
            (root / name).write_text(template.format(noun))
            texts.append(dict(file=name, label=label))
    groups["clip-texts"] = texts
    for model in models:
        onnx = ort.InferenceSession(
            str(work / "sources" / (model.file + ".onnx")),
            options,
            providers=["CPUExecutionProvider"],
        )
        vectors = []
        labels = []
        for rec in groups[model.corpus]:
            x = prepare_clip(model, read_sample(root, rec))[0]
            out = onnx.run(None, {onnx.get_inputs()[0].name: x})[0]
            vectors.append(unit(out.reshape(1, -1))[0])
            labels.append(rec["label"])
        vectors = np.asarray(vectors)
        labels = np.asarray(labels)
        if model.pixels == "cliptext":
            keys = np.unique(labels)
            vectors = unit(np.stack([vectors[labels == k].mean(0) for k in keys]))
            labels = keys
        np.savez(root / (model.file + "-gallery.npz"), vectors=vectors, labels=labels)
