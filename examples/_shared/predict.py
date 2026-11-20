"""Prepare one user input and decode one RKNN result using the benchmark contracts."""

import json
from pathlib import Path
import cv2
import numpy as np

from imagenet_inputs import MODELS as BASE_MODELS, preprocess, top_five
from clip_benchmark import unit
from task_decode import decode
from task_inputs import prepare
from task_models import MODELS

CONTRACTS = {m.file: m for m in MODELS}


def image(path, shape):
    if path:
        color = cv2.IMREAD_GRAYSCALE if len(shape) == 2 else cv2.IMREAD_COLOR
        value = cv2.imread(str(path), color)
        if value is None:
            raise ValueError(f"Cannot decode image {path}")
        return value
    y, x = np.indices(shape[:2], dtype=np.uint16)
    if len(shape) == 2:
        return ((x + y) % 32 < 10).astype(np.uint8) * 255
    return np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)


def model_inputs(spec, first=None, second=None):
    key, kind, shape = spec["id"], spec["input_kind"], spec["sample_shape"]
    model = CONTRACTS.get(key)
    if kind == "tokens":
        ids = np.load(first, allow_pickle=False) if first else np.array([49406, 320] + [49407] * 18)
        if ids.shape not in ((20,), (1, 20)) or ids.dtype.kind not in "iu" or ids.min() < 0 or ids.max() > 49407:
            raise ValueError("Expected 20 CLIP token IDs in [0,49407]")
        tensors = [ids.astype(np.int64).reshape(1, 20)]
    else:
        if kind == "image":
            sample = image(first, shape)
            if model and model.pixels in ("std-patch", "bayer4"):
                sample = cv2.resize(sample, (shape[1], shape[0]))
        elif kind in ("mask", "pair"):
            sample = tuple(cv2.resize(image(path, dims), (dims[1], dims[0]))
                           for path, dims in zip((first, second), shape))
        elif kind == "text":
            sample = (first.read_text(encoding="utf-8") if first else "A small camera on a table", None)
        else:
            if first:
                sample = np.load(first, allow_pickle=False)
            else:
                values = np.arange(np.prod(shape), dtype=np.float32)
                dtype = np.dtype(spec["sample_dtype"])
                values = (values % 251) if dtype.kind in "ui" else np.sin(values * 0.09)
                sample = values.reshape(shape).astype(dtype)
            if list(sample.shape) != shape:
                raise ValueError(f"Expected raw .npy shape {shape}, got {sample.shape}")
        tensors = [preprocess(sample)] if key in BASE_MODELS else prepare(model, sample, spec["metadata"])
    for tensor, expected in zip(tensors, spec["tensors"], strict=True):
        if list(tensor.shape) != expected["shape"] or str(tensor.dtype) != expected["dtype"]:
            raise ValueError(f"{key} expected {expected}, got {tensor.shape} {tensor.dtype}")
    if model and model.layout == "nchw":
        tensors = [np.ascontiguousarray(t.transpose(0, 2, 3, 1)) if t.ndim == 4 else t for t in tensors]
    return tensors


def prediction(spec, outputs, gallery_path=None):
    key = spec["id"]
    if not outputs or any(not np.isfinite(t).all() for t in outputs):
        raise ValueError("Missing or nonfinite model output")
    if key in BASE_MODELS:
        return {"top": top_five(outputs)}
    model = CONTRACTS[key]
    gallery = None
    if model.decoder == "clip-rank":
        if gallery_path is None:
            return {"projection": unit(outputs[0].reshape(1, -1))[0]}
        with np.load(gallery_path, allow_pickle=False) as data:
            gallery = data["vectors"], data["labels"]
    return decode(model, outputs, spec["metadata"].get("tokens"), gallery)


def save_prediction(value, directory):
    arrays = {}

    def encode(item):
        if isinstance(item, dict):
            return {key: encode(val) for key, val in item.items()}
        if isinstance(item, np.ndarray):
            if item.size <= 48:
                return item.tolist()
            key = f"array_{len(arrays)}"
            arrays[key] = item
            return {"npz_key": key, "shape": list(item.shape), "dtype": str(item.dtype)}
        if isinstance(item, (tuple, list)):
            return [encode(part) for part in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    result = encode(value)
    directory.mkdir(parents=True, exist_ok=True)
    # Publish the JSON descriptor only after its optional tensor file is complete.
    if arrays:
        tmp = directory / "prediction.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.replace(directory / "prediction.npz")
    tmp = directory / "prediction.tmp.json"
    tmp.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(directory / "prediction.json")
    return result
