"""CLIP preprocessing and cosine search for the fixed English retrieval benchmark."""

from functools import lru_cache
import json
import re
import cv2
import numpy as np


@lru_cache(maxsize=2)
def tokenizer(root):
    vocab = json.loads((root / "clip-tokenizer/vocab.json").read_text())
    lines = (root / "clip-tokenizer/merges.txt").read_text().splitlines()
    ranks = {
        tuple(line.split()): i
        for i, line in enumerate(lines)
        if line and not line.startswith("#")
    }
    return vocab, ranks


def encode_english(text, root):
    # The benchmark prompts contain ASCII English words and punctuation.
    vocab, ranks = tokenizer(root)
    if not text.isascii():
        raise ValueError("CLIP benchmark expects ASCII English prompts")
    ids = [vocab["<|startoftext|>"]]
    for token in re.findall(r"[a-z]+|[0-9]|[^\w\s]", text.lower()):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pair = min(zip(word, word[1:]), key=lambda p: ranks.get(p, float("inf")))
            if pair not in ranks:
                break
            merged = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i : i + 2] == pair:
                    merged.append("".join(pair))
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
        ids.extend(vocab[piece] for piece in word)
    end = vocab["<|endoftext|>"]
    return np.asarray([(ids[:19] + [end] * 20)[:20]], np.int64)


def prepare_clip(model, data):
    if model.pixels == "cliptext":
        return [encode_english(*data)]
    h, w = data.shape[:2]
    side = 224
    image = cv2.resize(
        data,
        (w * side // min(h, w), h * side // min(h, w)),
        interpolation=cv2.INTER_CUBIC,
    )
    y, x = (int(round((v - side) / 2)) for v in image.shape[:2])
    image = (
        cv2.cvtColor(image[y : y + side, x : x + side], cv2.COLOR_BGR2RGB).astype(
            np.float32
        )
        / 255
    )
    mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    return [np.ascontiguousarray(((image - mean) / std).transpose(2, 0, 1)[None])]


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def gallery(root, model):
    name = "clip-text-gallery" if model.file == "clip-image" else "clip-image-gallery"
    with np.load(root / (name + ".npz"), allow_pickle=False) as z:
        return z["vectors"], z["labels"]


def rank(out, reference):
    query = unit(out[0].reshape(1, -1))[0]
    vectors, labels = reference
    ids = np.argsort(-(vectors @ query))[:5]
    return {"top": labels[ids].tolist(), "projection": query.tolist()}
