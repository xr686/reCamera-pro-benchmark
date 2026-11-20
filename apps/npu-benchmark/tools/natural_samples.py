"""Build fixed real-video tracking pairs and labelled environmental audio fixtures."""

import argparse
import csv
import json
import math
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ESC = "https://raw.githubusercontent.com/karolpiczak/ESC-50/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/"
TRACK = "https://raw.githubusercontent.com/opencv/opencv_extra/ebf9366b3845efe1e54bf2440cf238fd68905bab/testdata/cv/tracking/"
# Single primary ESC label mapped to the corresponding AudioSet class index.
LABELS = {
    "rain": 283,
    "helicopter": 333,
    "chainsaw": 341,
    "toilet_flush": 368,
    "vacuum_cleaner": 371,
    "clapping": 58,
    "siren": 390,
    "church_bells": 196,
}


def fetch(cache, name, url):
    path = cache / name
    if not path.is_file():
        with urlopen(url, timeout=60) as response:
            data = response.read()
        path.write_bytes(data)
    return path


def audio(cache, fixtures):
    rows = list(
        csv.DictReader(
            fetch(cache, "esc50.csv", ESC + "meta/esc50.csv").read_text().splitlines()
        )
    )
    records = []
    for category, label in LABELS.items():
        choices = sorted(
            r["filename"]
            for r in rows
            if r["category"] == category and r["fold"] == "5"
        )
        if len(choices) < 2:
            raise ValueError(category)
        for filename in choices[:2]:
            rate, pcm = wavfile.read(fetch(cache, filename, ESC + "audio/" + filename))
            d = math.gcd(rate, 16000)
            signal = resample_poly(
                pcm.astype(np.float32) / 32768, 16000 // d, rate // d
            )
            name = "esc-" + Path(filename).stem + ".npy"
            # Use the central three seconds of each five-second labelled clip.
            signal = signal[(len(signal) - 48000) // 2 :][:48000].copy()
            np.save(fixtures / name, signal)
            records.append(
                dict(
                    file=name,
                    seconds=len(signal) / 16000,
                    label=label,
                    category=category,
                )
            )
    return records


def crop(frame, cx, cy, side, output):
    left, top = round(cx - side / 2), round(cy - side / 2)
    transform = np.float32(
        [
            [output / side, 0, -left * output / side],
            [0, output / side, -top * output / side],
        ]
    )
    return cv2.warpAffine(frame, transform, (output, output)), left, top


def tracking(cache, fixtures):
    records = []
    for seq, first in (("david", 300), ("dudek", 1), ("faceocc2", 1)):
        gt = np.loadtxt(
            fetch(cache, seq + "-gt.txt", TRACK + seq + "/gt.txt"), delimiter=","
        )
        path = fetch(cache, seq + ".webm", TRACK + seq + "/data/" + seq + ".webm")
        camera = cv2.VideoCapture(str(path))
        try:
            for i in (0, 37, 82, 135, 204, 265):
                x, y, w, h = gt[i]
                cx, cy = x + w / 2, y + h / 2
                outputs = []
                for step, ratio, output in ((0, 2, 128), (5, 4, 256)):
                    camera.set(cv2.CAP_PROP_POS_FRAMES, first - 1 + i + step)
                    success, frame = camera.read()
                    if not success:
                        raise ValueError(f"Missing frame {seq} {i + step}")
                    side = max(1, round(ratio * math.sqrt(w * h)))
                    img, left, top = crop(frame, cx, cy, side, output)
                    name = f"track/{seq}-{i}-{step}.png"
                    if not cv2.imwrite(str(fixtures / name), img):
                        raise OSError(name)
                    outputs.append(name)
                gx, gy, gw, gh = gt[i + 5]
                rect = (
                    (np.array([gx, gy, gx + gw, gy + gh]) - [left, top, left, top])
                    / side
                ).tolist()
                records.append(
                    dict(file=outputs[1], template=outputs[0], box=rect, sequence=seq)
                )
        finally:
            camera.release()
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    index = args.fixtures / "groups.json"
    data = json.loads(index.read_text())
    data["sound"] = audio(args.cache, args.fixtures)
    data["track"] = tracking(args.cache, args.fixtures)
    index.write_text(json.dumps(data, ensure_ascii=False))
