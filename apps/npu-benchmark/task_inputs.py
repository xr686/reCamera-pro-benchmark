from functools import lru_cache
import cv2
import numpy as np


def read_sample(root, record):
    path = root / record["file"]
    if path.suffix == ".txt":
        return path.read_text().strip(), root
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read {path}")
    if "hole" in record:
        return (image, cv2.imread(str(root / record["hole"]), cv2.IMREAD_GRAYSCALE))
    if "template" in record:
        return (cv2.imread(str(root / record["template"])), image)
    return image


@lru_cache(maxsize=1)
def mel_filters():
    hz = np.arange(257) * 16000 / 512
    mel = 1127 * np.log1p(hz / 700)
    edges = np.linspace(1127 * np.log1p(20 / 700), 1127 * np.log1p(8000 / 700), 82)
    left = (mel[:, None] - edges[:-2]) / (edges[1:-1] - edges[:-2])
    right = (edges[2:] - mel[:, None]) / (edges[2:] - edges[1:-1])
    return np.maximum(0, np.minimum(left, right)).astype(np.float32)


def fbank(sample):
    # Kaldi: 25 ms Povey windows, 10 ms shift, reflected edges and utterance CMVN.
    pcm = sample[:96000] * 32768
    count = max(1, (len(pcm) + 80) // 160)
    framed = np.lib.stride_tricks.sliding_window_view(
        np.pad(pcm, (120, 280), mode="symmetric"), 400
    )[::160][:count].copy()
    framed -= framed.mean(1, keepdims=True)
    framed[:, 1:] -= 0.97 * framed[:, :-1].copy()
    framed[:, 0] *= 0.03
    framed *= np.hanning(400).astype(np.float32) ** 0.85
    power = np.abs(np.fft.rfft(framed, 512)) ** 2
    feat = np.log(np.maximum(power @ mel_filters(), np.finfo(np.float32).eps))
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-5)
    result = np.zeros((1, 80, 600), np.float32)
    result[0, :, : len(feat)] = feat.T
    return result


def prepare(model, sample, metadata=None):
    fmt = model.pixels
    if fmt == "short-wave":
        return [np.ascontiguousarray(sample, dtype=np.float32).reshape(1, 1, 1, 1024)]
    if fmt in ("hashed-text", "tabular", "bayer4", "std-patch", "stereo-pair"):
        from extended_tasks import prepare_extended
        return prepare_extended(model, sample, metadata)
    if fmt in ("cliptext", "clipimage"):
        from clip_benchmark import prepare_clip

        return prepare_clip(model, sample)
    if fmt in ("signal", "logmel", "gray01", "cropnet", "saliency", "line4"):
        from extra_tasks import prepare_extra

        return prepare_extra(model, sample, metadata)
    if fmt == "inpaint":
        image, mask = sample
        return [
            cv2.dnn.blobFromImage(image, 1 / 255, model.size),
            (cv2.resize(mask, model.size, interpolation=cv2.INTER_NEAREST) > 0).astype(
                np.float32
            )[None, None],
        ]
    if fmt == "fbank":
        return [fbank(sample)]
    if fmt in ("pcm", "wave_norm"):
        value = np.zeros(model.size[0], np.float32)
        count = min(len(value), len(sample))
        wave = sample[:count]
        value[:count] = (
            (wave - wave.mean()) / np.sqrt(wave.var() + 1e-7)
            if fmt == "wave_norm"
            else wave
        )
        return [value[None]]
    if fmt == "track":
        # OpenCV TrackerVit uses Image2BlobParams with its default BGR ordering.
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        return [
            cv2.dnn.blobFromImage((img.astype(np.float32) / 255 - mean) / std)
            for img in sample
        ]
    image = cv2.resize(sample, model.size, interpolation=cv2.INTER_LINEAR)
    if fmt in ("gray11", "y01"):
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if fmt == "gray11"
            else cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        )
        value = gray.astype(np.float32)
        value = (value / 127.5 - 1) if fmt == "gray11" else value / 255
        return [value[None, None]]
    if fmt not in ("bgr", "bgr8", "bgr11", "db", "edge"):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if fmt not in ("rgb8", "bgr8"):
        image = image.astype(np.float32)
    if fmt in ("rgb11", "bgr11"):
        image = image / 127.5 - 1
    elif fmt == "rgb01":
        image = image / 255
    elif fmt in ("imagenet", "db", "nanodet"):
        mu = np.array([123.675, 116.28, 103.53], np.float32)
        std = np.array([58.395, 57.12, 57.375], np.float32)
        if fmt == "nanodet":
            mu, std = mu[::-1], std[::-1]
        image = (image - mu) / std
    elif fmt == "edge":
        image -= np.array([103.5, 116.2, 123.6], np.float32)
    elif fmt == "retina":
        image -= np.array([104, 117, 123], np.float32)
    if model.layout == "nchw":
        image = image.transpose(2, 0, 1)
    return [np.ascontiguousarray(image[None])]


def fingerprint(outputs):
    rows = []
    for tensor in outputs:
        x = np.asarray(tensor)
        v = x.ravel()
        indexes = np.linspace(0, len(v) - 1, min(512, len(v)), dtype=int)
        rows.append(
            {"shape": list(x.shape), "values": v[indexes].astype(float).tolist()}
        )
    return rows
