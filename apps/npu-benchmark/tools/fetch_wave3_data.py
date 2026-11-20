"""Fetch the public source assets for the extended benchmark and record digests."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse, hashlib, json, subprocess

FILES = {
    "sms.zip": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
    "airfoil.zip": "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip",
    "wine.zip": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    "energy.zip": "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
    "tfeat.pth": "https://raw.githubusercontent.com/vbalnt/tfeat/master/pretrained-models/tfeat-liberty.params",
    "hardnet.pth": "https://raw.githubusercontent.com/DagnyT/hardnet/master/pretrained/train_liberty_with_aug/checkpoint_liberty_with_aug.pth",
    "sosnet.pth": "https://raw.githubusercontent.com/yuruntian/SOSNet/master/sosnet-weights/sosnet_32x32_liberty.pth",
}
for split in ("train", "t10k"):
    for kind in ("images-idx3", "labels-idx1"):
        filename = f"{split}-{kind}-ubyte.gz"
        FILES[filename] = (
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
            + filename
        )


for scene in ("cones", "teddy"):
    for view in ("im2", "im6", "disp2"):
        FILES[f"{scene}-{view}.png"] = f"https://vision.middlebury.edu/stereo/data/scenes2003/newdata/{scene}/{view}.png"


def fetch(work):
    root = work / "data"
    root.mkdir(parents=True, exist_ok=True)

    def one(k):
        path = root / k
        if not path.exists():
            tmp = path.with_suffix(".part")
            subprocess.run(
                ["curl", "-fsSL", "--retry", "3", FILES[k], "-o", str(tmp)], check=True
            )
            tmp.replace(path)
        return k, dict(
            url=FILES[k], sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )

    with ThreadPoolExecutor(5) as pool:
        assets = dict(pool.map(one, FILES))
    (work / "source-assets.json").write_text(json.dumps(assets, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    fetch(p.parse_args().work)
