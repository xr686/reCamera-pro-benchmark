"""Download the public datasets used in the fourth RV1126B benchmark pack."""

import argparse, concurrent.futures, hashlib, json, subprocess
from pathlib import Path

MID = "https://vision.middlebury.edu/flow/data/comp/zip/"
UCI = "https://archive.ics.uci.edu/static/public/"
GTS = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
SOURCES = {
    "flow.zip": MID + "other-color-twoframes.zip",
    "flow-gt.zip": MID + "other-gt-flow.zip",
    "eurosat.zip": "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
    "flowers.tgz": "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    "gts-train.zip": GTS + "GTSRB-Training_fixed.zip",
    "gts-test.zip": GTS + "GTSRB_Final_Test_Images.zip",
    "gts-labels.zip": GTS + "GTSRB_Final_Test_GT.zip",
    "wifi.zip": UCI + "422/wireless+indoor+localization.zip",
    "grid.zip": UCI + "471/electrical+grid+stability+simulated+data.zip",
    "seeds.zip": UCI + "236/seeds.zip",
    "raisins.zip": UCI + "850/raisin.zip",
    "sentiment.zip": UCI + "331/sentiment+labelled+sentences.zip",
    "trec-train.txt": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
    "trec-test.txt": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
}


def fetch(work):
    def one(item):
        name, url = item
        p = work / "data" / name
        try:
            if not p.exists():
                tmp = p.with_suffix(".part")
                subprocess.run(
                    [
                        "curl",
                        "-fsSL",
                        "--connect-timeout",
                        "12",
                        "--max-time",
                        "1800",
                        "--retry",
                        "2",
                        url,
                        "-o",
                        str(tmp),
                    ],
                    check=True,
                )
                tmp.replace(p)
            print(name, p.stat().st_size, flush=True)
            with p.open("rb") as f:
                return name, dict(
                    url=url, sha256=hashlib.file_digest(f, "sha256").hexdigest()
                )
        except Exception as e:
            print(name, str(e), flush=True)
            return name, None

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        out = dict(pool.map(one, SOURCES.items()))
    (work / "source-assets.json").write_text(
        json.dumps({k: v for k, v in out.items() if v}, indent=2)
    )


    if any(v is None for v in out.values()):
        raise RuntimeError("Some datasets could not be fetched; rerun to resume verified files")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    fetch(p.parse_args().work)
