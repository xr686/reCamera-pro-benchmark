"""Select source-separated ESC-50 recordings for speech-presence negatives."""

from pathlib import Path
import argparse, concurrent.futures, csv, hashlib, json, subprocess


def get(work, esc_index):
    with esc_index.open() as f:
        rows = list(csv.DictReader(f))
    chosen = []
    for fold in (1, 2, 3, 4, 5):
        candidates = [
            r
            for r in rows
            if int(r["fold"]) == fold and int(r["target"]) not in range(20, 30)
        ]
        seen = set()
        for row in sorted(
            candidates, key=lambda v: hashlib.sha256(v["filename"].encode()).digest()
        ):
            if row["src_file"] in seen:
                continue
            seen.add(row["src_file"])
            chosen.append(row)
            if len(seen) == (32 if fold == 5 else 24):
                break
    base = work / "data/esc"
    base.mkdir(exist_ok=True)

    def one(row):
        name = row["filename"]
        p = base / name
        src = (
            "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/" + name
        )
        if not p.exists():
            tmp = p.with_suffix(".part")
            subprocess.run(
                [
                    "curl",
                    "-fsSL",
                    "--retry",
                    "3",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "10",
                    src,
                    "-o",
                    str(tmp),
                ],
                check=True,
            )
            tmp.replace(p)
        return dict(**row, sha256=hashlib.sha256(p.read_bytes()).hexdigest(), url=src)

    with concurrent.futures.ThreadPoolExecutor(10) as pool:
        files = list(pool.map(one, chosen))
    (work / "data/esc-sources.json").write_text(json.dumps(files, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("work", type=Path)
    p.add_argument("esc_index", type=Path)
    a = p.parse_args()
    get(a.work, a.esc_index)
