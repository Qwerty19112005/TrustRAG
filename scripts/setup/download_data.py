"""Download the datasets into data/. Files already there are skipped.

    python scripts/setup/download_data.py
"""

import json
import os
import sys
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

# dataset, config, split, destination under ./data
WANT = [
    ("wandb/RAGTruth-processed", "default",    "train",      "ragtruth/train.parquet"),
    ("wandb/RAGTruth-processed", "default",    "test",       "ragtruth/test.parquet"),
    ("pminervini/HaluEval",      "qa",         "data",       "halueval/qa.parquet"),
    ("hotpotqa/hotpot_qa",       "distractor", "validation", "hotpotqa/distractor_validation.parquet"),
]

API = "https://datasets-server.huggingface.co/parquet?dataset="


def listing(dataset):
    url = API + dataset.replace("/", "%2F")
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["parquet_files"]


def fetch(url, dest, expected):
    if os.path.exists(dest) and abs(os.path.getsize(dest) - expected) < 1024:
        print("    skip (already present): %s" % os.path.basename(dest))
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    done = 0
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            sys.stdout.write("\r    downloading %s ... %6.1f MB" %
                             (os.path.basename(dest), done / 1e6))
            sys.stdout.flush()
    os.replace(tmp, dest)
    print("\r    saved %s (%.1f MB)%s" % (os.path.basename(dest), done / 1e6, " " * 12))


def main():
    cache = {}
    for dataset, config, split, rel in WANT:
        if dataset not in cache:
            cache[dataset] = listing(dataset)
        hits = [f for f in cache[dataset]
                if f["config"] == config and f["split"] == split]
        if not hits:
            print("!! not found: %s %s/%s" % (dataset, config, split))
            continue
        print("%s  [%s/%s]" % (dataset, config, split))
        if len(hits) == 1:
            fetch(hits[0]["url"], os.path.join(ROOT, rel), hits[0].get("size", 0))
        else:
            base, ext = os.path.splitext(rel)
            for i, f in enumerate(hits):
                fetch(f["url"], os.path.join(ROOT, "%s_%04d%s" % (base, i, ext)),
                      f.get("size", 0))
    print("\nAll files under: %s" % ROOT)


if __name__ == "__main__":
    main()
