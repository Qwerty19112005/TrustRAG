"""Check packages, GPU and data files. Add --models to also load the NLI model.

    python scripts/setup/check_env.py
"""

import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

PACKAGES = ["torch", "transformers", "sentencepiece", "numpy", "pandas", "pyarrow", "sklearn", "spacy", "matplotlib"]
DATA = [
    ("data/ragtruth/train.parquet", 15090),
    ("data/ragtruth/test.parquet", 2700),
    ("data/halueval/qa.parquet", 10000),
    ("data/hotpotqa/distractor_validation.parquet", 7405),
]


def line(ok, label, detail=""):
    print("[%s] %-46s %s" % (" ok " if ok else "MISS", label, detail))
    return ok


def check_packages():
    ok = line(sys.version_info >= (3, 9), "python", "%d.%d.%d" % sys.version_info[:3])
    for name in PACKAGES:
        try:
            ok &= line(True, name, getattr(importlib.import_module(name), "__version__", ""))
        except ImportError:
            ok = line(False, name, "not installed")
    try:
        import spacy
        spacy.load("en_core_web_sm")
        line(True, "spaCy model en_core_web_sm")
    except (ImportError, OSError):
        ok = line(False, "spaCy model en_core_web_sm", "run: python -m spacy download en_core_web_sm")
    return ok


def check_gpu():
    import torch
    if torch.cuda.is_available():
        line(True, "GPU", "%s, %.1f GB" % (torch.cuda.get_device_name(0),
                                          torch.cuda.get_device_properties(0).total_memory / 1e9))
    else:
        print("[note] no GPU found, everything still runs but the NLI step is much slower")


def check_data():
    import pyarrow.parquet as pq
    ok = True
    for rel, rows in DATA:
        path = os.path.join(ROOT, rel)
        if os.path.exists(path):
            n = pq.ParquetFile(path).metadata.num_rows
            ok &= line(n == rows, rel, "%s rows" % format(n, ","))
        else:
            ok = line(False, rel, "run: python scripts/setup/download_data.py")
    return ok


def check_model():
    from trustrag import features as F
    print("\nLoading the NLI model (about 1.7 GB the first time)")
    P = F.nli_scores([("Arthur's Magazine was started in 1844.", "Arthur's Magazine was founded in 1844."),
                      ("Arthur's Magazine was started in 1844.", "Arthur's Magazine was founded in 1901.")])
    ok = P[0, 0] > 0.5 and P[1, 2] > 0.5
    return line(ok, "NLI model", "entailment %.2f, contradiction %.2f" % (P[0, 0], P[1, 2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true", help="also load the NLI model")
    args = ap.parse_args()
    ok = check_packages()
    if ok:
        check_gpu()
        ok &= check_data()
        if args.models:
            ok &= check_model()
    print("\nAll good." if ok else "\nSomething is missing, see above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
