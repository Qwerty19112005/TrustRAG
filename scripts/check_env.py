"""Check packages, GPU and data files. Pass --models to also load the verifier."""

import argparse
import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

OK, WARN, BAD = "  ok  ", " warn ", " MISS "

CORE = ["transformers", "tokenizers", "pyarrow", "numpy", "sklearn", "torch"]
LATER = ["spacy", "matplotlib", "pandas", "datasets",
         "sentence_transformers", "faiss", "gradio"]

FILES = [
    ("data/ragtruth/train.parquet", 15090),
    ("data/ragtruth/test.parquet", 2700),
    ("data/halueval/qa.parquet", 10000),
    ("data/hotpotqa/distractor_validation.parquet", 7405),
]

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def line(status, label, detail=""):
    print("[%s] %-34s %s" % (status, label, detail))


def check_python():
    v = sys.version_info
    detail = "%d.%d.%d" % (v.major, v.minor, v.micro)
    line(OK if v >= (3, 9) else BAD, "python", detail)


def check_packages(names, required):
    missing = []
    for n in names:
        try:
            m = importlib.import_module(n)
            line(OK, n, getattr(m, "__version__", ""))
        except Exception:
            missing.append(n)
            line(BAD if required else WARN, n, "not installed")
    return missing


def check_torch():
    try:
        import torch
    except Exception:
        line(BAD, "torch", "not installed")
        return
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        line(OK, "GPU", "%s, %.1f GB" % (name, gb))
        if gb < 5:
            line(WARN, "GPU memory", "under 5 GB, use batch_size 4 or lower")
    else:
        line(WARN, "GPU", "CUDA not available, everything will run on CPU "
                          "(NLI scoring will be roughly 20x slower)")


def check_data():
    try:
        import pyarrow.parquet as pq
    except Exception:
        line(WARN, "data files", "install pyarrow to verify row counts")
        return
    for rel, expect in FILES:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            line(BAD, rel, "missing, run: python download_data.py")
            continue
        try:
            n = pq.ParquetFile(path).metadata.num_rows
            good = (n == expect)
            line(OK if good else WARN, rel,
                 "%s rows%s" % (format(n, ","), "" if good else
                                " (expected %s)" % format(expect, ",")))
        except Exception as e:
            line(BAD, rel, "unreadable: %s" % type(e).__name__)


def check_models():
    print("\n--- loading the NLI model (first run downloads ~1.7 GB) ---")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        line(BAD, "transformers/torch", str(e)[:60])
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        model.to(dev).eval()
        line(OK, "NLI model", "on %s, %d labels" % (dev, len(model.config.id2label)))
        print("       id2label:", model.config.id2label)

        premise = "Arthur's Magazine was started in 1844."
        pairs = [
            ("Arthur's Magazine was founded in 1844.", "entailment"),
            ("Arthur's Magazine was founded in 1901.", "contradiction"),
            ("Arthur's Magazine was printed in Philadelphia.", "neutral"),
        ]
        with torch.no_grad():
            for hyp, want in pairs:
                x = tok(premise, hyp, return_tensors="pt", truncation=True).to(dev)
                p = torch.softmax(model(**x).logits[0], -1)
                top = model.config.id2label[int(p.argmax())]
                mark = "ok" if top.lower() == want else "UNEXPECTED (wanted %s)" % want
                print("       %-46s -> %-14s %s" % (hyp[:46], top, mark))

        # throughput, so a full RAGTruth pass can be sized before starting it
        import time
        batch = [(premise, pairs[0][0])] * 128
        for bs in (8, 16, 32):
            try:
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    for i in range(0, len(batch), bs):
                        chunk = batch[i:i + bs]
                        x = tok([a for a, _ in chunk], [b for _, b in chunk],
                                return_tensors="pt", padding=True,
                                truncation=True, max_length=256).to(dev)
                        model(**x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                rate = len(batch) / (time.time() - t0)
                print("       batch %-3d -> %7.1f pairs/sec" % (bs, rate))
            except RuntimeError as e:
                print("       batch %-3d -> failed (%s)" % (bs, str(e)[:40]))
                break
        if dev == "cuda":
            print("       peak VRAM: %.2f GB of %.1f GB"
                  % (torch.cuda.max_memory_allocated() / 1e9,
                     torch.cuda.get_device_properties(0).total_memory / 1e9))
        print("       a full RAGTruth pass is roughly 1.2M pairs "
              "(15,090 responses x ~8 claims x ~10 evidence sentences)")
    except Exception as e:
        line(BAD, "NLI model", "%s: %s" % (type(e).__name__, str(e)[:70]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true",
                    help="also download and smoke-test the NLI model")
    args = ap.parse_args()

    print("=" * 62)
    print("TrustRAG environment check")
    print("=" * 62)
    check_python()
    print("\n--- packages needed now ---")
    missing_core = check_packages(CORE, required=True)
    check_torch()
    print("\n--- packages needed later ---")
    check_packages(LATER, required=False)
    print("\n--- data ---")
    check_data()

    if args.models:
        check_models()

    print("\n" + "=" * 62)
    if missing_core:
        print("NOT READY. Install: pip install " + " ".join(
            {"sklearn": "scikit-learn"}.get(m, m) for m in missing_core))
    else:
        print("Core stack is ready.")
    print("=" * 62)


if __name__ == "__main__":
    main()
