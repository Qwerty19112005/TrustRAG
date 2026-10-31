"""Load RAGTruth and turn its annotated spans into sentence labels."""

import hashlib
import json
import os
import random

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "ragtruth")
CACHE = os.path.join(ROOT, "data", "cache")

SEED = 42
MIN_OVERLAP_CHARS = 3
_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm",
                          disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])
    return _NLP


def _source_id(context):
    # six models answer each passage, so we split on the passage
    return hashlib.md5((context or "").encode("utf-8")).hexdigest()[:16]


def _parse_spans(raw):
    if not raw:
        return []
    try:
        spans = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return spans if isinstance(spans, list) else []


def load_claims(split="train", task_type=None, limit=None, use_cache=True):
    """One row per sentence."""
    tag = "%s_%s" % (split, task_type or "all")
    cache_path = os.path.join(CACHE, "claims_%s.parquet" % tag)
    if use_cache and limit is None and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    src = os.path.join(DATA, "%s.parquet" % split)
    t = pq.ParquetFile(src).read().to_pydict()
    n = len(t["id"])

    order = list(range(n))
    random.Random(SEED).shuffle(order)
    if task_type:
        order = [i for i in order if t["task_type"][i] == task_type]
    if limit:
        order = order[:limit]

    texts = [t["output"][i] or "" for i in order]
    rows = []
    for i, doc in zip(order, _nlp().pipe(texts, batch_size=64)):
        out = t["output"][i] or ""
        spans = _parse_spans(t["hallucination_labels"][i])

        flag = bytearray(len(out))
        types = {}
        for s in spans:
            a = max(0, int(s.get("start", 0)))
            b = min(len(out), int(s.get("end", 0)))
            for j in range(a, b):
                flag[j] = 1
            types[(a, b)] = s.get("label_type", "")

        for k, sent in enumerate(doc.sents):
            a, b = sent.start_char, sent.end_char
            length = b - a
            if length <= 0:
                continue
            # one made-up clause makes the whole sentence unsupported
            hit = sum(flag[a:b])
            label = int(hit >= MIN_OVERLAP_CHARS)
            kinds = sorted({lt for (sa, sb), lt in types.items()
                            if sa < b and sb > a and lt})
            rows.append({
                "response_id": t["id"][i],
                "source_id": _source_id(t["context"][i]),
                "task_type": t["task_type"][i],
                "model": t["model"][i],
                "claim_idx": k,
                "claim": " ".join(sent.text.split()),
                "context": t["context"][i],
                "query": t["query"][i],
                "label": label,
                "overlap_frac": hit / length,
                "label_types": "|".join(kinds),
            })

    df = pd.DataFrame(rows)
    if limit is None:
        os.makedirs(CACHE, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def make_splits(df, val_frac=0.15, seed=SEED):
    """Split by source passage so no evidence leaks between splits."""
    per_source = (df.groupby("source_id")["label"]
                    .mean().gt(0).astype(int).rename("has_pos").reset_index())
    rng = pd.Series(range(len(per_source))).sample(frac=1, random_state=seed).values
    per_source = per_source.iloc[rng].reset_index(drop=True)

    val_ids = []
    for _, grp in per_source.groupby("has_pos"):
        k = max(1, int(round(len(grp) * val_frac)))
        val_ids.extend(grp["source_id"].tolist()[:k])
    val_ids = set(val_ids)

    val = df[df["source_id"].isin(val_ids)].reset_index(drop=True)
    train = df[~df["source_id"].isin(val_ids)].reset_index(drop=True)
    return train, val


def describe(df, name="dataset"):
    n, pos = len(df), int(df["label"].sum())
    print("%-22s %7s claims  %6s positive (%.1f%%)  ratio 1:%.1f  %s sources"
          % (name, format(n, ","), format(pos, ","), 100 * pos / n,
             (n - pos) / max(pos, 1), format(df["source_id"].nunique(), ",")))
