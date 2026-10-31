"""Word-matching features, and the word overlap baseline. No models here except spaCy."""

import re

import numpy as np

from trustrag.features import _spacy, split_sentences

LEXICAL_NAMES = ["Word overlap (best sentence)", "Novel content words", "Unseen numbers",
                 "Unseen named entities", "Position in answer", "Claims in answer"]

_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_STOP = set("a an the of to in on at by for with from and or but is are was were be been "
            "it its this that these those as has have had which who what".split())


def content_words(text):
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def lexical_features(df):
    """df needs claim and context. response_id and claim_idx are used when present."""
    ctx = {}
    for c in df["context"].unique():
        low = c.lower()
        ctx[c] = (low, set(_WORD.findall(low)), [set(_WORD.findall(s.lower())) for s in split_sentences(c)])

    if "response_id" in df:
        n_claims = df.groupby("response_id")["claim_idx"].transform("max").values + 1
        pos = df["claim_idx"].values / np.maximum(n_claims - 1, 1)
    else:
        n_claims, pos = np.ones(len(df)), np.zeros(len(df))

    docs = _spacy().pipe(df["claim"].tolist(), batch_size=256)
    X = np.zeros((len(df), 6), dtype=np.float32)
    for i, (claim, c, doc) in enumerate(zip(df["claim"], df["context"], docs)):
        low, vocab, sents = ctx[c]
        words = content_words(claim)
        if words:
            X[i, 0] = max(len(words & s) / len(words) for s in sents)
            X[i, 1] = len(words - vocab) / len(words)
        X[i, 2] = sum(n not in low for n in _NUM.findall(claim))
        X[i, 3] = sum(e.text.lower() not in low for e in doc.ents)
    X[:, 4] = pos
    X[:, 5] = n_claims
    return X


def overlap_unsupported(df):
    """Word overlap baseline: 1 minus the best share of content words found in one sentence."""
    sents = {c: [set(_WORD.findall(x.lower())) for x in split_sentences(c)] for c in df["context"].unique()}
    out = []
    for claim, c in zip(df["claim"], df["context"]):
        words = content_words(claim)
        out.append(1.0 - max(len(words & x) / len(words) for x in sents[c]) if words else 0.0)
    return np.array(out)
