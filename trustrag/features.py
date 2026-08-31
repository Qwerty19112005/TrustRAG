"""The six features: three from NLI, one from embeddings, two from the claim text."""

import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
EMB_MODEL = "BAAI/bge-small-en-v1.5"
FEATURE_NAMES = ["f1_support", "f2_peak_contra", "f3_mean_top3_ent",
                 "f4_max_cosine", "f5_specificity", "f6_claim_len"]

_state = {}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _nli():
    if "nli" not in _state:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        m = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(_device()).eval()
        if _device() == "cuda":
            m = m.half()
        # reorder to [entail, neutral, contra]; the config order is not guaranteed
        lab = {v.lower(): k for k, v in m.config.id2label.items()}
        _state["nli"] = (tok, m, [lab["entailment"], lab["neutral"], lab["contradiction"]])
    return _state["nli"]


def _emb():
    if "emb" not in _state:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(EMB_MODEL)
        m = AutoModel.from_pretrained(EMB_MODEL).to(_device()).eval()
        _state["emb"] = (tok, m)
    return _state["emb"]


def _spacy():
    if "nlp" not in _state:
        import spacy
        _state["nlp"] = spacy.load("en_core_web_sm", disable=["lemmatizer", "attribute_ruler"])
    return _state["nlp"]


def split_sentences(text, max_chars=9000, min_len=20, cap=20):
    doc = _spacy()(text[:max_chars])
    out = [" ".join(s.text.split()) for s in doc.sents if len(s.text.strip()) >= min_len]
    return out[:cap] if out else [" ".join(text[:400].split())]


def nli_scores(pairs, batch_size=64, max_length=256):
    """[(premise, hypothesis)] -> array [n, 3] of P(entail), P(neutral), P(contra)."""
    tok, model, order = _nli()
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            ch = pairs[i:i + batch_size]
            x = tok([p for p, _ in ch], [h for _, h in ch], return_tensors="pt",
                    padding=True, truncation=True, max_length=max_length).to(_device())
            p = torch.softmax(model(**x).logits.float(), -1).cpu().numpy()
            out.append(p[:, order])
    return np.vstack(out) if out else np.zeros((0, 3))


def embed(texts, batch_size=64, max_length=256):
    """CLS pooling then L2 normalise, so a dot product gives cosine similarity."""
    tok, model = _emb()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            x = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length).to(_device())
            h = model(**x).last_hidden_state[:, 0]
            vecs.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
    return np.vstack(vecs)


def extract(df, verbose=True):
    """df needs columns: claim, context, label. Returns X [n,6], y [n]."""
    contexts = list(dict.fromkeys(df["context"].tolist()))
    sents = {c: split_sentences(c) for c in contexts}
    if verbose:
        print("  %d unique contexts, %.1f evidence sentences each"
              % (len(contexts), np.mean([len(s) for s in sents.values()])))

    ev_all = sorted({s for v in sents.values() for s in v})
    ev_vec = dict(zip(ev_all, embed(ev_all)))
    claims = df["claim"].tolist()
    cl_vec = embed(claims)

    docs = list(_spacy().pipe(claims, batch_size=64))

    # flatten to one list so the GPU gets full batches; scoring per claim is ~5x slower
    rows = df.reset_index(drop=True)
    pairs, bounds = [], []
    for _, row in rows.iterrows():
        ev = sents[row["context"]]
        bounds.append((len(pairs), len(pairs) + len(ev)))
        pairs.extend((e, row["claim"]) for e in ev)
    if verbose:
        print("  %s claim-evidence pairs to score" % format(len(pairs), ","))

    P = nli_scores(pairs, batch_size=128)
    if verbose:
        print("  NLI done")

    X = np.zeros((len(rows), 6), dtype=np.float32)
    for i, (_, row) in enumerate(rows.iterrows()):
        a, b = bounds[i]
        ent, con = P[a:b, 0], P[a:b, 2]
        ev = sents[row["context"]]

        X[i, 0] = float((ent - con).max())
        X[i, 1] = float(con.max())
        X[i, 2] = float(np.sort(ent)[-3:].mean())
        X[i, 3] = float(np.max(np.stack([ev_vec[e] for e in ev]) @ cl_vec[i]))
        d = docs[i]
        X[i, 4] = len(d.ents) + sum(t.like_num for t in d)
        X[i, 5] = len(d)

    return X, rows["label"].values.astype(int)


def cached_extract(df, tag, verbose=True):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "features_%s.npz" % tag)
    if os.path.exists(path):
        z = np.load(path)
        return z["X"], z["y"]
    X, y = extract(df, verbose=verbose)
    np.savez_compressed(path, X=X, y=y)
    return X, y
