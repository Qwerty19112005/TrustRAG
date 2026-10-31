"""HotpotQA demo: retrieve paragraphs, generate an answer, check every sentence.

    python scripts/demo/run_demo.py --smoke            # 3 questions, quick check
    python scripts/demo/run_demo.py                    # 100 questions, writes results/demo/demo.json
    python scripts/demo/run_demo.py --ask "Who is Arun Date, the Marathi singer?"

The first full run embeds all 66,581 paragraphs, about 12 minutes on our GPU, then keeps them in data/cache.
HotpotQA has no sentence labels, so we can only check how risk lines up with wrong answers and retrieval.
"""

import argparse
import json
import os
import string
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from trustrag import evaluate as E                      # noqa: E402
from trustrag import features as F                      # noqa: E402
from trustrag import rag                                # noqa: E402
from trustrag.detector import T_HOLD, Detector, check   # noqa: E402

SEED = 42
HOTPOT = os.path.join(ROOT, "data", "hotpotqa", "distractor_validation.parquet")
OUT = os.path.join(ROOT, "results", "demo")
PUNCT = set(string.punctuation)


def corpus(only_ids=None):
    """Every paragraph of the HotpotQA dev set, one per title."""
    h = pd.read_parquet(HOTPOT)
    if only_ids is not None:
        h = h[h["id"].isin(only_ids)]
    paras = {}
    for ctx in h["context"]:
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            paras.setdefault(title, " ".join(s.strip() for s in sents))
    titles = list(paras)
    return titles, ["%s: %s" % (t, paras[t]) for t in titles]


def corpus_vectors(texts, tag):
    path = os.path.join(F.CACHE, "hotpot_corpus_%s.npy" % tag)
    if os.path.exists(path):
        V = np.load(path)
        if len(V) == len(texts):
            return V
    os.makedirs(F.CACHE, exist_ok=True)
    t0 = time.time()
    V = F.embed(texts, batch_size=128)
    np.save(path, V)
    print("  embedded %s paragraphs in %.0fs" % (format(len(texts), ","), time.time() - t0), flush=True)
    return V


def norm(s):
    s = s.lower().replace("'s ", " ").replace("’s ", " ")
    s = "".join(" " if ch in PUNCT else ch for ch in s)
    return [w for w in s.split() if w not in ("a", "an", "the")]


def outcome(answer, gold):
    """Gold answers are short and ours are sentences, so we match on the gold's words."""
    if "unable to answer" in answer.lower():
        return "refused"
    a, g = set(norm(answer)), norm(gold)
    if g in (["yes"], ["no"]):
        other = "no" if g == ["yes"] else "yes"
        return "correct" if g[0] in a and other not in a else "wrong"
    return "correct" if sum(w in a for w in g) >= 0.8 * len(g) else "wrong"


def show(rec):
    tag = {"show": "  show", "flag": "  FLAG", "hold": "  HOLD"}
    print("\nQ: %s" % rec["question"])
    if rec.get("gold"):
        print("Gold answer: %s   (%s)" % (rec["gold"], rec["outcome"]))
    print("Retrieved: %s" % " | ".join(rec["retrieved"]))
    for c in rec["claims"]:
        print("%s %.2f  %s" % (tag[c["action"]], c["p"], c["claim"]))
        if c["action"] != "show":
            print("             closest support: %s" % c["evidence"][:160])


def ask(question):
    titles, texts = corpus()
    V = corpus_vectors(texts, "dev")
    idx = rag.retrieve([question], V)[0]
    passages = [texts[j] for j in idx]
    answer = rag.Generator()(rag.prompt_for(question, passages))
    claims, _ = check(answer, passages, Detector())
    show({"question": question, "retrieved": [titles[j] for j in idx], "claims": claims})


def evaluate(n, smoke):
    h = pd.read_parquet(HOTPOT).sample(n, random_state=SEED)
    titles, texts = corpus(set(h["id"]) if smoke else None)
    print("== corpus: %s paragraphs" % format(len(titles), ","), flush=True)
    V = corpus_vectors(texts, "smoke" if smoke else "dev")
    F.embed(["warm-up"])     # keeps model loading out of the timing
    t0 = time.time()
    top = rag.retrieve(h["question"].tolist(), V)
    retrieve_ms = 1000 * (time.time() - t0) / len(h)

    os.makedirs(OUT, exist_ok=True)
    cache_path = os.path.join(OUT, "answers%s.json" % ("_smoke" if smoke else ""))
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    missing = [i for i, qid in enumerate(h["id"]) if qid not in cache]
    print("== generating %d answers (%d saved)" % (len(missing), len(h) - len(missing)), flush=True)
    if missing:
        gen = rag.Generator()
        for k, i in enumerate(missing):
            r = h.iloc[i]
            t1 = time.time()
            ans = gen(rag.prompt_for(r["question"], [texts[j] for j in top[i]]))
            cache[r["id"]] = {"answer": ans, "ms": 1000 * (time.time() - t1)}
            if (k + 1) % 10 == 0 or k + 1 == len(missing):
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False)
                print("  %d/%d" % (k + 1, len(missing)), flush=True)
        gen.close()

    print("== checking", flush=True)
    detector = Detector()
    recs = []
    for i in range(len(h)):
        r = h.iloc[i]
        passages = [texts[j] for j in top[i]]
        ans = cache[r["id"]]["answer"]
        claims, ms = check(ans, passages, detector)
        gold_titles = sorted(set(r["supporting_facts"]["title"]))
        got = [titles[j] for j in top[i]]
        recs.append({"id": r["id"], "question": r["question"], "gold": r["answer"], "type": r["type"],
                     "level": r["level"], "gold_titles": gold_titles, "retrieved": got,
                     "gold_retrieved": sum(t in got for t in gold_titles), "answer": ans, "claims": claims,
                     "outcome": outcome(ans, r["answer"]), "generate_ms": cache[r["id"]]["ms"],
                     "check_ms": ms})
    summary = summarise(recs, detector.t_flag, len(titles), retrieve_ms)
    path = os.path.join(OUT, "demo%s.json" % ("_smoke" if smoke else ""))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**summary, "records": recs}, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1))
    for rec in recs[:3]:
        show(rec)
    print("\nSaved:", path)


def summarise(recs, t_flag, n_paragraphs, retrieve_ms):
    claims = [c for r in recs for c in r["claims"]]
    risk = np.array([max(c["p"] for c in r["claims"]) if r["claims"] else 0.0 for r in recs])
    raw = np.array([max(c["raw"] for c in r["claims"]) if r["claims"] else 0.0 for r in recs])
    flagged = np.array([any(c["action"] != "show" for c in r["claims"]) for r in recs])
    oc = np.array([r["outcome"] for r in recs])
    both = np.array([r["gold_retrieved"] == len(r["gold_titles"]) for r in recs])
    answered = oc != "refused"

    def group(mask):
        return {"answers": int(mask.sum()), "mean_max_risk": float(risk[mask].mean()) if mask.any() else None,
                "any_flag": float(flagged[mask].mean()) if mask.any() else None}

    wrong = (oc == "wrong")[answered].astype(int)
    return {
        "generator": rag.GENERATOR.split("/")[-1], "retriever": "bge-small-en-v1.5, top %d" % rag.TOP_K,
        "corpus_paragraphs": n_paragraphs, "questions": len(recs), "claims": len(claims),
        "claims_per_answer": len(claims) / len(recs), "t_flag": t_flag, "t_hold": T_HOLD,
        "actions": {a: float(np.mean([c["action"] == a for c in claims])) for a in ("show", "flag", "hold")},
        "retrieval": {"both_gold": float(both.mean()),
                      "any_gold": float(np.mean([r["gold_retrieved"] > 0 for r in recs]))},
        "outcomes": {o: float((oc == o).mean()) for o in ("correct", "wrong", "refused")},
        "by_outcome": {o: group(oc == o) for o in ("correct", "wrong", "refused")},
        "by_retrieval": {"both_gold": group(both), "missing_gold": group(~both)},
        "outcome_by_gold_retrieved": {str(k): {o: int(sum(r["outcome"] == o and r["gold_retrieved"] == k
                                                         for r in recs)) for o in ("correct", "wrong", "refused")}
                                      for k in (0, 1, 2)},
        "auroc_wrong_vs_correct": (E.detection(wrong, raw[answered])["auroc"]
                                   if 0 < wrong.sum() < len(wrong) else None),
        "latency_ms": {"retrieve": retrieve_ms,
                       "generate": float(np.median([r["generate_ms"] for r in recs])),
                       "check": float(np.median([r["check_ms"] for r in recs]))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="how many dev questions to run")
    ap.add_argument("--smoke", action="store_true", help="3 questions against their own paragraphs")
    ap.add_argument("--ask", help="answer and check one question")
    args = ap.parse_args()
    if args.ask:
        ask(args.ask)
    else:
        evaluate(3 if args.smoke else args.n, args.smoke)


if __name__ == "__main__":
    main()
