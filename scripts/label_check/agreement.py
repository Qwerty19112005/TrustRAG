"""Compare RAGTruth's labels with the LLM baseline, and score each method against both.

An LLM labelled the 108 sentences under written guidelines, see llm_labels.json.
Intervals resample whole answers. Writes results/label_check/agreement.json.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from trustrag import evaluate as E                  # noqa: E402
from trustrag.data import load_claims               # noqa: E402
from trustrag.lexical import overlap_unsupported    # noqa: E402

HC = os.path.join(ROOT, "results", "label_check")
N_BOOT = 2000
SEED = 42

with open(os.path.join(HC, "sample.json"), encoding="utf-8") as f:
    sample = json.load(f)
rows = [{"answer": it["id"], "claim_id": c["id"], "text": c["text"], "ragtruth": int(c["ragtruth_label"]),
         "types": c["label_types"]} for it in sample for c in it["claims"]]
df = pd.DataFrame(rows)
with open(os.path.join(HC, "llm_labels.json"), encoding="utf-8") as f:
    llm = json.load(f)["labels"]
df["llm"] = df["claim_id"].map(lambda k: int(llm[k] == "unsupported"))

pred = pd.read_parquet(os.path.join(ROOT, "results", "qa", "test_predictions.parquet"))
pred["claim_id"] = pred["response_id"].astype(str) + "_" + pred["claim_idx"].astype(str)
df = df.merge(pred[["claim_id", "score_v1", "score_v2", "score_trees", "nli_support", "p_unsupported"]],
              on="claim_id", how="left")

claims = load_claims("test", task_type="QA")
claims["claim_id"] = claims["response_id"].astype(str) + "_" + claims["claim_idx"].astype(str)
part = df[["claim_id"]].merge(claims[["claim_id", "claim", "context"]], on="claim_id", how="left")
df["overlap"] = overlap_unsupported(part)

METHODS = [("Word overlap", df["overlap"].values), ("NLI, best sentence", -df["nli_support"].values),
           ("TrustRAG v1 (6 features)", df["score_v1"].values), ("TrustRAG v2 (12 features)", df["score_v2"].values),
           ("TrustRAG v2, boosted trees", df["score_trees"].values)]
index = {a: np.flatnonzero(df["answer"].values == a) for a in df["answer"].unique()}
groups = list(index)
rng = np.random.default_rng(SEED)
picks = [np.concatenate([index[groups[i]] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(N_BOOT)]


def ci(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


detector = []
for src in ("llm", "ragtruth"):
    y = df[src].values
    for name, s in METHODS:
        boot = [E.detection(y[p], s[p])["auroc"] for p in picks if 0 < y[p].sum() < len(p)]
        detector.append({"labels": src, "method": name, **E.detection(y, s), "auroc_ci": ci(boot)})

# same resamples for both label sets, so the gap is paired
trees = df["score_trees"].values
gap = [E.detection(df["llm"].values[p], trees[p])["auroc"] - E.detection(df["ragtruth"].values[p], trees[p])["auroc"]
       for p in picks if 0 < df["llm"].values[p].sum() < len(p) and 0 < df["ragtruth"].values[p].sum() < len(p)]

x, y = df["ragtruth"].values, df["llm"].values
tp, fp, fn = int(((x == 1) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
kappa_boot = [cohen_kappa_score(x[p], y[p]) for p in picks]

with open(os.path.join(ROOT, "results", "qa", "metrics.json"), encoding="utf-8") as f:
    t_flag = json.load(f)["selective"]["t_flag"]
p = df["p_unsupported"].values
actions = {}
for name, m in (("show", p <= t_flag), ("flag", (p > t_flag) & (p < 0.5)), ("hold", p >= 0.5)):
    actions[name] = {"sentences": int(m.sum()), "llm_unsupported": int(df["llm"].values[m].sum()),
                     "ragtruth_unsupported": int(df["ragtruth"].values[m].sum())}

# how many of these sentences are also in the judge sample
with open(os.path.join(ROOT, "results", "judges", "sample.json")) as f:
    judged = set(json.load(f))
overlap_judges = sum(len(it["claims"]) for it in sample if str(it["response_id"]) in judged)

report = {
    "sample": {"answers": len(sample), "sentences": len(df), "llm_unsupported": int(y.sum()),
               "ragtruth_unsupported": int(x.sum())},
    "pairs": [{"a": "llm", "b": "ragtruth", "n": len(df), "agreement": float((x == y).mean()),
               "kappa": float(cohen_kappa_score(x, y)), "kappa_ci": ci(kappa_boot)}],
    "ragtruth_against_baseline": {"both_unsupported": tp, "only_ragtruth": fp, "only_baseline": fn,
                                  "precision": tp / (tp + fp), "recall": tp / (tp + fn)},
    "detector": detector,
    "trees_gap": {"auroc_llm_minus_ragtruth": float(np.mean(gap)), "ci": ci(gap)},
    "actions": actions,
    "judge_sentences_on_sample": overlap_judges,
    "disagreements": [{"claim_id": r["claim_id"], "text": r["text"], "ragtruth": int(r["ragtruth"]), "llm": int(r["llm"]),
                       "ragtruth_types": r["types"]} for _, r in df[df["llm"] != df["ragtruth"]].iterrows()],
}
with open(os.path.join(HC, "agreement.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

P = report["pairs"][0]
print("RAGTruth vs LLM baseline: n=%d  agreement %.2f  kappa %.2f [%.2f, %.2f]" % (P["n"], P["agreement"], P["kappa"], *P["kappa_ci"]))
print("RAGTruth against baseline:", report["ragtruth_against_baseline"])
for d in detector:
    print("%-8s %-28s AUROC %.3f [%.2f, %.2f]  AUPRC %.3f" % (d["labels"], d["method"], d["auroc"], *d["auroc_ci"], d["auprc"]))
print("trees, LLM minus RAGTruth labels:", report["trees_gap"])
print("actions:", actions)
print("judge sentences on this sample:", overlap_judges)
