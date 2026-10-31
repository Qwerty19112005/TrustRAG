"""Summary AUROC split by whether the article fits the verifier's 512-token window.

    python scripts/experiments/length_split.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from trustrag import evaluate as E     # noqa: E402
from trustrag import features as F     # noqa: E402
from trustrag.data import load_claims  # noqa: E402

WINDOW = 512
OUT = os.path.join(ROOT, "results", "summary")

pred = pd.read_parquet(os.path.join(OUT, "test_predictions.parquet"))
claims = load_claims("test", task_type="Summary")
whole = pred["whole_passage"].values

# test_predictions has no passage text, so look it up
ctx = claims.set_index(["response_id", "claim_idx"])["context"]
ctx = [ctx[(r, k)] for r, k in zip(pred["response_id"], pred["claim_idx"])]
tok = AutoTokenizer.from_pretrained(F.NLI_MODEL)
n_tok = {c: len(tok(c)["input_ids"]) for c in set(ctx)}
length = np.array([n_tok[c] for c in ctx])

y = pred["label"].values
scores = {"NLI, whole passage": whole, "NLI, best sentence": -pred["nli_support"].values,
          "TrustRAG v2 (12 features)": pred["score_v2"].values,
          "TrustRAG v2, boosted trees": pred["score_trees"].values}
groups = []
for name, m in (("fits", length <= WINDOW), ("truncated", length > WINDOW)):
    groups.append({"group": name, "claims": int(m.sum()), "unsupported": int(y[m].sum()),
                   "passages": len({c for c, k in zip(ctx, m) if k}),
                   "auroc": {s: E.detection(y[m], v[m])["auroc"] for s, v in scores.items()}})

passages = np.array(list(n_tok.values()))
report = {"window": WINDOW, "passage_tokens_median": float(np.median(passages)),
          "passage_tokens_p90": float(np.percentile(passages, 90)),
          "share_passages_truncated": float((passages > WINDOW).mean()), "groups": groups}
with open(os.path.join(OUT, "by_length.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("passages: median %d tokens, %.0f%% over %d" % (report["passage_tokens_median"],
                                                     100 * report["share_passages_truncated"], WINDOW))
for g in groups:
    print("%-10s %5d claims  " % (g["group"], g["claims"]) +
          "  ".join("%s %.3f" % (s, a) for s, a in g["auroc"].items()))
