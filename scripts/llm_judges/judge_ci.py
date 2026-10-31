"""Bootstrap intervals for the judge comparison. Needs no API key, it reads the saved answers.

    python scripts/llm_judges/judge_ci.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import run_judges as RJ                     # noqa: E402
from trustrag import evaluate as E          # noqa: E402
from trustrag.data import load_claims       # noqa: E402

N_BOOT = 2000
SEED = 42

rids = set(RJ.make_sample())
df = load_claims("test", task_type="QA")
df = df[df["response_id"].astype(str).isin(rids)].copy()
df["response_id"] = df["response_id"].astype(str)
pred = pd.read_parquet(os.path.join(ROOT, "results", "qa", "test_predictions.parquet"))
pred["response_id"] = pred["response_id"].astype(str)
t = df[["response_id", "claim_idx", "label"]].merge(
    pred[["response_id", "claim_idx", "score_v2", "score_trees"]], on=["response_id", "claim_idx"])

names = {"TrustRAG v2": "score_v2", "Boosted trees": "score_trees"}
for model in RJ.JUDGES.values():
    rows = []
    for path in glob.glob(os.path.join(RJ.OUT, "cache", model, "*.json")):
        rid = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            out = json.load(f)
        g = df[df["response_id"] == rid].sort_values("claim_idx")
        p = RJ.parse(out["choices"][0]["message"]["content"], len(g))
        rows += [{"response_id": rid, "claim_idx": k, model: v} for k, v in zip(g["claim_idx"], p)]
    t = t.merge(pd.DataFrame(rows), on=["response_id", "claim_idx"], how="left")
    names[model.split("/")[-1]] = model

# a sentence a judge skipped counts as supported
scores = {k: t[c].fillna(0.0).values for k, c in names.items()}
y = t["label"].values
index = {rid: np.flatnonzero(t["response_id"].values == rid) for rid in t["response_id"].unique()}
groups = list(index)

rng = np.random.default_rng(SEED)
draws = {k: {"auroc": [], "auprc": []} for k in names}
for _ in range(N_BOOT):
    pick = np.concatenate([index[groups[i]] for i in rng.integers(0, len(groups), len(groups))])
    if y[pick].sum() == 0:
        continue
    for k, s in scores.items():
        d = E.detection(y[pick], s[pick])
        draws[k]["auroc"].append(d["auroc"])
        draws[k]["auprc"].append(d["auprc"])


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


report = {"answers": len(groups), "claims": len(y), "unsupported": int(y.sum()), "n_boot": N_BOOT, "methods": {},
          "differences": []}
for k, s in scores.items():
    d = E.detection(y, s)
    report["methods"][k] = {"auroc": d["auroc"], "auroc_ci": ci(draws[k]["auroc"]),
                            "auprc": d["auprc"], "auprc_ci": ci(draws[k]["auprc"])}
judge = "gpt-oss-120b"
for k in ("TrustRAG v2", "Boosted trees", "gpt-oss-20b"):
    for metric in ("auroc", "auprc"):
        diff = np.array(draws[judge][metric]) - np.array(draws[k][metric])
        report["differences"].append({"a": judge, "b": k, "metric": metric, "diff": float(diff.mean()), "ci": ci(diff),
                                      "ci_excludes_zero": bool(ci(diff)[0] > 0 or ci(diff)[1] < 0)})
with open(os.path.join(RJ.OUT, "bootstrap.json"), "w") as f:
    json.dump(report, f, indent=2)
for k, m in report["methods"].items():
    print("%-15s AUROC %.3f [%.3f, %.3f]   AUPRC %.3f [%.3f, %.3f]"
          % (k, m["auroc"], *m["auroc_ci"], m["auprc"], *m["auprc_ci"]))
for d in report["differences"]:
    print("%s minus %-14s %s %+.3f [%+.3f, %+.3f]%s"
          % (d["a"], d["b"], d["metric"], d["diff"], *d["ci"], "  excludes 0" if d["ci_excludes_zero"] else ""))
