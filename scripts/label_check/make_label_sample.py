"""Pick about 100 test sentences for the label check, whole answers from different passages.

About a third are unsupported on purpose. At the 6% base rate, 100 random sentences hold about six.
"""

import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from trustrag.data import load_claims    # noqa: E402

TARGET = 100
df = load_claims("test", task_type="QA")
answers = []
for rid, g in df.groupby("response_id"):
    if 4 <= len(g) <= 9:
        answers.append((rid, g.sort_values("claim_idx")))

rng = random.Random(42)
rng.shuffle(answers)
# answers with the most errors first, two for every clean one
dirty = sorted((a for a in answers if a[1]["label"].any()), key=lambda a: -a[1]["label"].sum())
clean = [a for a in answers if not a[1]["label"].any()]
order = [x for i in range(len(clean)) for x in (dirty[2 * i:2 * i + 2] + [clean[i]])]

picked, seen, n = [], set(), 0
for pool in order:
    rid, g = pool
    src = g["source_id"].iloc[0]
    if src in seen:
        continue
    seen.add(src)
    picked.append(pool)
    n += len(g)
    if n >= TARGET:
        break

items = []
for order, (rid, g) in enumerate(picked, 1):
    items.append({
        "id": "a%02d" % order, "order": order, "response_id": str(rid),
        "question": g["query"].iloc[0], "passage": g["context"].iloc[0],
        "claims": [{"id": "%s_%d" % (rid, k), "text": t, "ragtruth_label": int(lab),
                    "label_types": lt}
                   for k, t, lab, lt in zip(g["claim_idx"], g["claim"], g["label"], g["label_types"])],
    })

out = os.path.join(ROOT, "results", "label_check")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "sample.json"), "w", encoding="utf-8") as f:
    json.dump(items, f, indent=1, ensure_ascii=False)
claims = [c for it in items for c in it["claims"]]
print("%d answers, %d claims, %d unsupported by RAGTruth (%.0f%%), passage chars max %d"
      % (len(items), len(claims), sum(c["ragtruth_label"] for c in claims),
         100 * sum(c["ragtruth_label"] for c in claims) / len(claims),
         max(len(it["passage"]) for it in items)))
