"""Time one verifier three times and store the median in its metrics.json.

    python scripts/experiments/bench_verifiers.py large
    python scripts/experiments/bench_verifiers.py base

Run them back to back on an idle GPU so the times can be compared.
"""

import json
import os
import sys

import run_experiment as R
from trustrag import features as F
from trustrag.data import load_claims

which = sys.argv[1]
if which == "base":
    F.NLI_MODEL = F.NLI_MODEL_BASE
runs = sorted((R.benchmark(load_claims("test", task_type="QA")) for _ in range(3)),
              key=lambda c: c["ms_per_answer"])
cost = dict(runs[1], repeats=[round(c["ms_per_answer"]) for c in runs])
path = os.path.join(R.ROOT, "results", "qa" if which == "large" else "qa_base", "metrics.json")
with open(path) as f:
    m = json.load(f)
m["cost"] = cost
with open(path, "w") as f:
    json.dump(m, f, indent=2)
print(which, "median %.0f ms per answer, runs %s, peak %.2f GB"
      % (cost["ms_per_answer"], cost["repeats"], cost["peak_gpu_mb"] / 1024))
