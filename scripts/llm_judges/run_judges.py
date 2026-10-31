"""LLM judges on a fixed sample of about 1,000 test sentences, through the Groq API.

    set GROQ_API_KEY first (console.groq.com)
    python scripts/llm_judges/run_judges.py --list-models
    python scripts/llm_judges/run_judges.py

Every response is saved in results/judges/cache, so a rerun only asks for what is missing.
"""

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from trustrag import evaluate as E          # noqa: E402
from trustrag.data import load_claims       # noqa: E402

API = "https://api.groq.com/openai/v1"
# the API rejects urllib's default User-Agent
UA = "trustrag/1.0"
OUT = os.path.join(ROOT, "results", "judges")
SAMPLE_CLAIMS = 1000
# Groq no longer serves Llama 3, so we use this large and small pair
JUDGES = {"large": "openai/gpt-oss-120b", "small": "openai/gpt-oss-20b"}

PROMPT = """You check answers written by a retrieval-augmented assistant.

Passages the assistant was given:
<<<
{passage}
>>>

Question: {question}

The assistant's answer, one numbered sentence per line:
{claims}

For each sentence decide whether the passages support it. A sentence is UNSUPPORTED if any
part of it adds information the passages do not contain (a number, name, date, reason or
step) or contradicts them. Judge only against the passages, not your own knowledge. Ignore
which passage a sentence cites. Sentences with no factual content count as supported.

Reply with JSON only, in this shape:
{{"sentences": [{{"n": 1, "supported": true, "p_unsupported": 5}}, ...]}}
with one entry per numbered sentence, where p_unsupported is your probability (0-100)
that the sentence is unsupported."""


def make_sample():
    """Whole answers, drawn at random until we have SAMPLE_CLAIMS sentences."""
    path = os.path.join(OUT, "sample.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    df = load_claims("test", task_type="QA")
    rids = sorted(df["response_id"].unique())
    random.Random(42).shuffle(rids)
    picked, n = [], 0
    for rid in rids:
        picked.append(str(rid))
        n += int((df["response_id"] == rid).sum())
        if n >= SAMPLE_CLAIMS:
            break
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(picked, f)
    return picked


def call(model, prompt, key, retries=10, json_mode=True):
    payload = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(API + "/chat/completions", data=body, method="POST",
                                 headers={"Authorization": "Bearer " + key, "User-Agent": UA,
                                          "Content-Type": "application/json"})
    for attempt in range(retries):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
            out["_latency_s"] = time.time() - t0
            return out
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                wait = float(e.headers.get("retry-after") or 2 ** attempt * 5)
                print("    %d from API, waiting %.0fs" % (e.code, wait), flush=True)
                time.sleep(wait)
                continue
            msg = e.read().decode()
            if e.code == 400 and "json_validate_failed" in msg and json_mode:
                # the reply failed Groq's JSON check, so ask again as plain text
                print("    JSON mode failed, retrying as plain text", flush=True)
                out = call(model, prompt, key, retries, json_mode=False)
                out["_json_fallback"] = True
                return out
            raise SystemExit("API error %d: %s" % (e.code, msg[:300]))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            # dropped connection: wait and try again
            if attempt < retries - 1:
                wait = min(60, 5 * 2 ** attempt)
                print("    network error (%s), waiting %ds" % (getattr(e, "reason", e), wait), flush=True)
                time.sleep(wait)
                continue
            raise
    raise SystemExit("gave up after %d retries" % retries)


def parse(content, n):
    """P(unsupported) per sentence. A sentence the judge skipped gets NaN."""
    m = re.search(r"\{.*\}", content or "", re.S)
    try:
        rows = json.loads(m.group(0)).get("sentences", []) if m else []
    except (ValueError, AttributeError):
        rows = []
    p = np.full(n, np.nan)
    for r in rows:
        try:
            k = int(r["n"]) - 1
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= k < n:
            v = r.get("p_unsupported")
            if isinstance(v, (int, float)):
                p[k] = min(max(v / 100.0, 0.0), 1.0)
            elif isinstance(r.get("supported"), bool):
                p[k] = 0.0 if r["supported"] else 1.0
    return p


def judge(name, model, answers, key):
    cache = os.path.join(OUT, "cache", model)
    os.makedirs(cache, exist_ok=True)
    rows = []
    for i, (rid, g) in enumerate(answers):
        path = os.path.join(cache, "%s.json" % rid)
        if os.path.exists(path):
            with open(path) as f:
                out = json.load(f)
        else:
            claims = "\n".join("%d. %s" % (k + 1, c) for k, c in enumerate(g["claim"]))
            out = call(model, PROMPT.format(passage=g["context"].iloc[0], question=g["query"].iloc[0],
                                            claims=claims), key)
            with open(path, "w") as f:
                json.dump(out, f)
        p = parse(out["choices"][0]["message"]["content"], len(g))
        u = out.get("usage", {})
        for k, (cid, lab) in enumerate(zip(g["claim_idx"], g["label"])):
            rows.append({"response_id": rid, "claim_idx": cid, "label": int(lab), "p": p[k]})
        if (i + 1) % 10 == 0:
            print("  %-6s %d/%d answers" % (name, i + 1, len(answers)), flush=True)
        rows[-1]["_latency_s"] = out.get("_latency_s")
        rows[-1]["_json_fallback"] = bool(out.get("_json_fallback"))
        rows[-1]["_in"], rows[-1]["_out"] = u.get("prompt_tokens"), u.get("completion_tokens")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--only", choices=list(JUDGES), help="run one judge")
    args = ap.parse_args()
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Set GROQ_API_KEY first (console.groq.com -> API Keys).")
    if args.list_models:
        req = urllib.request.Request(API + "/models", headers={"Authorization": "Bearer " + key, "User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            for m in sorted(x["id"] for x in json.load(r)["data"]):
                print(m)
        return

    rids = set(make_sample())
    df = load_claims("test", task_type="QA")
    df = df[df["response_id"].astype(str).isin(rids)]
    answers = [(str(rid), g.sort_values("claim_idx")) for rid, g in df.groupby("response_id")]
    print("sample: %d answers, %d claims, %d unsupported"
          % (len(answers), len(df), int(df["label"].sum())), flush=True)

    results = {}
    for name, model in JUDGES.items():
        if args.only and name != args.only:
            continue
        print("== judge %s (%s)" % (name, model), flush=True)
        results[name] = (model, judge(name, model, answers, key))

    pred = pd.read_parquet(os.path.join(ROOT, "results", "qa", "test_predictions.parquet"))
    pred["response_id"] = pred["response_id"].astype(str)
    ours = df.assign(response_id=df["response_id"].astype(str))[["response_id", "claim_idx", "label"]].merge(
        pred[["response_id", "claim_idx", "score_v2", "p_unsupported"] +
             (["score_trees"] if "score_trees" in pred else [])], on=["response_id", "claim_idx"])
    y = ours["label"].values
    with open(os.path.join(ROOT, "results", "qa", "metrics.json")) as f:
        thr = json.load(f)["trustrag"]["threshold"]      # chosen on the calibration split
    report = {"sample": {"answers": len(answers), "claims": len(df), "unsupported": int(df["label"].sum())},
              "methods": []}
    report["methods"].append({"name": "TrustRAG v2 (12 features)", **E.detection(y, ours["score_v2"].values),
                              **E.prf_at(y, ours["p_unsupported"].values, thr)})
    if "score_trees" in ours:
        report["methods"].append({"name": "TrustRAG v2, boosted trees",
                                  **E.detection(y, ours["score_trees"].values)})
    for name, (model, r) in results.items():
        ok = r["p"].notna().values
        rr = r[ok]
        lat = r["_latency_s"].dropna()
        # TrustRAG on exactly the sentences this judge answered
        same = rr.assign(response_id=rr["response_id"].astype(str))[["response_id", "claim_idx"]].merge(
            ours, on=["response_id", "claim_idx"])
        report["methods"].append({
            "name": "%s judge" % model, "model": model, "parsed": float(ok.mean()),
            "json_fallbacks": int(r["_json_fallback"].fillna(False).sum()),
            "trustrag_auroc_same_claims": E.detection(same["label"].values, same["score_v2"].values)["auroc"],
            **E.detection(rr["label"].values, rr["p"].values),
            **E.prf_at(rr["label"].values, rr["p"].values, 0.5),
            "s_per_answer": float(lat.mean()) if len(lat) else None,
            "tokens_in_per_answer": float(r["_in"].dropna().mean()) if r["_in"].notna().any() else None,
            "tokens_out_per_answer": float(r["_out"].dropna().mean()) if r["_out"].notna().any() else None,
        })
    with open(os.path.join(OUT, "judges.json"), "w") as f:
        json.dump(report, f, indent=2)
    for mth in report["methods"]:
        print("  %-36s AUROC %.3f  AUPRC %.3f  F1 %.3f" % (mth["name"], mth["auroc"], mth["auprc"],
                                                         mth.get("f1", float("nan"))))
    print("wrote", os.path.join(OUT, "judges.json"))


if __name__ == "__main__":
    main()
