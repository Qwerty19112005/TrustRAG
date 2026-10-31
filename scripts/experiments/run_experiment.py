"""The main experiment: features, detector, calibration, baselines and the HaluEval test.

    python scripts/experiments/run_experiment.py --smoke            # quick check, a few minutes
    python scripts/experiments/run_experiment.py                    # RAGTruth QA
    python scripts/experiments/run_experiment.py --verifier base    # QA with the 0.18B verifier
    python scripts/experiments/run_experiment.py --task Summary     # news summaries

Features are cached per chunk of passages, so an interrupted run picks up where it stopped.
Writes results/<run>/: metrics.json, detector.json, test_predictions.parquet and the figures.
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from trustrag import evaluate as E                              # noqa: E402
from trustrag import features as F                              # noqa: E402
from trustrag.data import describe, load_claims, make_splits    # noqa: E402
from trustrag.detector import (ALL, V1, fit_lr, fit_trees, proba,    # noqa: E402
                               save_detector)
from trustrag.lexical import (LEXICAL_NAMES, lexical_features,       # noqa: E402
                              overlap_unsupported)

SEED = 42
TYPES = ["Evident Baseless Info", "Subtle Baseless Info", "Evident Conflict", "Subtle Conflict"]
FEATURE_LABELS = ["Support score", "Peak contradiction", "Mean top-3 entailment",
                  "Embedding similarity", "Claim specificity", "Claim length"] + LEXICAL_NAMES
RISK_TARGET = 0.05     # share of unsupported claims allowed among those shown


# ------------------------------------------------------------------ data and features

def sample_sources(df, n):
    if not n:
        return df
    ids = df["source_id"].drop_duplicates().sample(frac=1, random_state=SEED)
    return df[df["source_id"].isin(set(ids[:n]))].reset_index(drop=True)


def score_in_chunks(df, tag, chunk=40):
    ids = df["source_id"].drop_duplicates().tolist()
    n = (len(ids) + chunk - 1) // chunk
    parts, Xs, t0 = [], [], time.time()
    for k in range(n):
        part = df[df["source_id"].isin(set(ids[k * chunk:(k + 1) * chunk]))].reset_index(drop=True)
        X, _ = F.cached_extract(part, tag="%s_%03d" % (tag, k), verbose=False)
        parts.append(part)
        Xs.append(X)
        print("  %-14s chunk %3d/%d  %5d claims  %6.0fs" % (tag, k + 1, n, len(part), time.time() - t0),
              flush=True)
    return pd.concat(parts, ignore_index=True), np.vstack(Xs)


def load_halueval(n, matched=False):
    """One correct and one hallucinated answer per row. matched keeps rows of similar length."""
    t = pq.ParquetFile(os.path.join(ROOT, "data", "halueval", "qa.parquet")).read().to_pandas()
    if matched:
        gap = (t["right_answer"].str.split().str.len() - t["hallucinated_answer"].str.split().str.len()).abs()
        t = t[gap <= 2]
    t = t.sample(n, random_state=SEED) if n < len(t) else t
    rows = []
    for i, r in t.iterrows():
        # the knowledge field glues sentences together ("century.First")
        ctx = re.sub(r"\.(?=[A-Z])", ". ", r["knowledge"])
        # answers are often bare names, so the question makes them a claim
        for ans, lab in ((r["right_answer"], 0), (r["hallucinated_answer"], 1)):
            rows.append({"source_id": "he%05d" % i, "context": ctx, "label": lab,
                         "claim": "%s %s" % (r["question"].strip(), str(ans).strip())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ baselines

def whole_passage_unsupported(df, tag):
    path = os.path.join(F.CACHE, "wholepassage_%s.npy" % tag)
    if os.path.exists(path):
        return np.load(path)
    P = F.nli_scores(list(zip(df["context"], df["claim"])), batch_size=16, max_length=512)
    s = P[:, 2] - P[:, 0]
    np.save(path, s)
    return s


# ------------------------------------------------------------------ detector

def grouped_cv(X, y, groups, cols, fitter=fit_lr):
    aucs = [E.detection(y[te], proba(fitter(X[tr], y[tr], cols), X[te], cols))["auroc"]
            for tr, te in GroupKFold(5).split(X, y, groups)]
    return float(np.mean(aucs)), float(np.std(aucs))


def cached_lexical(df, tag):
    path = os.path.join(F.CACHE, "lexical_%s.npy" % tag)
    if os.path.exists(path):
        return np.load(path)
    L = lexical_features(df)
    np.save(path, L)
    return L


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


def benchmark(df, n_resp=30):
    """Time the whole feature pass on fresh answers."""
    import torch
    ids = df["response_id"].drop_duplicates().iloc[:n_resp]
    part = df[df["response_id"].isin(set(ids))].reset_index(drop=True)
    F.extract(part.head(20), verbose=False)   # warm-up
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    F.extract(part, verbose=False)
    if cuda:
        torch.cuda.synchronize()
    dt = time.time() - t0
    pairs = sum(len(F.split_sentences(c)) for c in part["context"])
    return {"device": torch.cuda.get_device_name(0) if cuda else "cpu",
            "answers": len(ids), "claims": len(part), "pairs": pairs,
            "ms_per_answer": 1000 * dt / len(ids), "ms_per_claim": 1000 * dt / len(part),
            "pairs_per_sec": pairs / dt,
            "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else None}


# ------------------------------------------------------------------ figures

INK, MUTED, GRID = "#1A2332", "#6B7A8C", "#E3E8EE"
BLUE, AMBER = "#0F6FC5", "#D68910"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Calibri", "Segoe UI", "DejaVu Sans"], "font.size": 11,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight"})
    return plt


def figures(out, m, curves, rel):
    plt = _style()

    # detection
    names = [b["name"] for b in m["baselines"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, key, chance, label in ((axes[0], "auroc", 0.5, "dashed = random, 0.50"),
                                   (axes[1], "auprc", m["test"]["base_rate"],
                                    "dashed = base rate, %.2f" % m["test"]["base_rate"])):
        vals = [b[key] for b in m["baselines"]]
        y = np.arange(len(vals))
        cols = [BLUE if n.startswith("TrustRAG") else MUTED for n in names]
        ax.hlines(y, chance, vals, color=GRID, linewidth=2, zorder=1)
        ax.scatter(vals, y, s=70, color=cols, zorder=3, edgecolor="white", linewidth=1.5)
        for v, yy in zip(vals, y):
            ax.text(v + 0.012, yy, "%.3f" % v, va="center", fontsize=10, color=INK)
        ax.axvline(chance, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlim(min(chance, min(vals)) - 0.03, max(vals) + 0.09)
        ax.set_title(key.upper(), loc="left")
        ax.set_title(label, loc="right", fontsize=9, fontweight="normal", color=MUTED)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(names)), names)
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_detection.png"))
    plt.close(fig)

    # reliability
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.62, 0.55, "perfect calibration", color=MUTED, fontsize=9, rotation=40)
    for key, col, lab in (("raw", AMBER, "Logistic regression"), ("isotonic", BLUE, "+ isotonic")):
        px, py, _ = zip(*rel[key])
        ax.plot(px, py, color=col, linewidth=2, marker="o", markersize=6,
                markeredgecolor="white", label="%s  (ECE %.3f)" % (lab, m["calibration"][key]["ece"]))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability unsupported")
    ax.set_ylabel("Observed share unsupported")
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_reliability.png"))
    plt.close(fig)

    # risk-coverage
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key, col, lab in (("trustrag", BLUE, "TrustRAG"), ("nli_only", AMBER, "NLI support only")):
        cov, risk = curves[key]
        ax.plot(100 * cov, 100 * risk, color=col, linewidth=2)
    base = 100 * m["test"]["base_rate"]
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1)
    ax.text(2, base + 0.3, "show everything: %.1f%% unsupported" % base, color=MUTED, fontsize=9)
    r90 = 100 * m["selective"]["risk_at_90"]
    ax.scatter([90], [r90], s=60, color=BLUE, zorder=4, edgecolor="white", linewidth=1.5)
    ax.annotate("%.1f%% unsupported at 90%% coverage" % r90, (90, r90), xytext=(62, base * 1.3),
                textcoords="data", ha="center", fontsize=9.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    lines = [plt.Line2D([], [], color=BLUE, linewidth=2), plt.Line2D([], [], color=AMBER, linewidth=2)]
    ax.legend(lines, ["TrustRAG v2 (12 features)", "NLI support score only"],
              loc="upper left", fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, max(base * 1.6, 1))
    ax.set_xlabel("Coverage: % of claims shown, most trusted first")
    ax.set_ylabel("% of shown claims unsupported")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_risk_coverage.png"))
    plt.close(fig)

    # error types
    rows = [r for r in m["per_type"] if r["n"] > 0]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(rows))
    for key, col, off, lab in (("auroc", BLUE, -0.12, "TrustRAG"),
                               ("auroc_nli_only", AMBER, 0.12, "NLI support only")):
        ax.scatter([r[key] for r in rows], y + off, s=60, color=col, label=lab, zorder=3,
                   edgecolor="white", linewidth=1.5)
    for r, yy in zip(rows, y):
        ax.text(r["auroc"] + 0.015, yy - 0.12, "%.2f" % r["auroc"], va="center", fontsize=9.5)
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set_yticks(y, ["%s  (n=%d)" % (r["type"], r["n"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("AUROC against all supported claims")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_per_type.png"))
    plt.close(fig)

    # feature weights
    w = m["feature_weights"]
    order = np.argsort([abs(x["weight"]) for x in w])[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    vals = [w[i]["weight"] for i in order]
    ax.barh(np.arange(len(vals)), vals, color=[AMBER if v > 0 else BLUE for v in vals], height=0.6)
    for yy, v in enumerate(vals):
        ax.text(v + (0.03 if v > 0 else -0.03), yy, "%+.2f" % v, va="center",
                ha="left" if v > 0 else "right", fontsize=9.5)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(np.arange(len(vals)), [w[i]["feature"] for i in order])
    ax.invert_yaxis()
    lim = max(abs(v) for v in vals) * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Standardised weight   (← supported  |  unsupported →)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_weights.png"))
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--task", default="QA", help="RAGTruth task_type, or 'all'")
    ap.add_argument("--train-sources", type=int, default=0, help="cap on training passages, 0 = all")
    ap.add_argument("--halueval", type=int, default=1000, help="HaluEval rows, two answers each")
    ap.add_argument("--verifier", choices=["large", "base"], default="large",
                    help="large is 0.4B, base is 0.18B")
    ap.add_argument("--keep-cost", action="store_true",
                    help="reuse the timing already in metrics.json")
    args = ap.parse_args()

    if args.verifier == "base":
        F.NLI_MODEL = F.NLI_MODEL_BASE
    if args.task != "QA":
        F.MAX_SENTENCES = 60     # news articles are longer than QA passages
    lex = "smoke" if args.smoke else args.task.lower()
    run = "smoke" if args.smoke else lex + ("" if args.verifier == "large" else "_base")
    out = os.path.join(ROOT, "results", run)
    os.makedirs(out, exist_ok=True)
    task = None if args.task == "all" else args.task
    t_start = time.time()

    print("== data", flush=True)
    train_all = load_claims("train", task_type=task)
    test = load_claims("test", task_type=task)
    fit, val = make_splits(sample_sources(train_all, 14 if args.smoke else args.train_sources))
    test = sample_sources(test, 4 if args.smoke else 0)
    leaks = {"fit_val": len(set(fit.source_id) & set(val.source_id)),
             "fit_test": len(set(fit.source_id) & set(test.source_id)),
             "val_test": len(set(val.source_id) & set(test.source_id))}
    for name, d in (("fit", fit), ("val (calibration)", val), ("test", test)):
        describe(d, name)
    print("shared passages:", leaks, flush=True)

    print("== features", flush=True)
    fit, Xfit = score_in_chunks(fit, run + "_fit")
    val, Xval = score_in_chunks(val, run + "_val")
    test, Xte = score_in_chunks(test, run + "_test")
    yfit, yval, yte = (d["label"].values.astype(int) for d in (fit, val, test))
    # lexical features do not depend on the verifier, so both verifiers share them
    Xfit, Xval, Xte = (np.hstack([X, cached_lexical(d, "%s_%s" % (lex, tag))])
                       for X, d, tag in ((Xfit, fit, "fit"), (Xval, val, "val"), (Xte, test, "test")))

    print("== model selection (grouped CV on the training split)", flush=True)
    gfit = fit["source_id"].values
    cv = {"v1_lr": grouped_cv(Xfit, yfit, gfit, V1), "v2_lr": grouped_cv(Xfit, yfit, gfit, ALL),
          "v2_trees": grouped_cv(Xfit, yfit, gfit, ALL, fit_trees)}
    for k, (mu, sd) in cv.items():
        print("  %-9s CV AUROC %.3f +- %.3f" % (k, mu, sd), flush=True)

    print("== detector", flush=True)
    allc = ALL
    model = fit_lr(Xfit, yfit, allc)
    raw_val, raw_te = proba(model, Xval, allc), proba(model, Xte, allc)
    raw_te_v1 = proba(fit_lr(Xfit, yfit, V1), Xte, V1)
    raw_te_trees = proba(fit_trees(Xfit, yfit, ALL), Xte, ALL)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_val, yval)
    p_val, p_te = iso.predict(raw_val), iso.predict(raw_te)
    platt = LogisticRegression().fit(logit(raw_val), yval)
    p_te_platt = platt.predict_proba(logit(raw_te))[:, 1]
    thr = E.best_f1_threshold(yval, p_val)
    # isotonic ties many claims, so ranking metrics use the raw LR score

    print("== baselines", flush=True)
    rng = np.random.default_rng(SEED)
    nli_only = -Xte[:, 0]
    whole = whole_passage_unsupported(test, run + "_test")
    scores = [("Random", rng.random(len(yte))),
              ("Word overlap", overlap_unsupported(test)),
              ("Embedding similarity", -Xte[:, 3]),
              ("NLI, whole passage", whole),
              ("NLI, best sentence", nli_only),
              ("TrustRAG v1 (6 features)", raw_te_v1),
              ("TrustRAG v2 (12 features)", raw_te),
              ("TrustRAG v2, boosted trees", raw_te_trees)]
    baselines = [dict(name=n, **E.detection(yte, s)) for n, s in scores]

    print("== analysis", flush=True)
    per_type = []
    lt = test["label_types"].fillna("")
    for t in TYPES:
        pos = (yte == 1) & lt.str.contains(t, regex=False).values
        mask = pos | (yte == 0)
        n = int(pos.sum())
        per_type.append({"type": t, "n": n,
                         "auroc": E.detection(yte[mask], raw_te[mask])["auroc"] if n else None,
                         "auroc_nli_only": E.detection(yte[mask], nli_only[mask])["auroc"] if n else None,
                         "recall": float((p_te[pos] >= thr).mean()) if n else None})

    per_model = []
    for name, g in test.groupby("model"):
        idx = g.index.values
        if 0 < yte[idx].sum() < len(idx):
            per_model.append({"model": name, "claims": len(idx), "base_rate": float(yte[idx].mean()),
                              "auroc": E.detection(yte[idx], raw_te[idx])["auroc"]})

    ablation = []
    for name, cols in (("All 12 features", ALL),
                       ("Without the lexical features (v1)", V1),
                       ("Without the NLI features", [c for c in ALL if c not in (0, 1, 2)]),
                       ("Without embedding similarity", [c for c in ALL if c != 3]),
                       ("Without position and answer length", [c for c in ALL if c not in (10, 11)]),
                       ("Lexical features only", list(range(6, 12))),
                       ("NLI features only", [0, 1, 2])):
        ablation.append({"model": name, "cv_auroc": grouped_cv(Xfit, yfit, gfit, cols)[0],
                         "auroc": E.detection(yte, proba(fit_lr(Xfit, yfit, cols), Xte, cols))["auroc"]})

    lr = model[-1]
    weights = [{"feature": FEATURE_LABELS[j], "weight": float(lr.coef_[0][k])} for k, j in enumerate(allc)]

    t_flag = E.threshold_for_risk(yval, p_val, RISK_TARGET)
    show, hold = p_te <= t_flag, p_te >= 0.5
    flag = ~show & ~hold
    actions = {name: {"share": float(mk.mean()),
                      "unsupported_rate": float(yte[mk].mean()) if mk.any() else None}
               for name, mk in (("show", show), ("flag", flag), ("hold", hold))}

    cov_t, risk_t = E.risk_coverage(yte, raw_te)
    cov_n, risk_n = E.risk_coverage(yte, nli_only)

    print("== HaluEval transfer", flush=True)
    he = load_halueval(30 if args.smoke else args.halueval)
    he, Xhe = score_in_chunks(he, run + "_halueval", chunk=250)
    Xhe = np.hstack([Xhe, cached_lexical(he, lex + "_halueval")])
    yhe = he["label"].values.astype(int)
    no_len = [c for c in ALL if c != 5]
    m_nolen = fit_lr(Xfit, yfit, no_len)
    transfer = [
        {"name": "TrustRAG v2, all features", **E.detection(yhe, proba(model, Xhe, allc))},
        {"name": "TrustRAG without claim length", **E.detection(yhe, proba(m_nolen, Xhe, no_len))},
        {"name": "NLI support score only", **E.detection(yhe, -Xhe[:, 0])},
        {"name": "Claim length only (artefact check)", **E.detection(yhe, Xhe[:, 5])},
    ]
    hm = load_halueval(30 if args.smoke else 100000, matched=True)
    hm, Xhm = score_in_chunks(hm, run + "_halueval_matched", chunk=250)
    Xhm = np.hstack([Xhm, cached_lexical(hm, lex + "_halueval_matched")])
    yhm = hm["label"].values.astype(int)
    transfer_matched = [
        {"name": "TrustRAG v2, all features", **E.detection(yhm, proba(model, Xhm, allc))},
        {"name": "TrustRAG without claim length", **E.detection(yhm, proba(m_nolen, Xhm, no_len))},
        {"name": "NLI support score only", **E.detection(yhm, -Xhm[:, 0])},
        {"name": "Claim length only (artefact check)", **E.detection(yhm, Xhm[:, 5])},
    ]

    print("== cost", flush=True)
    prev = os.path.join(out, "metrics.json")
    if args.keep_cost and os.path.exists(prev):
        with open(prev) as f:
            cost = json.load(f)["cost"]
    else:
        cost = benchmark(test)

    m = {
        "run": run, "task": args.task, "seed": SEED,
        "sizes": {name: {"claims": len(d), "unsupported": int(d["label"].sum()),
                         "passages": int(d["source_id"].nunique()),
                         "answers": int(d["response_id"].nunique())}
                  for name, d in (("fit", fit), ("val", val), ("test", test))},
        "halueval_examples": len(he), "shared_passages": leaks,
        "test": {"base_rate": float(yte.mean())},
        "baselines": baselines,
        "trustrag": {**E.detection(yte, raw_te), **E.prf_at(yte, p_te, thr)},
        "trustrag_v1": E.detection(yte, raw_te_v1), "trustrag_trees": E.detection(yte, raw_te_trees),
        "cv": {k: {"mean": mu, "std": sd} for k, (mu, sd) in cv.items()},
        "feature_names": FEATURE_LABELS,
        "calibration": {"raw": {"ece": E.ece(yte, raw_te), "brier": E.brier(yte, raw_te)},
                        "platt": {"ece": E.ece(yte, p_te_platt), "brier": E.brier(yte, p_te_platt)},
                        "isotonic": {"ece": E.ece(yte, p_te), "brier": E.brier(yte, p_te)}},
        "selective": {"risk_at_90": E.risk_at(yte, raw_te, 0.9), "risk_at_80": E.risk_at(yte, raw_te, 0.8),
                      "risk_at_50": E.risk_at(yte, raw_te, 0.5),
                      "nli_only_risk_at_90": E.risk_at(yte, nli_only, 0.9),
                      "risk_target": RISK_TARGET, "t_flag": t_flag, "t_hold": 0.5, "actions": actions},
        "per_type": per_type, "per_model": per_model, "ablation": ablation,
        "feature_weights": weights, "transfer": transfer, "cost": cost,
        "transfer_matched": transfer_matched, "halueval_matched_examples": len(hm),
        "runtime_min": (time.time() - t_start) / 60,
    }
    curves = {"trustrag": (cov_t, risk_t), "nli_only": (cov_n, risk_n)}
    rel = {"raw": E.reliability(yte, raw_te), "isotonic": E.reliability(yte, p_te)}
    m["reliability"] = rel
    keep = np.linspace(0, len(cov_t) - 1, 101).astype(int)
    m["risk_coverage"] = {"coverage": cov_t[keep].tolist(), "trustrag": risk_t[keep].tolist(),
                          "nli_only": risk_n[keep].tolist()}

    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    save_detector(os.path.join(out, "detector.json"), model, iso, t_flag)
    test.assign(p_unsupported=p_te, score_v2=raw_te, score_v1=raw_te_v1, score_trees=raw_te_trees,
                nli_support=Xte[:, 0], whole_passage=whole).drop(columns=["context"]).to_parquet(
        os.path.join(out, "test_predictions.parquet"), index=False)
    figures(out, m, curves, rel)

    print("\n== summary (%s, %.1f min)" % (run, m["runtime_min"]))
    for b in baselines:
        print("  %-24s AUROC %.3f  AUPRC %.3f" % (b["name"], b["auroc"], b["auprc"]))
    print("  F1 %.3f at threshold %.3f (chosen on val)" % (m["trustrag"]["f1"], thr))
    print("  ECE raw %.3f -> isotonic %.3f" % (m["calibration"]["raw"]["ece"],
                                               m["calibration"]["isotonic"]["ece"]))
    print("  risk at 90%% coverage %.3f (base rate %.3f)" % (m["selective"]["risk_at_90"], yte.mean()))
    for t in transfer:
        print("  HaluEval %-36s AUROC %.3f" % (t["name"], t["auroc"]))
    for t in transfer_matched:
        print("  HaluEval matched %-28s AUROC %.3f" % (t["name"], t["auroc"]))
    print("  %.0f ms per answer on %s" % (cost["ms_per_answer"], cost["device"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
