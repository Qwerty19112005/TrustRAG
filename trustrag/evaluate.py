"""Detection, calibration and selective-prediction metrics. Label 1 = unsupported."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def detection(y, s):
    """s: higher means more likely unsupported."""
    return {"auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "base_rate": float(np.mean(y))}


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    i = int(np.nanargmax(f[:-1]))
    return float(t[i])


def prf_at(y, s, thr):
    pred = s >= thr
    tp = int(np.sum(pred & (y == 1)))
    prec = tp / max(int(pred.sum()), 1)
    rec = tp / max(int(np.sum(y == 1)), 1)
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12), "threshold": float(thr)}


def _bins(p, n):
    return np.clip(np.digitize(p, np.linspace(0, 1, n + 1)[1:-1]), 0, n - 1)


def ece(y, p, n_bins=15):
    idx, e = _bins(p, n_bins), 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def reliability(y, p, n_bins=10):
    """Per non-empty bin: mean predicted probability, observed rate, count."""
    idx, rows = _bins(p, n_bins), []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def risk_coverage(y, p):
    """Show the most trusted claims first. Risk = unsupported share among those shown."""
    order = np.argsort(p, kind="stable")
    hits = np.cumsum(y[order])
    k = np.arange(1, len(y) + 1)
    return k / len(y), hits / k


def risk_at(y, p, coverage):
    cov, risk = risk_coverage(y, p)
    return float(risk[min(np.searchsorted(cov, coverage), len(cov) - 1)])


def threshold_for_risk(y, p, target):
    """Largest score threshold whose shown claims stay at or under the target risk."""
    order = np.argsort(p, kind="stable")
    risk = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
    ok = np.where(risk <= target)[0]
    return float(p[order][ok[-1]]) if len(ok) else 0.0
