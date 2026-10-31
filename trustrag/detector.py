"""The detector: logistic regression on twelve features, then isotonic calibration."""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trustrag import features as F
from trustrag.data import ROOT, _nlp
from trustrag.lexical import lexical_features

SEED = 42
V1 = list(range(6))               # the six features from the proposal
ALL = list(range(12))             # plus the six added later
COUNT_COLS = [4, 5, 8, 9, 11]     # skewed counts, so we take logs
T_HOLD = 0.5
DEFAULT_PATH = os.path.join(ROOT, "results", "qa", "detector.json")


def prep(X):
    X = X.astype(np.float64).copy()
    cols = [c for c in COUNT_COLS if c < X.shape[1]]
    X[:, cols] = np.log1p(X[:, cols])
    return X


def fit_lr(X, y, cols):
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    return m.fit(prep(X)[:, cols], y)


def fit_trees(X, y, cols):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                       random_state=SEED)
    return m.fit(prep(X)[:, cols], y)


def proba(m, X, cols):
    return m.predict_proba(prep(X)[:, cols])[:, 1]


def save_detector(path, model, iso, t_flag):
    scaler, lr = model[0], model[-1]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                   "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
                   "iso_x": iso.X_thresholds_.tolist(), "iso_y": iso.y_thresholds_.tolist(),
                   "t_flag": t_flag}, f, indent=1)


class Detector:
    """The fitted model from detector.json, so the demo needs no training features."""

    def __init__(self, path=DEFAULT_PATH):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.mean, self.scale = np.array(d["mean"]), np.array(d["scale"])
        self.coef, self.intercept = np.array(d["coef"]), d["intercept"]
        self.iso_x, self.iso_y = np.array(d["iso_x"]), np.array(d["iso_y"])
        self.t_flag = d["t_flag"]

    def score(self, X):
        """Returns the raw score and the calibrated probability."""
        z = (prep(X)[:, ALL] - self.mean) / self.scale
        raw = 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))
        return raw, np.interp(raw, self.iso_x, self.iso_y)

    def action(self, p):
        return "show" if p <= self.t_flag else ("hold" if p >= T_HOLD else "flag")


def check(answer, passages, detector):
    """Score each sentence of an answer against the passages. Returns the rows and the time in ms."""
    claims = [" ".join(s.text.split()) for s in _nlp()(answer).sents if s.text.strip()]
    if not claims:
        return [], 0.0
    context = "\n\n".join(passages)
    df = pd.DataFrame({"claim": claims, "context": context, "label": 0, "response_id": 0,
                       "claim_idx": range(len(claims))})
    t0 = time.time()
    X6, _ = F.extract(df, verbose=False)
    X = np.hstack([X6, lexical_features(df)])
    raw, p = detector.score(X)
    ms = 1000 * (time.time() - t0)
    # best supporting sentence, shown next to a flagged claim (not timed)
    evidence = F.split_sentences(context)
    rows = []
    for i, c in enumerate(claims):
        P = F.nli_scores([(e, c) for e in evidence])
        rows.append({"claim": c, "raw": float(raw[i]), "p": float(p[i]), "action": detector.action(p[i]),
                     "support": float(X[i, 0]), "novel_words": float(X[i, 7]),
                     "evidence": evidence[int(np.argmax(P[:, 0] - P[:, 2]))]})
    return rows, ms
