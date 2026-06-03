"""Per-model composite trim-risk model.

A small logistic regression over orthogonal trim-effort features that predicts
whether a track fails linearity. Its per-unit score (0..1) is persisted and
watched at the group level for upstream drift. Honest grouped-CV + a deploy-gate
keep it from shipping where it doesn't actually beat the best single signal.

See docs/superpowers/plans/2026-06-01-composite-trim-risk-early-warning.md.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Candidate features, in priority order. The model uses whichever have data.
FEATURES: List[str] = [
    "untrimmed_error_max",        # worst raw point (best single signal)
    "untrimmed_sigma_gradient",   # raw steepness (often redundant)
    "resistance_change_percent",  # orthogonal: resistance-offset mode
    "trim_pass_count",            # orthogonal: trim-headroom mode (reprocess gate)
]

MIN_SAMPLES = 60       # below this we don't fit
MIN_FAILS = 15         # need both classes with enough fails
MIN_LIFT = 0.02        # CV AUC must beat best single feature by this
MIN_CONFIDENCE = 0.20  # honest confidence floor to deploy


@dataclass
class CompositeTrainingResult:
    model_name: str
    n_samples: int
    n_fails: int
    features_used: List[str]
    cv_auc: float                 # grouped out-of-fold AUC of the composite
    best_single_auc: float        # best single-feature grouped OOF AUC
    confidence: float             # rank-AUC honest confidence in [0,1]
    deployed: bool                # passed the gate?
    reason: str = ""              # why deployed / not
    coef: Dict[str, float] = field(default_factory=dict)


class CompositeRiskModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.features_used: List[str] = []
        self._pipe = None            # sklearn Pipeline (imputer+scaler+logreg)
        self._feat_median: Dict[str, float] = {}
        self.result: Optional[CompositeTrainingResult] = None
        self.is_trained = False

    # ---- training -------------------------------------------------------
    def _grouped_oof_auc(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Optional[float]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.metrics import roc_auc_score
        k = min(5, len(set(groups.tolist())))
        if k < 2 or len(set(y.tolist())) < 2:
            return None
        pipe = make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler(),
                             LogisticRegression(max_iter=1000))
        try:
            p = cross_val_predict(pipe, X, y, cv=GroupKFold(k), groups=groups,
                                  method="predict_proba")[:, 1]
            return float(roc_auc_score(y, p))
        except Exception:
            return None

    def train(self, df: pd.DataFrame) -> CompositeTrainingResult:
        # label: fail = 1
        y_full = (~df["linearity_pass"].astype(bool)).astype(int).to_numpy()
        groups_full = df.get("serial")
        groups_full = (groups_full.fillna(pd.Series([f"_{i}" for i in range(len(df))]))
                       if groups_full is not None else pd.Series(range(len(df)))).to_numpy()

        # keep only features that have at least some non-null data
        feats = [f for f in FEATURES if f in df.columns and df[f].notna().any()]
        self.features_used = feats

        def _result(**kw):
            self.result = CompositeTrainingResult(
                model_name=self.model_name, features_used=list(feats), **kw)
            return self.result

        n = len(df); n_fail = int(y_full.sum())
        if n < MIN_SAMPLES or n_fail < MIN_FAILS or (n - n_fail) < MIN_FAILS or not feats:
            return _result(n_samples=n, n_fails=n_fail, cv_auc=0.5, best_single_auc=0.5,
                           confidence=0.0, deployed=False,
                           reason="insufficient data", coef={})

        X = df[feats].to_numpy(dtype=float)

        # composite grouped OOF AUC
        cv_auc = self._grouped_oof_auc(X, y_full, groups_full) or 0.5
        # best single-feature grouped OOF AUC
        best_single = 0.5
        for j, f in enumerate(feats):
            a = self._grouped_oof_auc(X[:, [j]], y_full, groups_full)
            if a is not None:
                best_single = max(best_single, a, 1 - a)  # direction-agnostic

        # honest confidence (threshold_optimizer pattern)
        strength = max(0.0, min(1.0, 2.0 * (cv_auc - 0.5)))
        n_factor = min(1.0, n / 200.0)
        confidence = round(strength * n_factor, 3)

        lift = cv_auc - best_single
        deployed = (lift >= MIN_LIFT) and (confidence >= MIN_CONFIDENCE)
        reason = (f"lift={lift:+.3f} (need >= {MIN_LIFT}), conf={confidence} "
                  f"(need >= {MIN_CONFIDENCE})")

        # fit the final pipeline on ALL rows (median-impute remembered for scoring)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        self._feat_median = {f: float(np.nanmedian(df[f].to_numpy(dtype=float)))
                             for f in feats}
        self._pipe = make_pipeline(SimpleImputer(strategy="median"),
                                   StandardScaler(),
                                   LogisticRegression(max_iter=1000)).fit(X, y_full)
        self.is_trained = True
        coef = dict(zip(feats, self._pipe.steps[-1][1].coef_.ravel().tolist()))

        return _result(n_samples=n, n_fails=n_fail, cv_auc=round(cv_auc, 3),
                       best_single_auc=round(best_single, 3), confidence=confidence,
                       deployed=deployed, reason=reason, coef=coef)

    # ---- scoring --------------------------------------------------------
    def predict_proba(self, feat: Dict[str, float]) -> float:
        if not self.is_trained or self._pipe is None:
            return float("nan")
        row = []
        for f in self.features_used:
            v = feat.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = self._feat_median.get(f, 0.0)
            row.append(float(v))
        X = np.asarray(row, dtype=float).reshape(1, -1)
        return float(self._pipe.predict_proba(X)[0, 1])

    # ---- persistence (predictor.py pattern) ----------------------------
    def save(self, path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "model_name": self.model_name,
                "features_used": self.features_used,
                "pipe": self._pipe,
                "feat_median": self._feat_median,
                "result": self.result,
                "is_trained": self.is_trained,
            }, fh)

    @classmethod
    def load(cls, path) -> "CompositeRiskModel":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        m = cls(d["model_name"])
        m.features_used = d["features_used"]; m._pipe = d["pipe"]
        m._feat_median = d["feat_median"]; m.result = d["result"]
        m.is_trained = d["is_trained"]
        return m
