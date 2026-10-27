"""Side baseline on tabular SpamBase (57 precomputed features, no raw text).

Usage: python -m src.models.spambase_baseline
Not mixed with the text corpora: its features cannot be computed for new emails
the same way, so it only serves as a reference point. Output: reports/spambase_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .train_classic import metrics, pick_threshold

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "processed" / "spambase.parquet")
    X, y = df.drop(columns="label").to_numpy(), df.label.to_numpy()
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    models = {
        "logreg": make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=5000)),
        "histgb": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=42),
    }
    out = {}
    for n, m in models.items():
        s = cross_val_predict(m, X, y, cv=skf, method="predict_proba", n_jobs=5)[:, 1]
        out[n] = {"default": metrics(y, s, 0.5), "tuned": metrics(y, s, pick_threshold(y, s))}
        print(n, {k: round(v, 4) for k, v in out[n]["default"].items() if isinstance(v, float)})
    (ROOT / "reports" / "spambase_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
