"""Phase 2b: frozen multilingual-e5 sentence embeddings -> LogisticRegression.

Usage: python -m src.models.train_embed [small|base]

Same folds as the TF-IDF models (data/processed/folds.parquet), same OOF threshold rule.
The encoder is frozen, so embeddings are computed once for all texts and cached.
Outputs: models/embed_e5<size>.joblib, models/oof_e5<size>.parquet, models/test_e5<size>.parquet,
reports/e5<size>_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .train_classic import metrics, pick_threshold

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
MAX_LEN = 256
MAX_CHARS = 3000


def e5_inputs(subject, text) -> list[str]:
    # e5 expects a "query: " prefix for classification-style use.
    return [f"query: {s}\n{t[:MAX_CHARS]}" for s, t in zip(subject, text)]


def encode(model_name: str, texts: list[str], batch_size: int = 128) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SentenceTransformer(model_name, device=device)
    enc.max_seq_length = MAX_LEN
    if device == "cuda":
        enc.half()
    return enc.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                      show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def main(size: str = "small") -> None:
    name = f"e5{size}"
    model_name = f"intfloat/multilingual-e5-{size}"
    t0 = time.time()
    df = pd.read_parquet(PROC / "train.parquet")
    test = pd.read_parquet(PROC / "course_test.parquet")
    folds = pd.read_parquet(PROC / "folds.parquet")
    assert (folds.file.values == df.file.values).all()
    fold = folds.fold.to_numpy()
    y, yt = df.label.to_numpy(), test.label.to_numpy()

    cache = PROC / f"emb_{name}.npz"
    if cache.exists():
        z = np.load(cache)
        E, Et = z["train"], z["test"]
    else:
        E = encode(model_name, e5_inputs(df.subject, df.text))
        Et = encode(model_name, e5_inputs(test.subject, test.text))
        np.savez(cache, train=E, test=Et)
    print(f"embeddings {E.shape} in {time.time() - t0:.0f}s")

    def clf():
        return LogisticRegression(C=10.0, max_iter=5000)

    oof = np.zeros(len(df))
    for k in range(fold.max() + 1):
        tr, va = fold != k, fold == k
        oof[va] = clf().fit(E[tr], y[tr]).predict_proba(E[va])[:, 1]
    thr = pick_threshold(y, oof)
    final = clf().fit(E, y)
    st = final.predict_proba(Et)[:, 1]

    res = {"model": model_name, "threshold": thr,
           "cv": {"default": metrics(y, oof, 0.5), "tuned": metrics(y, oof, thr)},
           "test": {"default": metrics(yt, st, 0.5), "tuned": metrics(yt, st, thr)}}
    pd.DataFrame({"file": df.file, "label": y, "fold": fold, f"oof_{name}": oof}).to_parquet(
        MODELS_DIR / f"oof_{name}.parquet", index=False)
    pd.DataFrame({"file": test.file, "label": yt, f"test_{name}": st}).to_parquet(
        MODELS_DIR / f"test_{name}.parquet", index=False)
    joblib.dump({"encoder": model_name, "max_len": MAX_LEN, "clf": final, "threshold": thr},
                MODELS_DIR / f"embed_{name}.joblib")
    (REPORTS / f"{name}_results.json").write_text(json.dumps(res, indent=2))
    for part in ("cv", "test"):
        for k in ("default", "tuned"):
            m = res[part][k]
            print(part, k, {x: round(v, 4) for x, v in m.items() if isinstance(v, float)}, m["cm"])
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "small")
