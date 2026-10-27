"""Phase 2a: TF-IDF (word + char) -> LogReg / LinearSVC / ComplementNB.

Usage: python -m src.models.train_classic

- Stratified 5-fold CV (stratified by source x label), folds saved to data/processed/folds.parquet
  so other model families reuse the same split.
- Leave-one-source-out (LOSO): train on N-1 sources, test on the held-out one.
- Threshold picked on OOF scores: max recall s.t. precision >= 0.99 and FPR <= 0.5%.
- Final model refit on the whole training set, evaluated once on the course test set.
Outputs: models/classic_<name>.joblib, models/oof_classic.parquet, reports/classic_results.json,
reports/classic_tables.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import join_fields, make_vectorizer

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
SEED = 42
N_FOLDS = 5
MIN_PRECISION = 0.99
MAX_FPR = 0.005

MODELS = {
    "logreg": lambda: LogisticRegression(C=10.0, solver="liblinear", max_iter=2000),
    "linsvc": lambda: LinearSVC(C=0.5),
    "cnb": lambda: ComplementNB(alpha=0.1),
}
# Natural decision threshold of each model's score.
DEFAULT_THR = {"logreg": 0.5, "linsvc": 0.0, "cnb": 0.5}


def score(clf, X) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def metrics(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    p = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "threshold": float(thr),
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "accuracy": (tp + tn) / len(y),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, s)
        out["pr_auc"] = average_precision_score(y, s)
    return out


def pick_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Lowest threshold (max recall) with precision >= MIN_PRECISION and FPR <= MAX_FPR."""
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    # Only evaluate at the last index of each run of equal scores.
    last = np.r_[ss[1:] != ss[:-1], True]
    prec = tp / (tp + fp)
    fpr = fp / max(1, (y == 0).sum())
    ok = last & (prec >= MIN_PRECISION) & (fpr <= MAX_FPR)
    if not ok.any():
        return float(ss[0]) + 1e-9
    idx = np.flatnonzero(ok)
    best = idx[np.argmax(tp[idx])]
    return float(ss[best])


def fit_vec(docs_tr, docs_te):
    vec = make_vectorizer()
    Xtr = vec.fit_transform(docs_tr)
    return vec, Xtr, vec.transform(docs_te)


def run_split(docs, y, tr, te):
    vec, Xtr, Xte = fit_vec([docs[i] for i in tr], [docs[i] for i in te])
    out = {}
    for name, mk in MODELS.items():
        clf = mk().fit(Xtr, y[tr])
        out[name] = score(clf, Xte)
    return te, out


def main() -> None:
    t0 = time.time()
    df = pd.read_parquet(PROC / "train.parquet")
    test = pd.read_parquet(PROC / "course_test.parquet")
    docs = join_fields(df.subject, df.text)
    tdocs = join_fields(test.subject, test.text)
    y = df.label.to_numpy()
    yt = test.label.to_numpy()
    print(f"train {len(df)} test {len(test)} normalised in {time.time() - t0:.0f}s")

    strat = df.source + "_" + df.label.astype(str)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.zeros(len(df), dtype=int)
    for k, (_, va) in enumerate(skf.split(df, strat)):
        fold[va] = k
    pd.DataFrame({"file": df.file, "source": df.source, "label": y, "fold": fold}).to_parquet(
        PROC / "folds.parquet", index=False)

    # --- 5-fold CV (folds in parallel) ---
    splits = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(N_FOLDS)]
    res = Parallel(n_jobs=N_FOLDS)(delayed(run_split)(docs, y, tr, va) for tr, va in splits)
    oof = {n: np.zeros(len(df)) for n in MODELS}
    for va, out in res:
        for n, s in out.items():
            oof[n][va] = s
    print(f"cv done {time.time() - t0:.0f}s")
    pd.DataFrame({"file": df.file, "source": df.source, "label": y, "fold": fold,
                  **{f"oof_{n}": s for n, s in oof.items()}}).to_parquet(
        MODELS_DIR / "oof_classic.parquet", index=False)

    results: dict = {"cv": {}, "loso": {}, "test": {}, "per_fold": {}, "per_source_oof": {}}
    thresholds = {}
    for n in MODELS:
        thr = pick_threshold(y, oof[n])
        thresholds[n] = thr
        results["cv"][n] = {"default": metrics(y, oof[n], DEFAULT_THR[n]),
                            "tuned": metrics(y, oof[n], thr)}
        results["per_fold"][n] = [
            {k: v for k, v in metrics(y[fold == k], oof[n][fold == k], DEFAULT_THR[n]).items() if k != "cm"}
            for k in range(N_FOLDS)]
        results["per_source_oof"][n] = {
            src: {k: v for k, v in metrics(y[m], oof[n][m], thr).items()}
            for src in sorted(df.source.unique()) for m in [(df.source == src).to_numpy()]}

    # --- leave-one-source-out ---
    sources = sorted(df.source.unique())
    loso_splits = [(np.flatnonzero(df.source != s), np.flatnonzero(df.source == s)) for s in sources]
    res = Parallel(n_jobs=len(sources))(delayed(run_split)(docs, y, tr, te) for tr, te in loso_splits)
    for src, (te, out) in zip(sources, res):
        results["loso"][src] = {n: metrics(y[te], s, DEFAULT_THR[n]) for n, s in out.items()}
    print(f"loso done {time.time() - t0:.0f}s")

    # --- final refit on all training data, evaluate every model on the course test ---
    vec = make_vectorizer()
    X = vec.fit_transform(docs)
    Xt = vec.transform(tdocs)
    for n, mk in MODELS.items():
        clf = mk().fit(X, y)
        st = score(clf, Xt)
        results["test"][n] = {"default": metrics(yt, st, DEFAULT_THR[n]),
                              "tuned": metrics(yt, st, thresholds[n])}
        pipe = Pipeline([("vec", vec), ("clf", clf)])
        joblib.dump({"pipeline": pipe, "threshold": thresholds[n], "default_threshold": DEFAULT_THR[n],
                     "score": "proba" if hasattr(clf, "predict_proba") else "decision",
                     "preprocess": "src.models.features.join_fields"},
                    MODELS_DIR / f"classic_{n}.joblib", compress=3)
        pd.DataFrame({"file": test.file, "label": yt, "score": st,
                      "pred": (st >= thresholds[n]).astype(int)}).to_csv(
            REPORTS / f"course_test_predictions_{n}.csv", index=False)
    results["thresholds"] = thresholds
    results["n_features"] = int(X.shape[1])

    # Model choice uses OOF only (never the course test): max recall at the FP constraint.
    chosen = max(MODELS, key=lambda n: results["cv"][n]["tuned"]["recall"])
    art = joblib.load(MODELS_DIR / f"classic_{chosen}.joblib")
    # Platt scaling fitted on OOF scores so the service can return a probability for any model;
    # monotone, so the threshold maps 1:1 onto the probability scale.
    platt = LogisticRegression(C=1e6).fit(oof[chosen].reshape(-1, 1), y)
    art["calibrator"] = platt
    art["threshold_proba"] = float(platt.predict_proba([[thresholds[chosen]]])[0, 1])
    art["name"] = chosen
    joblib.dump(art, MODELS_DIR / "final_classic.joblib", compress=3)
    results["chosen"] = {"model": chosen, "threshold": thresholds[chosen],
                         "threshold_proba": art["threshold_proba"]}
    print(f"final done {time.time() - t0:.0f}s")

    (REPORTS / "classic_results.json").write_text(json.dumps(results, indent=2))
    write_tables(results)


def fmt_row(name, m):
    auc = f"{m['roc_auc']:.4f} | {m['pr_auc']:.4f}" if "roc_auc" in m else "- | -"
    return (f"| {name} | {m['threshold']:.4g} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
            f"{m['fpr'] * 100:.2f}% | {m['accuracy']:.4f} | {auc} | {m['cm']} |")


HDR = ("| модель | порог | Precision | Recall | F1 | FPR | Accuracy | ROC-AUC | PR-AUC | CM [[TN,FP],[FN,TP]] |\n"
       "|---|---|---|---|---|---|---|---|---|---|")


def write_tables(r: dict) -> None:
    L = ["### 5-fold CV (OOF), порог по умолчанию", HDR]
    L += [fmt_row(n, r["cv"][n]["default"]) for n in r["cv"]]
    L += ["", "### 5-fold CV (OOF), порог под Precision >= 0.99 и FPR <= 0.5%", HDR]
    L += [fmt_row(n, r["cv"][n]["tuned"]) for n in r["cv"]]
    L += ["", "### Разброс F1 по фолдам (порог по умолчанию)", "| модель | F1 по фолдам | mean ± std |", "|---|---|---|"]
    for n, folds in r["per_fold"].items():
        f1 = np.array([f["f1"] for f in folds])
        L.append(f"| {n} | {', '.join(f'{v:.4f}' for v in f1)} | {f1.mean():.4f} ± {f1.std():.4f} |")
    L += ["", "### Leave-one-source-out (обучение на остальных источниках), порог по умолчанию"]
    for src, d in r["loso"].items():
        L += ["", f"Отложенный источник: **{src}**", HDR] + [fmt_row(n, m) for n, m in d.items()]
    L += ["", "### OOF по источникам (подобранный порог)"]
    for n, d in r["per_source_oof"].items():
        L += ["", f"**{n}**", HDR] + [fmt_row(src, m) for src, m in d.items()]
    L += ["", "### Тестовый набор курса (1600 писем: 600 ham / 1000 spam)", "", "Порог по умолчанию:", HDR]
    L += [fmt_row(n, r["test"][n]["default"]) for n in r["test"]]
    L += ["", "Порог, подобранный на OOF:", HDR] + [fmt_row(n, r["test"][n]["tuned"]) for n in r["test"]]
    (REPORTS / "classic_tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
