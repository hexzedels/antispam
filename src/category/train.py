"""Category model: weak labels from zero-shot NLI + keyword rules -> light student classifier.

Usage: python -m src.category.train

1. Weak labels for the training sample (data/category/sample.parquet, split == "train"):
   zero-shot probabilities (src/category/zeroshot.py), optionally divided by the mean
   probability of each class over the sample (removes the NLI model's bias towards
   "newsletter"), plus a bonus for keyword-rule hits (src/category/rules.py). The spam
   label of the training corpus is known, so ham emails may only get ham-like classes.
2. Students trained on the weak labels: TF-IDF word+char -> LogisticRegression,
   multilingual-e5-small embeddings -> LogisticRegression (embeddings cached by train_embed),
   and the average of their probabilities.
   Only students with the multilingual encoder are deployable (the training sample is
   English only, the mail it will see is not).
3. The hand-labelled gold set (data/category/gold.csv) is split in half: "dev" picks the
   weak-label variant and the student, "test" is used only for the reported numbers.
   Gold emails are never used for training.
Outputs: models/category/model.joblib, reports/category_results.json
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from .common import CAT_DATA, CAT_MODELS, CLASSES, PROC, ROOT
from .rules import BONUS, rule_hits

SEED = 42
HAM_CLASSES = ["job", "newsletter", "other"]
CLASSES_SORTED = sorted(CLASSES)  # sklearn orders classes_ alphabetically
MIN_CONF = 0.35
# Source labelling functions for ham (training sample only, the spam label and source are
# known there): Enron mailboxes and SMS are personal or business correspondence, Ling-Spam
# ham is a linguistics mailing list. Used as a variant, kept only if it helps on gold dev.
SRC_LF = {"enron": "other", "sms": "other", "lingspam": "newsletter"}
SRC_BONUS = 0.5
REPORTS = ROOT / "reports"


def weak_scores(s: pd.DataFrame, zs: np.ndarray, calibrate: bool, rules: bool, prior: np.ndarray) -> np.ndarray:
    p = zs / prior if calibrate else zs.copy()
    p = p / p.sum(1, keepdims=True)
    if rules:
        p = p + BONUS * rule_hits(s.input.str.lower().tolist())
    return p


def constrain_ham(p: np.ndarray, is_ham: np.ndarray) -> np.ndarray:
    p = p.copy()
    p[np.ix_(is_ham, [CLASSES.index(c) for c in CLASSES if c not in HAM_CLASSES])] = -1
    return p


def source_lf(s: pd.DataFrame, p: np.ndarray, is_ham: np.ndarray) -> np.ndarray:
    p = p.copy()
    for src, c in SRC_LF.items():
        p[is_ham & (s.source == src).to_numpy(), CLASSES.index(c)] += SRC_BONUS
    return p


def report(y_true, y_pred) -> dict:
    pr, rc, f1, n = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "per_class": {c: {"precision": pr[i], "recall": rc[i], "f1": f1[i], "support": int(n[i])}
                      for i, c in enumerate(CLASSES)},
        "cm": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
    }


def tfidf_student() -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=150_000, sublinear_tf=True)),
    ])
    return Pipeline([("vec", feats), ("clf", LogisticRegression(C=5.0, max_iter=3000, class_weight="balanced"))])


def emb_student() -> LogisticRegression:
    return LogisticRegression(C=5.0, max_iter=3000, class_weight="balanced")


def main() -> None:
    t0 = time.time()
    s = pd.read_parquet(CAT_DATA / "sample.parquet")
    zs_all = {}
    for tag in ("mdeberta", "large"):
        f = CAT_DATA / f"zs_{tag}.parquet"
        if f.exists():
            z = pd.read_parquet(f)
            assert (z.id.values == s.id.values).all()
            zs_all[tag] = z[CLASSES].to_numpy()
    if len(zs_all) == 2:
        zs_all["avg"] = (zs_all["mdeberta"] + zs_all["large"]) / 2
    gold = pd.read_csv(CAT_DATA / "gold.csv", dtype=str).fillna("")
    assert gold.label.isin(CLASSES).all(), "every gold email needs a label from CLASSES"
    pos = pd.Series(np.arange(len(s)), index=s.id)

    # e5 embeddings: rows of train.parquet / course_test.parquet, ids are "tr<row>" / "te<row>".
    z = np.load(PROC / "emb_e5small.npz")
    row = s.id.str.slice(2).astype(int).to_numpy()
    is_tr = (s.split == "train").to_numpy()
    E = np.empty((len(s), z["train"].shape[1]), dtype=np.float32)
    E[is_tr], E[~is_tr] = z["train"][row[is_tr]], z["test"][row[~is_tr]]

    dev, test = train_test_split(gold, test_size=0.5, random_state=SEED, stratify=gold.label)
    gi = {"dev": pos[dev.id].to_numpy(), "test": pos[test.id].to_numpy()}
    gy = {"dev": dev.label.to_numpy(), "test": test.label.to_numpy()}

    train = (s.split == "train").to_numpy() & ~s.id.isin(gold.id).to_numpy()
    is_ham = (s.label == 0).to_numpy()
    docs = s.input.tolist()
    res: dict = {"n_gold": {"dev": len(dev), "test": len(test)}, "weak": {}, "student": {}}

    # --- weak-label variants, evaluated directly on gold (no spam label used there) ---
    variants = {f"{t}{'_cal' if c else ''}{'_rules' if r else ''}": (t, c, r)
                for t in zs_all for c in (False, True) for r in (False, True)}

    def weak(t, c, r):
        zs = zs_all[t]
        return weak_scores(s, zs, c, r, zs[is_tr].mean(0))

    for name, v in variants.items():
        pred = np.array(CLASSES)[weak(*v).argmax(1)]
        res["weak"][name] = {k: report(gy[k], pred[gi[k]]) for k in ("dev", "test")}
    best_weak = max(variants, key=lambda v: res["weak"][v]["dev"]["macro_f1"])
    print("weak dev macro-F1:", {v: round(res["weak"][v]["dev"]["macro_f1"], 3) for v in variants}, "->", best_weak)

    glang = {"dev": dev.lang.to_numpy(), "test": test.lang.to_numpy()}
    res["weak_train"], fitted, proba = {}, {}, {}
    for lf in (False, True):
        tag = "+srclf" if lf else ""
        p = weak(*variants[best_weak])
        p = constrain_ham(source_lf(s, p, is_ham) if lf else p, is_ham)
        conf = np.sort(p, 1)[:, -1]
        keep = train & (conf >= MIN_CONF)
        yw = np.array(CLASSES)[p.argmax(1)]
        res["weak_train"][f"{best_weak}{tag}"] = {
            "n": int(keep.sum()), "n_dropped_low_conf": int((train & ~keep).sum()),
            "class_counts": pd.Series(yw[keep]).value_counts().to_dict()}

        # Students; predict_proba over the same class order, so they can be averaged.
        tfidf = tfidf_student().fit([docs[i] for i in np.flatnonzero(keep)], yw[keep])
        emb = emb_student().fit(E[keep], yw[keep])
        assert list(tfidf.classes_) == list(emb.classes_) == CLASSES_SORTED
        fitted[f"tfidf_logreg{tag}"] = {"tfidf": tfidf}
        fitted[f"e5_logreg{tag}"] = {"e5": emb}
        fitted[f"ensemble{tag}"] = {"tfidf": tfidf, "e5": emb}
    print("weak train sets:", res["weak_train"])

    def make_proba(models):
        def f(idx):
            parts = []
            if "tfidf" in models:
                parts.append(models["tfidf"].predict_proba([docs[i] for i in idx]))
            if "e5" in models:
                parts.append(models["e5"].predict_proba(E[idx]))
            return np.mean(parts, axis=0)
        return f

    labels = np.array(CLASSES_SORTED)
    for name, models in fitted.items():
        f = proba[name] = make_proba(models)
        res["student"][name] = {}
        for k in ("dev", "test"):
            pred = labels[f(gi[k]).argmax(1)]
            r = report(gy[k], pred)
            r["accuracy_by_lang"] = {l: accuracy_score(gy[k][glang[k] == l], pred[glang[k] == l])
                                     for l in np.unique(glang[k])}
            res["student"][name][k] = r
    print("student dev macro-F1:", {n: round(res["student"][n]["dev"]["macro_f1"], 3) for n in proba})

    # The service has to cope with non-English mail and the training sample is English only,
    # so only students with the multilingual encoder are deployable; TF-IDF-only students
    # stay in the report for comparison. Among the deployable ones gold dev decides.
    deployable = [n for n in proba if "e5" in fitted[n]]
    best = max(deployable, key=lambda n: res["student"][n]["dev"]["macro_f1"])
    res["chosen"] = best
    res["weak_train_variants"] = res["weak_train"]
    wkey = best_weak + ("+srclf" if best.endswith("+srclf") else "")
    res["weak_train"] = {"variant": best_weak, "source_lf": best.endswith("+srclf"), **res["weak_train_variants"][wkey]}
    CAT_MODELS.mkdir(parents=True, exist_ok=True)
    models = fitted[best]
    art = {"kind": best, "classes": CLASSES_SORTED, "models": models, "weak_variant": best_weak}
    if "e5" in models:
        art["encoder"] = "intfloat/multilingual-e5-small"
    joblib.dump(art, CAT_MODELS / "model.joblib", compress=3)

    # The service path: raw "subject + body" text through predict.py (int8 ONNX encoder).
    if (CAT_MODELS / "e5" / "model.onnx").exists() or "e5" not in models:
        from .common import CLASS_RU
        from .predict import load, predict

        ru2id = {v: k for k, v in CLASS_RU.items()}
        g = s.iloc[gi["test"]]
        out = predict(load(), [f"{a or ''}\n{b or ''}" for a, b in zip(g.subject, g.text)])
        res["served_gold_test"] = report(gy["test"], np.array([ru2id[o["category"]] for o in out]))
        print("served gold test macro-F1", round(res["served_gold_test"]["macro_f1"], 3))
    (REPORTS / "category_results.json").write_text(json.dumps(res, indent=2, default=float))
    for k in ("dev", "test"):
        r = res["student"][best][k]
        print(k, best, f"acc {r['accuracy']:.3f} macro-F1 {r['macro_f1']:.3f}", r["accuracy_by_lang"])
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
