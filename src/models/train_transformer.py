"""Phase 2c: fine-tune a small multilingual transformer (xlm-roberta-base) on GPU.

Usage: python -m src.models.train_transformer [--model xlm-roberta-base] [--epochs 2]

Runs the same 5 folds as the other models (OOF scores), then one more run on the whole
training set whose weights are saved to models/xlmr/ and scored on the course test set.
Finished folds are cached in models/xlmr_folds/, so an interrupted run resumes.
Input text goes through the same normalisation as the TF-IDF models (features.join_fields).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .features import join_fields
from .train_classic import metrics, pick_threshold

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
MAX_LEN = 256
SEED = 42


def tokenize(tok, texts: list[str]) -> list[torch.Tensor]:
    enc = tok(texts, truncation=True, max_length=MAX_LEN)
    return [torch.tensor(ids) for ids in enc["input_ids"]]


def batches(ids: list[torch.Tensor], order: np.ndarray, bs: int, pad_id: int):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        x = pad_sequence([ids[j] for j in idx], batch_first=True, padding_value=pad_id)
        yield idx, x, (x != pad_id).long()


def train(model_name, ids, y, pad_id, epochs, bs=32, lr=3e-5, log_prefix=""):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(ids) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    yt = torch.tensor(y)
    model.train()
    step, t0 = 0, time.time()
    for ep in range(epochs):
        # Bucket by length inside shuffled chunks: less padding, still random.
        perm = rng.permutation(len(ids))
        chunks = [perm[i:i + bs * 50] for i in range(0, len(perm), bs * 50)]
        order = np.concatenate([c[np.argsort([len(ids[j]) for j in c])] for c in chunks])
        bl = [order[i:i + bs] for i in range(0, len(order), bs)]
        rng.shuffle(bl)
        for idx in bl:
            x = pad_sequence([ids[j] for j in idx], batch_first=True, padding_value=pad_id).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=x, attention_mask=(x != pad_id).long(), labels=yt[idx].cuda())
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0:
                print(f"{log_prefix} ep{ep} step {step}/{steps} loss {out.loss.item():.4f} {time.time() - t0:.0f}s",
                      flush=True)
    return model


@torch.no_grad()
def predict(model, ids, pad_id, bs=128) -> np.ndarray:
    model.eval()
    order = np.argsort([len(t) for t in ids])
    out = np.zeros(len(ids))
    for idx, x, m in batches(ids, order, bs, pad_id):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x.cuda(), attention_mask=m.cuda()).logits.float()
        out[idx] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()
    t0 = time.time()
    df = pd.read_parquet(PROC / "train.parquet")
    test = pd.read_parquet(PROC / "course_test.parquet")
    folds = pd.read_parquet(PROC / "folds.parquet")
    assert (folds.file.values == df.file.values).all()
    fold = folds.fold.to_numpy()
    y, yt = df.label.to_numpy(), test.label.to_numpy()

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tokenize(tok, join_fields(df.subject, df.text))
    tids = tokenize(tok, join_fields(test.subject, test.text))
    pad = tok.pad_token_id
    print(f"tokenised in {time.time() - t0:.0f}s, mean len {np.mean([len(t) for t in ids]):.0f}", flush=True)

    cache = MODELS_DIR / "xlmr_folds"
    cache.mkdir(exist_ok=True)
    oof = np.zeros(len(df))
    for k in range(fold.max() + 1):
        f = cache / f"fold{k}.npy"
        va = np.flatnonzero(fold == k)
        if not f.exists():
            tr = np.flatnonzero(fold != k)
            m = train(args.model, [ids[i] for i in tr], y[tr], pad, args.epochs, log_prefix=f"fold{k}")
            np.save(f, predict(m, [ids[i] for i in va], pad))
            del m
            torch.cuda.empty_cache()
        oof[va] = np.load(f)
        print(f"fold {k} F1 {metrics(y[va], oof[va], 0.5)['f1']:.4f} {time.time() - t0:.0f}s", flush=True)
    thr = pick_threshold(y, oof)

    out_dir = MODELS_DIR / "xlmr"
    if not (out_dir / "config.json").exists():
        m = train(args.model, ids, y, pad, args.epochs, log_prefix="full")
        m.save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
    else:
        m = AutoModelForSequenceClassification.from_pretrained(out_dir).cuda()
    st = predict(m, tids, pad)
    (out_dir / "threshold.json").write_text(json.dumps({"threshold": thr, "max_len": MAX_LEN}))

    res = {"model": args.model, "epochs": args.epochs, "max_len": MAX_LEN, "threshold": thr,
           "cv": {"default": metrics(y, oof, 0.5), "tuned": metrics(y, oof, thr)},
           "test": {"default": metrics(yt, st, 0.5), "tuned": metrics(yt, st, thr)}}
    pd.DataFrame({"file": df.file, "label": y, "fold": fold, "oof_xlmr": oof}).to_parquet(
        MODELS_DIR / "oof_xlmr.parquet", index=False)
    pd.DataFrame({"file": test.file, "label": yt, "test_xlmr": st}).to_parquet(
        MODELS_DIR / "test_xlmr.parquet", index=False)
    (REPORTS / "xlmr_results.json").write_text(json.dumps(res, indent=2))
    for part in ("cv", "test"):
        for k in ("default", "tuned"):
            mm = res[part][k]
            print(part, k, {x: round(v, 4) for x, v in mm.items() if isinstance(v, float)}, mm["cm"])
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
