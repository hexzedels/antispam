"""Zero-shot NLI scoring of emails against the category hypotheses.

Usage: python -m src.category.zeroshot [--model NAME] [--max-len 512]

Builds a stratified sample of the training corpus (spam and ham, all sources), scores it and
the course test set with an NLI model and writes data/category/zs_<tag>.parquet with one
probability column per class (softmax over the entailment logits of the 7 hypotheses).
The course test set is only scored for the category distribution, never used for training.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .common import CAT_DATA, CLASSES, HYPOTHESES, PROC, prepare

SEED = 42
# (source, label) -> sample size; spam is oversampled because most categories live there.
SAMPLE = {
    ("enron", 1): 4000, ("enron", 0): 1800,
    ("spamassassin", 1): 925, ("spamassassin", 0): 2000,
    ("lingspam", 1): 432, ("lingspam", 0): 800,
    ("sms", 1): 606, ("sms", 0): 800,
}
MODELS = {
    "mdeberta": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "large": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
}


def build_sample() -> pd.DataFrame:
    path = CAT_DATA / "sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    d = pd.read_parquet(PROC / "train.parquet")
    parts = []
    for (src, lab), n in SAMPLE.items():
        g = d[(d.source == src) & (d.label == lab)]
        parts.append(g.sample(min(n, len(g)), random_state=SEED))
    s = pd.concat(parts)
    s = s.assign(id="tr" + s.index.astype(str), split="train")
    t = pd.read_parquet(PROC / "course_test.parquet")
    t = t.assign(id="te" + t.index.astype(str), split="course_test")
    out = pd.concat([s, t], ignore_index=True)
    out["input"] = [prepare(a, b) for a, b in zip(out.subject, out.text)]
    out = out[["id", "split", "source", "label", "lang", "subject", "text", "input"]]
    CAT_DATA.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def entail_index(model) -> int:
    for k, v in model.config.id2label.items():
        if v.lower().startswith("entail"):
            return int(k)
    raise ValueError(model.config.id2label)


@torch.inference_mode()
def score(texts: list[str], name: str, max_len: int, batch: int) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float16).cuda().eval()
    ent = entail_index(model)
    hyps = [HYPOTHESES[c] for c in CLASSES]
    k = len(hyps)
    # Tokenise premises once to sort by length (less padding).
    lens = np.array([len(x) for x in tok(texts, truncation=True, max_length=max_len)["input_ids"]])
    order = np.argsort(-lens)
    logits = np.zeros((len(texts), k), dtype=np.float32)
    step = max(1, batch // k)
    t0 = time.time()
    i = 0
    while i < len(order):
        idx = order[i:i + step]
        prem = [texts[j] for j in idx for _ in range(k)]
        hyp = hyps * len(idx)
        try:
            enc = tok(prem, hyp, truncation="only_first", max_length=max_len,
                      padding=True, return_tensors="pt").to("cuda")
            out = model(**enc).logits[:, ent].float().view(len(idx), k)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if step == 1:
                raise
            step = max(1, step // 2)
            print(f"OOM, premises per batch -> {step}", flush=True)
            continue
        logits[idx] = out.cpu().numpy()
        i += len(idx)
        if (i // step) % 50 == 0:
            print(f"{i}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mdeberta", choices=list(MODELS))
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=56, help="premise-hypothesis pairs per batch")
    a = ap.parse_args()
    s = build_sample()
    print(s.groupby(["split", "source", "label"]).size())
    p = score(s.input.tolist(), MODELS[a.model], a.max_len, a.batch)
    out = pd.DataFrame(p, columns=CLASSES)
    out.insert(0, "id", s.id.values)
    out.to_parquet(CAT_DATA / f"zs_{a.model}.parquet", index=False)
    print(pd.crosstab(s.split + "/" + s.label.astype(str), out[CLASSES].idxmax(1).values))


if __name__ == "__main__":
    main()
