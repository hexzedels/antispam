"""CPU latency of the spam models and the category model, written to reports/latency.json.

Usage: CUDA_VISIBLE_DEVICES= .venv/bin/python scripts/bench_latency.py

Batch 1 = one email per call (what an online filter does), measured on the first 100 emails
of the course test set after a warm-up call; batch 1600 = the whole test set in one call.
Torch and ONNX Runtime are limited to 2 threads, as on a free Hugging Face Space.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from src.models.features import join_fields  # noqa: E402

N = 100
THREADS = 2


def per_email_ms(fn, docs) -> float:
    fn(docs[:1])
    t0 = time.perf_counter()
    for d in docs[:N]:
        fn([d])
    return (time.perf_counter() - t0) / N * 1000


def main() -> None:
    t = pd.read_parquet(ROOT / "data" / "processed" / "course_test.parquet")
    docs = join_fields(t.subject, t.text)
    raw = [f"{a or ''}\n{b or ''}" for a, b in zip(t.subject, t.text)]
    res = {"threads": THREADS, "n_batch1": N}

    pipe = joblib.load(ROOT / "models" / "final_classic.joblib")["pipeline"]
    res["linsvc_batch1_ms"] = per_email_ms(pipe.decision_function, docs)
    t0 = time.perf_counter()
    pipe.decision_function(docs)
    res["linsvc_batch1600_ms_per_email"] = (time.perf_counter() - t0) / len(docs) * 1000

    from src.category.predict import load, predict
    state = load()
    res["category_batch1_ms"] = per_email_ms(lambda x: predict(state, x), raw)

    xlmr = ROOT / "models" / "xlmr"
    if (xlmr / "config.json").exists():
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(THREADS)
        tok = AutoTokenizer.from_pretrained(xlmr)
        model = AutoModelForSequenceClassification.from_pretrained(xlmr).eval()

        @torch.inference_mode()
        def score(x):
            return model(**tok(x, truncation=True, max_length=256, return_tensors="pt", padding=True)).logits

        res["xlmr_batch1_ms"] = per_email_ms(score, docs)
        res["xlmr_weights_mb"] = sum(f.stat().st_size for f in xlmr.iterdir()) / 2**20

    res["linsvc_artifact_mb"] = os.path.getsize(ROOT / "models" / "final_classic.joblib") / 2**20
    res["category_artifacts_mb"] = sum(f.stat().st_size for f in (ROOT / "models" / "category").rglob("*")
                                       if f.is_file()) / 2**20
    res = {k: round(v, 2) if isinstance(v, float) else v for k, v in res.items()}
    (ROOT / "reports" / "latency.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
