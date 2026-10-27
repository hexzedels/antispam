"""XLM-R spam model for the service's module backend (MODEL_BACKEND=module).

Needs torch and transformers, which the default image does not install, and the weights in
models/xlmr/ (written by src/models/train_transformer.py). The threshold is the one picked
on OOF scores (models/xlmr/threshold.json).
"""
from __future__ import annotations

import json
from pathlib import Path

from .features import join_fields

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "xlmr"
MODEL_NAME = "xlmr"


def load() -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    meta = json.loads((MODEL_DIR / "threshold.json").read_text())
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
    return {"torch": torch, "tok": AutoTokenizer.from_pretrained(MODEL_DIR), "model": model,
            "threshold": float(meta["threshold"]), "max_len": int(meta["max_len"])}


def predict(state: dict, texts: list[str]):
    torch = state["torch"]
    docs = join_fields([""] * len(texts), texts)
    enc = state["tok"](docs, truncation=True, max_length=state["max_len"], padding=True, return_tensors="pt")
    with torch.inference_mode():
        proba = torch.softmax(state["model"](**enc).logits, -1)[:, 1].numpy()
    return proba, state["threshold"]
