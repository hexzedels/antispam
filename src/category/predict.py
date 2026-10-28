"""Category inference used by the service (service/app/category.py loads this module).

load() reads models/category/model.joblib written by src/category/train.py;
predict(state, texts) returns one {"category", "scores"} dict per text, with Russian
class names as in common.CLASS_RU. The artifact holds a TF-IDF student, an e5 student
or both (then their probabilities are averaged).
"""
from __future__ import annotations

import joblib
import numpy as np

from .common import CAT_MODELS, CLASS_RU, prepare

MODEL_NAME = "category"
E5_MAX_LEN = 256


def load(path=None) -> dict:
    art = joblib.load(path or CAT_MODELS / "model.joblib")
    if "e5" in art["models"]:
        # int8 ONNX export of the encoder (src/category/export_e5.py), no torch needed.
        from .export_e5 import E5Onnx

        enc = E5Onnx(max_len=E5_MAX_LEN)
        art["encode"] = enc.encode
    return art


def predict(state: dict, texts: list[str]) -> list[dict]:
    docs = [prepare("", t) for t in texts]
    parts = []
    if "tfidf" in state["models"]:
        parts.append(state["models"]["tfidf"].predict_proba(docs))
    if "e5" in state["models"]:
        parts.append(state["models"]["e5"].predict_proba(state["encode"](docs)))
    proba = np.mean(parts, axis=0)
    classes = state["classes"]
    out = []
    for p in proba:
        scores = {CLASS_RU[c]: round(float(v), 4) for c, v in sorted(zip(classes, p), key=lambda x: -x[1])}
        out.append({"category": CLASS_RU[classes[int(p.argmax())]], "scores": scores})
    return out
