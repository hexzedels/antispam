"""Spam model backends. The service only talks to `SpamModel`, so swapping the model is
a matter of env vars:

- MODEL_BACKEND=classic (default): joblib artifact from src/models/train_classic.py,
  path in MODEL_PATH (default models/final_classic.joblib).
- MODEL_BACKEND=module: any module (MODEL_MODULE, default src.models.predict) exposing
  load() and predict(texts) -> (probabilities, threshold).
"""
from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SpamModel:
    name: str
    backend: str
    threshold: float
    _score: Callable[[list[str]], np.ndarray]

    def predict(self, texts: Sequence[str]) -> tuple[np.ndarray, float]:
        proba = np.asarray(self._score(list(texts)), dtype=float).reshape(-1)
        return proba, self.threshold


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_classic(path: str) -> SpamModel:
    import joblib

    from src.models.features import join_fields

    art = joblib.load(_resolve(path))
    pipe, cal = art["pipeline"], art.get("calibrator")
    if cal is None and art.get("score") != "proba":
        raise ValueError(f"{path}: decision-score model without a calibrator, cannot return a probability")

    def score(texts: list[str]) -> np.ndarray:
        docs = join_fields([""] * len(texts), texts)
        if cal is None:
            return pipe.predict_proba(docs)[:, 1]
        return cal.predict_proba(pipe.decision_function(docs).reshape(-1, 1))[:, 1]

    thr = art["threshold_proba"] if cal is not None else art["threshold"]
    return SpamModel(name=f"classic_{art.get('name', Path(path).stem)}", backend="classic",
                     threshold=float(thr), _score=score)


def load_module(module: str) -> SpamModel:
    mod = importlib.import_module(module)
    state = mod.load()
    # predict(texts) or predict(state, texts), whichever the module defines.
    takes_state = len(inspect.signature(mod.predict).parameters) >= 2

    def call(texts: list[str]):
        return mod.predict(state, texts) if takes_state else mod.predict(texts)

    _, thr = call(["warm up"])
    return SpamModel(name=getattr(mod, "MODEL_NAME", module), backend="module",
                     threshold=float(thr), _score=lambda texts: call(texts)[0])


def load_from_env() -> SpamModel:
    backend = os.getenv("MODEL_BACKEND", "classic").lower()
    if backend == "classic":
        return load_classic(os.getenv("MODEL_PATH", "models/final_classic.joblib"))
    if backend == "module":
        return load_module(os.getenv("MODEL_MODULE", "src.models.predict"))
    raise ValueError(f"unknown MODEL_BACKEND={backend!r}, expected 'classic' or 'module'")
