"""Optional category model. Loaded lazily from CATEGORY_MODULE (default src.category.predict)
once its artifacts exist in CATEGORY_DIR (default models/category/). Until then /category
answers 503.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import os
import threading
from typing import Any, Callable

from .backends import _resolve

log = logging.getLogger(__name__)


class CategoryModel:
    def __init__(self) -> None:
        self.module = os.getenv("CATEGORY_MODULE", "src.category.predict")
        self.dir = _resolve(os.getenv("CATEGORY_DIR", "models/category"))
        self._predict: Callable[[list[str]], Any] | None = None
        self._lock = threading.Lock()
        self._failed_mtime: float | None = None
        self.error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._predict is not None

    def _artifacts_mtime(self) -> float | None:
        if not self.dir.is_dir() or not any(self.dir.iterdir()):
            return None
        return self.dir.stat().st_mtime

    def try_load(self) -> bool:
        if self._predict is not None:
            return True
        mtime = self._artifacts_mtime()
        # Do not retry a broken load on every request, only after the artifacts change.
        if mtime is None or mtime == self._failed_mtime:
            return False
        with self._lock:
            if self._predict is not None:
                return True
            try:
                mod = importlib.import_module(self.module)
                state = mod.load() if hasattr(mod, "load") else None
                takes_state = state is not None and len(inspect.signature(mod.predict).parameters) >= 2
                self._predict = (lambda t: mod.predict(state, t)) if takes_state else mod.predict
                self.error = None
                log.info("category model loaded from %s", self.module)
            except Exception as e:  # keep the spam endpoints alive whatever happens here
                self._failed_mtime = mtime
                self.error = f"{type(e).__name__}: {e}"
                log.warning("category model failed to load: %s", self.error)
        return self._predict is not None

    def predict(self, text: str) -> dict:
        out = self._predict([text])
        return to_response(out[0] if isinstance(out, (list, tuple)) else out)


def to_response(r: Any) -> dict:
    """Accept {"category", "scores"} dicts or (label, scores) pairs."""
    if isinstance(r, dict):
        label, scores = r["category"], r.get("scores", {})
    else:
        label, scores = r
    return {"category": str(label), "scores": {str(k): float(v) for k, v in dict(scores).items()}}
