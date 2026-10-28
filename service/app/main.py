"""Antispam microservice. Run from the project root:

    uvicorn service.app.main:app --port 7860
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .backends import load_from_env  # noqa: E402
from .category import CategoryModel  # noqa: E402
from .schemas import BatchIn, BatchOut, CategoryOut, Health, ModelInfo, Prediction, TextIn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("antispam")
STATIC = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = load_from_env()
        app.state.category = CategoryModel()
        app.state.category.try_load()
        m = app.state.model
        log.info("spam model %s (%s), threshold %.4f", m.name, m.backend, m.threshold)
        yield

    app = FastAPI(
        title="Antispam API",
        version="1.0.0",
        description="Email spam filter: TF-IDF + LinearSVC with a threshold tuned for few false positives.",
        lifespan=lifespan,
    )

    def run(texts: list[str]) -> list[Prediction]:
        proba, thr = app.state.model.predict(texts)
        return [Prediction(is_spam=bool(p >= thr), proba=round(float(p), 6), threshold=round(thr, 6))
                for p in proba]

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health", response_model=Health)
    def health():
        m, cat = app.state.model, app.state.category
        return Health(status="ok", model=ModelInfo(name=m.name, backend=m.backend, threshold=m.threshold),
                      category_loaded=cat.loaded, category_error=cat.error)

    @app.post("/predict", response_model=Prediction)
    def predict(req: TextIn):
        return run([req.text])[0]

    @app.post("/predict_batch", response_model=BatchOut)
    def predict_batch(req: BatchIn):
        return BatchOut(results=run(req.texts))

    @app.post("/category", response_model=CategoryOut, responses={503: {"description": "category model not loaded"}})
    def category(req: TextIn):
        cat = app.state.category
        if not cat.try_load():
            raise HTTPException(503, "category model not loaded")
        return cat.predict(req.text)

    return app


app = create_app()
