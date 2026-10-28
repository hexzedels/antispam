from __future__ import annotations

import os
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "20000"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))


def _not_blank(v: str) -> str:
    if not v.strip():
        raise ValueError("text is empty")
    return v


Text = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_CHARS), AfterValidator(_not_blank)]


class TextIn(BaseModel):
    text: Text = Field(description="Email text: subject and/or body", examples=["WIN a FREE iPhone now!!! Click here"])


class BatchIn(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=MAX_BATCH)


class Prediction(BaseModel):
    is_spam: bool
    proba: float = Field(description="Spam probability, 0..1")
    threshold: float = Field(description="Decision threshold on proba")


class BatchOut(BaseModel):
    results: list[Prediction]


class CategoryOut(BaseModel):
    category: str
    scores: dict[str, float]


class ModelInfo(BaseModel):
    name: str
    backend: str
    threshold: float


class Health(BaseModel):
    status: str
    model: ModelInfo
    category_loaded: bool
    category_error: str | None = None
