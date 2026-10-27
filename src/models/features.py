"""Text normalisation and TF-IDF feature extraction shared by training and serving."""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

MAX_CHARS = 6000

_URL = re.compile(r"(https?://|www\.)\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_NUM = re.compile(r"\d+")
_PUNCT = re.compile(r"([^\w\s])")
_WS = re.compile(r"\s+")
_ZW = re.compile("[\u200b-\u200f\u00ad\u2060\ufeff]")


def normalize(text: str) -> str:
    """Bring raw MIME text and the pre-tokenised Enron/Ling-Spam text to one form.

    Enron and Ling-Spam ship lower-cased with spaces around punctuation; raw emails do
    not. Without this the char n-grams would learn the corpus formatting, not spam.
    """
    s = _ZW.sub("", text)[:MAX_CHARS].lower()
    s = _URL.sub(" urltoken ", s)
    s = _EMAIL.sub(" emailtoken ", s)
    s = _NUM.sub("0", s)
    s = _PUNCT.sub(r" \1 ", s)
    return _WS.sub(" ", s).strip()


def join_fields(subject, text) -> list[str]:
    return [normalize(f"{s} \n {t}") for s, t in zip(subject, text)]


def make_vectorizer() -> FeatureUnion:
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.9,
        max_features=200_000, sublinear_tf=True, token_pattern=r"(?u)\b\w+\b|[^\w\s]",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=300_000,
        sublinear_tf=True, dtype=np.float32,
    )
    return FeatureUnion([("word", word), ("char", char)])
