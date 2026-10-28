"""Draw the gold sample for the category model, mostly from the course test set.

Usage: python -m src.category.gold

The course test set is never used to train the spam or the category model, so its emails
can serve as an untouched evaluation set for categories. The sample is stratified by the
weak label (zero-shot + rules) so that rare classes get enough examples, plus all Russian
emails up to a cap. Writes data/category/gold.csv with an empty `label` column to fill by
hand (allowed values: CLASSES from common.py). An existing gold.csv is never overwritten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import CAT_DATA, CLASSES
from .rules import BONUS, rule_hits

SEED = 42
N_GOLD = 150
PER_CLASS_MIN = 12
N_RU = 20
# The course test set has almost no job offers or dating emails, so these classes get extra
# gold examples from the training sample (they are then excluded from category training).
EXTRA_FROM_TRAIN = {"job": 15, "dating": 10}
# The weak label finds few real job emails, so more are drawn among keyword-rule hits.
EXTRA_RULE_HITS = {"job": 12}


def weak_scores(sample: pd.DataFrame, zs: pd.DataFrame) -> np.ndarray:
    p = zs[CLASSES].to_numpy()
    return p + BONUS * rule_hits(sample.input.str.lower().tolist())


def main() -> None:
    out = CAT_DATA / "gold.csv"
    if out.exists():
        print(f"{out} exists, not overwriting")
        return
    s = pd.read_parquet(CAT_DATA / "sample.parquet")
    zs = pd.read_parquet(CAT_DATA / "zs_mdeberta.parquet")
    s["weak"] = np.array(CLASSES)[weak_scores(s, zs).argmax(1)]
    t = s[s.split == "course_test"]
    rng = np.random.default_rng(SEED)

    picked = set(t[t.lang == "ru"].sample(min(N_RU, (t.lang == "ru").sum()), random_state=SEED).id)
    for c in CLASSES:
        pool = t[(t.weak == c) & ~t.id.isin(picked)]
        picked |= set(pool.sample(min(PER_CLASS_MIN, len(pool)), random_state=SEED).id)
    rest = t[~t.id.isin(picked)]
    picked |= set(rest.id.to_numpy()[rng.permutation(len(rest))[:N_GOLD - len(picked)]])

    tr = s[s.split == "train"]
    for c, n in EXTRA_FROM_TRAIN.items():
        picked |= set(tr[tr.weak == c].sample(n, random_state=SEED).id)
    hits = rule_hits(tr.input.str.lower().tolist())
    for c, n in EXTRA_RULE_HITS.items():
        pool = tr[(hits[:, CLASSES.index(c)] > 0) & ~tr.id.isin(picked)]
        picked |= set(pool.sample(n, random_state=SEED).id)

    g = s[s.id.isin(picked)].sort_values("id")
    g = g.assign(label="", body=g.text.str.slice(0, 600).str.replace(r"\s+", " ", regex=True))
    g[["id", "label", "weak", "lang", "subject", "body"]].to_csv(out, index=False)
    print(len(g), "emails ->", out)
    print(g.weak.value_counts())


if __name__ == "__main__":
    main()
