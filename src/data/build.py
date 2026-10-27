"""Build unified parquet datasets from data/raw.

Usage: python -m src.data.build

Outputs (data/processed/):
  train.parquet        all text sources after cleaning + dedupe (text, subject, label, source, lang)
  course_test.parquet  course test set in the same schema (never used for training)
  spambase.parquet     tabular SpamBase, side baseline only
  dedupe_stats.json    counts removed at each step
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from .emailparse import parse_message, split_preprocessed

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

NUM_PERM = 64
LSH_THRESHOLD = 0.85
SHINGLE = 5
_CYR = re.compile(r"[а-яё]", re.I)
_LAT = re.compile(r"[a-z]", re.I)
_TOK = re.compile(r"[a-z0-9]+")


def lang_of(s: str) -> str:
    c, l = len(_CYR.findall(s)), len(_LAT.findall(s))
    return "ru" if c > l else "en"


def rec(subject: str, body: str, label: int, source: str, file: str) -> dict:
    return {"subject": subject, "text": body, "label": label, "source": source, "file": file}


def load_spamassassin() -> list[dict]:
    rows = []
    for d in sorted(p for p in (RAW / "spamassassin" / "extracted").glob("*/*") if p.is_dir()):
        label = 1 if "spam" in d.name else 0
        for f in sorted(d.iterdir()):
            if f.name == "cmds":
                continue
            s, b = parse_message(f.read_bytes())
            rows.append(rec(s, b, label, "spamassassin", f"{d.name}/{f.name}"))
    return rows


def load_enron() -> list[dict]:
    rows = []
    for f in sorted((RAW / "enron" / "extracted").glob("enron*/*/*.txt")):
        label = 1 if f.parent.name == "spam" else 0
        s, b = split_preprocessed(f.read_text(encoding="latin-1"))
        rows.append(rec(s, b, label, "enron", f"{f.parts[-3]}/{f.parent.name}/{f.name}"))
    return rows


def load_lingspam() -> list[dict]:
    rows = []
    for f in sorted((RAW / "lingspam" / "extracted" / "lingspam_public" / "bare").glob("part*/*.txt")):
        label = 1 if f.name.startswith("spmsg") else 0
        s, b = split_preprocessed(f.read_text(encoding="latin-1"))
        rows.append(rec(s, b, label, "lingspam", f"{f.parent.name}/{f.name}"))
    return rows


def load_sms() -> list[dict]:
    rows = []
    lines = (RAW / "sms" / "extracted" / "SMSSpamCollection").read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        lab, _, txt = line.partition("\t")
        rows.append(rec("", txt.strip(), int(lab == "spam"), "sms", str(i)))
    return rows


def load_course_test() -> pd.DataFrame:
    rows = []
    for f in sorted((RAW / "course_test" / "emails").glob("*/*.eml")):
        s, b = parse_message(f.read_bytes())
        rows.append(rec(s, b, int(f.parent.name == "spam"), "course_test", f"{f.parent.name}/{f.name}"))
    return finalize(pd.DataFrame(rows))


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df["lang"] = (df["subject"] + " " + df["text"]).map(lang_of)
    return df[["text", "subject", "label", "source", "lang", "file"]]


def norm_tokens(subject: str, text: str) -> list[str]:
    return _TOK.findall(f"{subject} {text}".lower())


def exact_key(tokens: list[str]) -> str:
    return hashlib.md5(" ".join(tokens).encode()).hexdigest()


def minhash(tokens: list[str]) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    toks = tokens[:3000]
    sh = {" ".join(toks[i:i + SHINGLE]) for i in range(max(1, len(toks) - SHINGLE + 1))}
    m.update_batch([s.encode() for s in sh])
    return m


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    # Source order = priority when keeping one representative of a duplicate group.
    loaders = [("spamassassin", load_spamassassin), ("enron", load_enron),
               ("lingspam", load_lingspam), ("sms", load_sms)]
    frames = []
    for name, fn in loaders:
        d = pd.DataFrame(fn())
        stats.setdefault("raw", {})[name] = {"n": len(d), "spam": int(d.label.sum())}
        print(name, len(d), int(d.label.sum()))
        frames.append(d)
    df = finalize(pd.concat(frames, ignore_index=True))
    test = load_course_test()

    # Empty messages carry no text signal.
    empty = (df.subject.str.strip() + df.text.str.strip()).str.len() == 0
    stats["empty_removed"] = int(empty.sum())
    df = df[~empty].reset_index(drop=True)

    toks = [norm_tokens(s, t) for s, t in zip(df.subject, df.text)]
    ttoks = [norm_tokens(s, t) for s, t in zip(test.subject, test.text)]

    # 1) Exact dupes on normalised tokens; drop groups with conflicting labels.
    df["_key"] = [exact_key(t) for t in toks]
    conflict = df.groupby("_key").label.transform("nunique") > 1
    stats["exact_label_conflict_removed"] = int(conflict.sum())
    keep = ~conflict & ~df.duplicated("_key", keep="first")
    stats["exact_dupes_removed"] = int((~conflict & ~keep).sum())

    # 2) Near-dupes via MinHash LSH (only among survivors), plus overlap with course test.
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    test_keys = {exact_key(t) for t in ttoks}
    for j, t in enumerate(tqdm(ttoks, desc="minhash test")):
        lsh.insert(f"t{j}", minhash(t))
    near = 0
    test_overlap = 0
    for i in tqdm(range(len(df)), desc="minhash train"):
        if not keep.iat[i]:
            continue
        if df["_key"].iat[i] in test_keys:
            keep.iat[i] = False
            test_overlap += 1
            continue
        m = minhash(toks[i])
        hits = lsh.query(m)
        if any(h.startswith("t") for h in hits):
            keep.iat[i] = False
            test_overlap += 1
            continue
        if hits:
            keep.iat[i] = False
            near += 1
            continue
        lsh.insert(f"r{i}", m)
    stats["near_dupes_removed"] = near
    stats["course_test_overlap_removed"] = test_overlap
    removed_by_src = df[~keep].source.value_counts().to_dict()
    stats["removed_by_source"] = {k: int(v) for k, v in removed_by_src.items()}

    df = df[keep].drop(columns="_key").reset_index(drop=True)
    stats["final"] = {s: {"n": int(len(g)), "spam": int(g.label.sum())} for s, g in df.groupby("source")}
    stats["course_test"] = {"n": len(test), "spam": int(test.label.sum())}
    df.to_parquet(OUT / "train.parquet", index=False)
    test.to_parquet(OUT / "course_test.parquet", index=False)

    sb = pd.read_csv(RAW / "spambase" / "extracted" / "spambase.data", header=None)
    sb.columns = [f"f{i}" for i in range(sb.shape[1] - 1)] + ["label"]
    sb.to_parquet(OUT / "spambase.parquet", index=False)
    stats["spambase"] = {"n": len(sb), "spam": int(sb.label.sum()), "features": sb.shape[1] - 1}

    (OUT / "dedupe_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
