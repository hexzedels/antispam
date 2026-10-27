"""Idempotent download of the raw spam corpora into data/raw/.

Usage: python -m src.data.download  (from the final-team directory)

Every archive is downloaded once; a re-run skips files that already exist.
www2.aueb.gr serves an incomplete TLS chain (the HARICA intermediate is missing),
so we fetch that intermediate from the CA's AIA URL and verify against
certifi + intermediate instead of disabling verification.
"""
from __future__ import annotations

import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import certifi
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

SA_BASE = "https://spamassassin.apache.org/old/publiccorpus/"
SA_FILES = [
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
    "20030228_spam.tar.bz2",
    "20030228_spam_2.tar.bz2",
]
AUEB = "https://www2.aueb.gr/users/ion/data/"
ENRON_FILES = [f"enron-spam/preprocessed/enron{i}.tar.gz" for i in range(1, 7)]
LING_FILE = "lingspam_public.tar.gz"
SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SPAMBASE_URL = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
HARICA_INTERMEDIATE = "http://crt.harica.gr/HARICA-Institutional-TLS-Sub-R2.cer"
# Course test set (labelled ham/ spam/ .eml). Public Yandex folder link from the task docx.
COURSE_TEST_KEY = "https://disk.yandex.ru/d/mJ_9zG3dX3eZgg"
YANDEX_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"


def aueb_bundle() -> str:
    certs = RAW / "_certs"
    certs.mkdir(parents=True, exist_ok=True)
    bundle = certs / "bundle.pem"
    if not bundle.exists():
        der = requests.get(HARICA_INTERMEDIATE, timeout=60).content
        pem = subprocess.run(
            ["openssl", "x509", "-inform", "DER", "-outform", "PEM"],
            input=der, capture_output=True, check=True,
        ).stdout
        bundle.write_bytes(Path(certifi.where()).read_bytes() + b"\n" + pem)
    return str(bundle)


def fetch(url: str, dest: Path, verify: bool | str = True) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip {dest.relative_to(ROOT)}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"get  {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120, verify=verify) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def extract(archive: Path, out: Path) -> None:
    marker = out / f".extracted_{archive.name}"
    if marker.exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    else:
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")
    marker.touch()


def main() -> None:
    for name in SA_FILES:
        a = fetch(SA_BASE + name, RAW / "spamassassin" / "archives" / name)
        extract(a, RAW / "spamassassin" / "extracted" / name.split(".")[0])

    bundle = aueb_bundle()
    for rel in ENRON_FILES:
        name = Path(rel).name
        a = fetch(AUEB + rel, RAW / "enron" / "archives" / name, verify=bundle)
        extract(a, RAW / "enron" / "extracted")
    a = fetch(AUEB + LING_FILE, RAW / "lingspam" / LING_FILE, verify=bundle)
    extract(a, RAW / "lingspam" / "extracted")

    a = fetch(SMS_URL, RAW / "sms" / "smsspamcollection.zip")
    extract(a, RAW / "sms" / "extracted")
    a = fetch(SPAMBASE_URL, RAW / "spambase" / "spambase.zip")
    extract(a, RAW / "spambase" / "extracted")

    ct = RAW / "course_test" / "test_task_emails.tar.xz"
    if not ct.exists():
        href = json.loads(requests.get(YANDEX_API, params={"public_key": COURSE_TEST_KEY}, timeout=60).text)["href"]
        fetch(href, ct)
    extract(ct, RAW / "course_test" / "emails")


if __name__ == "__main__":
    main()
