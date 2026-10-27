"""Turn raw RFC 822 / MIME messages into (subject, body text).

Only the Subject header is kept. Every other header (Received, X-Spam-*,
List-*, Return-Path, ...) is dropped because it identifies the source corpus
or the filter verdict and would leak the label.
"""
from __future__ import annotations

import email
import email.policy
import html
import re
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Filter verdict markers that some corpora leave in subject or body.
_SUBJ_TAGS = re.compile(r"(\*{3,}\s*spam\s*\*{3,}|\[spam\]|\{spam\??\})", re.I)
_BODY_LEAK = re.compile(r"^\s*(spam:|x-spam-).*$", re.I | re.M)
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")


def html_to_text(s: str) -> str:
    soup = BeautifulSoup(s, "lxml")
    for t in soup(["script", "style", "head"]):
        t.decompose()
    return soup.get_text(" ")


def _decode(part) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        p = part.get_payload()
        return p if isinstance(p, str) else ""
    charset = part.get_content_charset() or "latin-1"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("latin-1", errors="replace")


def clean_text(s: str) -> str:
    s = html.unescape(s)
    s = _BODY_LEAK.sub("", s)
    s = _WS.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    return _NL.sub("\n\n", s).strip()


def clean_subject(s: str) -> str:
    return _WS.sub(" ", _SUBJ_TAGS.sub("", s or "")).strip()


def parse_message(raw: bytes) -> tuple[str, str]:
    """Return (subject, body) for a raw MIME message."""
    # Drop the mbox "From " envelope line if present.
    if raw.startswith(b"From "):
        raw = raw.split(b"\n", 1)[1] if b"\n" in raw else b""
    msg = email.message_from_bytes(raw, policy=email.policy.compat32)
    try:
        subj = str(email.header.make_header(email.header.decode_header(msg.get("Subject", ""))))
    except Exception:
        subj = str(msg.get("Subject", ""))

    plain, htmls = [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        ctype = part.get_content_type()
        if part.get_filename():  # attachments
            continue
        if ctype == "text/plain":
            plain.append(_decode(part))
        elif ctype == "text/html":
            htmls.append(_decode(part))
    if plain and any(p.strip() for p in plain):
        body = "\n".join(plain)
        # Some "plain" parts are actually HTML.
        if re.search(r"<(html|body|table|div|p|br|font)\b", body, re.I):
            body = html_to_text(body)
    else:
        body = "\n".join(html_to_text(h) for h in htmls)
    return clean_subject(subj), clean_text(body)


def split_preprocessed(text: str) -> tuple[str, str]:
    """Enron/Ling-Spam preprocessed files: first line 'Subject: ...', then body."""
    first, _, rest = text.partition("\n")
    if first.lower().startswith("subject:"):
        return clean_subject(first[8:]), clean_text(rest)
    return "", clean_text(text)
