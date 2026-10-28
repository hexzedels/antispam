"""Category taxonomy and text preparation shared by weak labelling, training and serving."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
CAT_DATA = ROOT / "data" / "category"
CAT_MODELS = ROOT / "models" / "category"

# Internal ids (stable, ASCII) -> Russian display names used in the API and the report.
CLASSES = ["ads", "job", "dating", "finance", "phishing", "newsletter", "other"]
CLASS_RU = {
    "ads": "реклама",
    "job": "предложение работы",
    "dating": "знакомства",
    "finance": "финансы/мошенничество",
    "phishing": "фишинг",
    "newsletter": "рассылки/уведомления",
    "other": "другое",
}

# NLI hypotheses, one per class. Wording was tuned on a handful of emails outside the gold set.
HYPOTHESES = {
    "ads": "This email advertises a product, a service, software or medication for sale.",
    "job": "This email is a job offer, a vacancy or an offer to work from home.",
    "dating": "This email is about dating, meeting singles, sex or adult content.",
    "finance": "This email is about loans, mortgages, debt, investments, stocks, "
               "winning a prize or lottery, or transferring money.",
    "phishing": "This email asks the reader to verify or update an account, password "
                "or bank details.",
    "newsletter": "This email is a newsletter, a mailing list post, a digest or an "
                  "automated notification.",
    "other": "This email is a personal or business message between colleagues or friends.",
}

MAX_CHARS = 3000
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{2,}")
_ZW = re.compile("[​-‏­⁠﻿]")


def prepare(subject: str | None, text: str | None) -> str:
    """Subject + body head, whitespace collapsed. Truncation to tokens happens in the model."""
    subject = (subject or "").strip()
    body = _ZW.sub("", (text or "")[:MAX_CHARS])
    body = _NL.sub("\n", _WS.sub(" ", body)).strip()
    return f"Subject: {subject}\n{body}" if subject else body
