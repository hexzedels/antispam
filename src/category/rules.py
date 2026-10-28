"""Keyword rules that add a fixed bonus to the zero-shot probability of a class.

Rules only nudge the NLI scores, they never decide alone: each pattern is precise but
fires on a minority of emails. Patterns run on the lower-cased subject + body head.
"""
from __future__ import annotations

import re

import numpy as np

from .common import CLASSES

BONUS = 0.35

RULES = {
    "phishing": [
        r"\b(verify|confirm|update|validate|reactivate|restore)\s+(your\s+)?"
        r"(account|identity|billing|password|login|information|details)",
        r"\baccount\s+(has\s+been\s+|will\s+be\s+|is\s+)?(suspended|locked|limited|disabled|"
        r"closed|restricted|frozen)",
        r"\b(unauthori[sz]ed|unusual|suspicious)\s+(access|activity|login|sign[- ]?in|transaction)",
        r"\b(security|fraud)\s+(alert|notice|measure|department)",
    ],
    "dating": [
        r"\b(singles?|dating|date\s+tonight|soul\s?mate|horny|xxx|porn\w*|nude|naked|"
        r"lonely\s+(wives|women|housewives)|hot\s+(girls|babes|chicks)|webcam\s+girls|"
        r"adult\s+(site|content|videos?|movies?|entertainment)|teen\s+(sex|girls))\b",
        r"знакомств|интим",
    ],
    "job": [
        r"\b(job\s+(offer|opening|vacanc\w+|opportunit\w+|position)|we\s+are\s+hiring|"
        r"now\s+hiring|send\s+(us\s+)?your\s+(resume|cv)|work\s+(from|at)\s+home|"
        r"home[- ]based\s+(business|job|work)|postdoc\w*|faculty\s+position|"
        r"(assistant|associate)\s+professor|lectureship|applications\s+are\s+invited)\b",
        r"ваканси|работа\s+на\s+дому|подработк|заработ",
    ],
    "finance": [
        r"\b(mortgage|refinanc\w+|loan|debt|credit\s+(card|score|report)|stock|shares|"
        r"invest\w*|lottery|you\s+have\s+won|winner|prize|inheritance|next\s+of\s+kin|"
        r"beneficiary|million\s+(us\s+)?dollars|bank\s+account|wire\s+transfer|"
        r"casino|forex|bitcoin|crypto\w*)\b",
        r"кредит|займ|инвести|выигр|наследств|казино",
    ],
    "ads": [
        r"\b(viagra|cialis|pharmacy|prescription|meds|pills?|weight\s+loss|herbal|"
        r"rolex|replica|software|oem|toner|cartridges?|free\s+shipping|"
        r"(\d+|huge)\s*%\s*off|special\s+offer|order\s+now|buy\s+now|discount)\b",
        r"скидк|акци[яи]|распродаж|купить|закажите",
    ],
    "newsletter": [
        r"\b(newsletter|mailing\s+list|digest|weekly\s+(update|news)|"
        r"you\s+are\s+receiving\s+this|this\s+(is\s+an\s+)?automated|do\s+not\s+reply|"
        r"call\s+for\s+papers|conference|workshop|announcement)\b",
        r"рассылк|уведомлени",
    ],
}
_COMPILED = {c: [re.compile(p, re.I) for p in ps] for c, ps in RULES.items()}


def rule_hits(texts: list[str]) -> np.ndarray:
    """Binary matrix (n, len(CLASSES)): 1 if any pattern of the class matches."""
    out = np.zeros((len(texts), len(CLASSES)), dtype=np.float32)
    for j, c in enumerate(CLASSES):
        for rx in _COMPILED.get(c, []):
            out[:, j] = np.maximum(out[:, j], [1.0 if rx.search(t) else 0.0 for t in texts])
    return out
