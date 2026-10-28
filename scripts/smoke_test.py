"""Smoke test of the service: build the Docker image, run it, send real requests and write
reports/service_smoke.md with the requests and answers.

Usage: python scripts/smoke_test.py [--no-build] [--url URL]
With --url the script only sends requests to an already running service (e.g. the HF Space)
and writes reports/service_smoke_<host>.md instead.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
IMAGE, NAME, PORT = "antispam", "antispam-smoke", 7860

EMAILS = [
    ("spam, en", True, "URGENT: Your account has been selected for a $5,000 cash reward! Click here "
     "http://claim-reward.example.biz to claim your prize now. Limited time offer, act now! Unsubscribe here."),
    ("spam, en", True, "Cheap V1agra and Cialis!!! Best prices online, no prescription needed. 80% OFF only "
     "today. Order now at www.best-pharmacy.example.com"),
    ("ham, en", False, "Hi Anna,\n\nthanks for sending the slides. The evaluation section looks good, I only "
     "left two comments on the threshold part. Can we move our call to Thursday at 3 pm?\n\nBest,\nIvan"),
    ("ham, en", False, "Subject: Re: build failure on master\n\nI reverted the last commit to the parser, the "
     "nightly build is green again. Will look into the test flakiness tomorrow morning."),
    ("spam, ru", True, "Поздравляем! Вы выиграли 500 000 рублей в нашей акции. Для получения выигрыша "
     "перейдите по ссылке http://priz-online.example.ru и оплатите комиссию 990 руб. Предложение "
     "действует только сегодня!"),
]
CATEGORY = [
    ("фишинг", "Dear PayPal member, your account has been limited. Please verify your billing information "
     "at http://paypal-secure.example.com within 24 hours."),
    ("предложение работы", "We are hiring! Work from home and earn $5000 a month, no experience required. "
     "Reply with your CV today."),
    ("знакомства", "Hot singles in your area want to meet you tonight. Click to see their photos."),
    ("финансы/мошенничество", "Refinance your mortgage today at 2.9% APR, approval in minutes, no credit check."),
    ("другое", "Hi Anna, thanks for the slides. Can we move our call to Thursday at 3 pm? Best, Ivan"),
]


def call(base: str, method: str, path: str, body: dict | None = None) -> tuple[int, str]:
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def curl(base: str, method: str, path: str, body: dict | None, code: bool = False) -> str:
    cmd = "curl -s" + (" -w '\\nHTTP %{http_code}'" if code else "")
    if method == "POST":
        cmd += f" -X POST {base}{path} -H 'Content-Type: application/json' -d '{json.dumps(body, ensure_ascii=False)}'"
    else:
        cmd += f" {base}{path}"
    return cmd


def run(*cmd: str) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--url")
    a = ap.parse_args()
    local = a.url is None
    base = a.url.rstrip("/") if a.url else f"http://localhost:{PORT}"
    out = ROOT / "reports" / ("service_smoke.md" if local else f"service_smoke_{urlparse(base).hostname}.md")

    if local:
        if not a.no_build:
            run("docker", "build", "-q", "-f", "service/Dockerfile", "-t", IMAGE, ".")
        subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
        run("docker", "run", "-d", "--name", NAME, "-p", f"{PORT}:{PORT}", IMAGE)
    try:
        for _ in range(120):
            try:
                if call(base, "GET", "/health")[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(1)
        lines = [f"# Smoke-тест сервиса{' в Docker' if local else ''}", ""]
        if local:
            size = run("docker", "images", IMAGE, "--format", "{{.Size}}")
            lines += ["Образ собран командой `docker build -f service/Dockerfile -t antispam .` из папки final-team, "
                      "запущен `docker run -p 7860:7860 antispam`. Ниже реальные ответы сервиса, запросы записаны "
                      "в виде curl. Отчет пишет `scripts/smoke_test.py`.", ""]
        else:
            size = None
            lines += [f"Сервис: {base}. Ниже реальные ответы, запросы записаны в виде curl. "
                      "Отчет пишет `scripts/smoke_test.py --url`.", ""]

        st, body = call(base, "GET", "/health")
        lines += ["## GET /health", "", "```", f"$ {curl(base, 'GET', '/health', None)}", body, "```", ""]

        rows, full, ok = [], [], 0
        for i, (what, expect, text) in enumerate(EMAILS, 1):
            t0 = time.perf_counter()
            _, b = call(base, "POST", "/predict", {"text": text})
            ms = (time.perf_counter() - t0) * 1000
            r = json.loads(b)
            ok += r["is_spam"] == expect
            rows.append(f"| {i} | {what} | {r['is_spam']} | {r['proba']:.4f} | {r['threshold']:.4f} | {ms:.0f} |")
            full += [f"### {i}. {what}", "", "```", f"$ {curl(base, 'POST', '/predict', {'text': text})}", b, "```", ""]
        lines += ["## POST /predict на 5 письмах", "",
                  "| # | что за письмо | is_spam | proba | порог | время ответа, мс |", "|---|---|---|---|---|---|",
                  *rows, "", f"Верно классифицировано {ok} из {len(EMAILS)}.", ""]

        crow, cfull, cok = [], [], 0
        for i, (expect, text) in enumerate(CATEGORY, 1):
            _, b = call(base, "POST", "/category", {"text": text})
            r = json.loads(b)
            top = list(r["scores"].items())[:2]
            cok += r["category"] == expect
            crow.append(f"| {i} | {expect} | {r['category']} | " + ", ".join(f"{k} {v:.2f}" for k, v in top) + " |")
            cfull += ["```", f"$ {curl(base, 'POST', '/category', {'text': text})}", b, "```", ""]
        lines += ["## POST /category на 5 письмах", "",
                  "| # | ожидаемая категория | ответ | две лучшие оценки |", "|---|---|---|---|",
                  *crow, "", f"Совпало {cok} из {len(CATEGORY)}.", ""]
        if size:
            lines += [f"Размер образа: {size.replace('MB', ' MB')} (python:3.12-slim, numpy/scipy/scikit-learn, "
                      "onnxruntime, модель спама 6 МБ, модель категорий с энкодером e5 в int8 ONNX 130 МБ).", ""]

        texts = [EMAILS[0][2][:60], EMAILS[2][2][:60]]
        _, b = call(base, "POST", "/predict_batch", {"texts": texts})
        lines += ["## POST /predict_batch", "", "```", f"$ {curl(base, 'POST', '/predict_batch', {'texts': texts})}",
                  b, "```", ""]
        st, b = call(base, "POST", "/predict", {"text": "  "})
        lines += ["## Пустой текст (422)", "", "```", f"$ {curl(base, 'POST', '/predict', {'text': '  '}, code=True)}",
                  b, f"HTTP {st}", "```", ""]
        lines += ["## Полные запросы и ответы /predict", "", *full, "## Полные запросы и ответы /category", "", *cfull]
        out.write_text("\n".join(lines))
        print(f"{out}: predict {ok}/{len(EMAILS)}, category {cok}/{len(CATEGORY)}, image {size}")
    finally:
        if local:
            subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)


if __name__ == "__main__":
    main()
