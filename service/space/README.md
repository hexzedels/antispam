---
title: Antispam
emoji: 📧
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Email spam filter (TF-IDF + LinearSVC), FastAPI
---

# Антиспам-фильтр

Демо командного проекта по курсу «Применение ML в кибербезопасности» (НИУ ВШЭ, МИЭМ).

Модель спама: TF-IDF по словам и символьным n-граммам + LinearSVC, порог подобран так, чтобы Precision был не ниже 0,99 и FPR не выше 0,5% на кросс-валидации.

Модель категорий: ученик на слабой разметке (zero-shot NLI + правила), среднее TF-IDF + LogReg и multilingual-e5-small + LogReg.

- `/` демо-страница
- `/docs` описание API (Swagger)
- `POST /predict` `{"text": "..."}` -> `{"is_spam", "proba", "threshold"}`
- `POST /predict_batch` `{"texts": [...]}`
- `POST /category` `{"text": "..."}` -> `{"category", "scores"}`: реклама, предложение работы, знакомства, финансы/мошенничество, фишинг, рассылки/уведомления, другое
- `GET /health`
