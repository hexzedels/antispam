# Smoke-тест сервиса в Docker

Образ собран командой `docker build -f service/Dockerfile -t antispam .` из папки final-team, запущен `docker run -p 7860:7860 antispam`. Ниже реальные ответы сервиса, запросы записаны в виде curl. Отчет пишет `scripts/smoke_test.py`.

## GET /health

```
$ curl -s http://localhost:7860/health
{"status":"ok","model":{"name":"classic_linsvc","backend":"classic","threshold":0.7579938865723634},"category_loaded":true,"category_error":null}
```

## POST /predict на 5 письмах

| # | что за письмо | is_spam | proba | порог | время ответа, мс |
|---|---|---|---|---|---|
| 1 | spam, en | True | 1.0000 | 0.7580 | 5 |
| 2 | spam, en | True | 1.0000 | 0.7580 | 3 |
| 3 | ham, en | False | 0.0010 | 0.7580 | 3 |
| 4 | ham, en | False | 0.0522 | 0.7580 | 3 |
| 5 | spam, ru | True | 0.8487 | 0.7580 | 3 |

Верно классифицировано 5 из 5.

## POST /category на 5 письмах

| # | ожидаемая категория | ответ | две лучшие оценки |
|---|---|---|---|
| 1 | фишинг | фишинг | фишинг 0.98, реклама 0.01 |
| 2 | предложение работы | предложение работы | предложение работы 0.95, другое 0.02 |
| 3 | знакомства | знакомства | знакомства 1.00, другое 0.00 |
| 4 | финансы/мошенничество | финансы/мошенничество | финансы/мошенничество 0.95, фишинг 0.04 |
| 5 | другое | другое | другое 0.87, рассылки/уведомления 0.05 |

Совпало 5 из 5.

Размер образа: 663 MB (python:3.12-slim, numpy/scipy/scikit-learn, onnxruntime, модель спама 6 МБ, модель категорий с энкодером e5 в int8 ONNX 130 МБ).

## POST /predict_batch

```
$ curl -s -X POST http://localhost:7860/predict_batch -H 'Content-Type: application/json' -d '{"texts": ["URGENT: Your account has been selected for a $5,000 cash rew", "Hi Anna,\n\nthanks for sending the slides. The evaluation sect"]}'
{"results":[{"is_spam":true,"proba":0.999522,"threshold":0.757994},{"is_spam":false,"proba":0.00908,"threshold":0.757994}]}
```

## Пустой текст (422)

```
$ curl -s -w '\nHTTP %{http_code}' -X POST http://localhost:7860/predict -H 'Content-Type: application/json' -d '{"text": "  "}'
{"detail":[{"type":"value_error","loc":["body","text"],"msg":"Value error, text is empty","input":"  ","ctx":{"error":{}}}]}
HTTP 422
```

## Полные запросы и ответы /predict

### 1. spam, en

```
$ curl -s -X POST http://localhost:7860/predict -H 'Content-Type: application/json' -d '{"text": "URGENT: Your account has been selected for a $5,000 cash reward! Click here http://claim-reward.example.biz to claim your prize now. Limited time offer, act now! Unsubscribe here."}'
{"is_spam":true,"proba":0.999999,"threshold":0.757994}
```

### 2. spam, en

```
$ curl -s -X POST http://localhost:7860/predict -H 'Content-Type: application/json' -d '{"text": "Cheap V1agra and Cialis!!! Best prices online, no prescription needed. 80% OFF only today. Order now at www.best-pharmacy.example.com"}'
{"is_spam":true,"proba":0.999998,"threshold":0.757994}
```

### 3. ham, en

```
$ curl -s -X POST http://localhost:7860/predict -H 'Content-Type: application/json' -d '{"text": "Hi Anna,\n\nthanks for sending the slides. The evaluation section looks good, I only left two comments on the threshold part. Can we move our call to Thursday at 3 pm?\n\nBest,\nIvan"}'
{"is_spam":false,"proba":0.001049,"threshold":0.757994}
```

### 4. ham, en

```
$ curl -s -X POST http://localhost:7860/predict -H 'Content-Type: application/json' -d '{"text": "Subject: Re: build failure on master\n\nI reverted the last commit to the parser, the nightly build is green again. Will look into the test flakiness tomorrow morning."}'
{"is_spam":false,"proba":0.052185,"threshold":0.757994}
```

### 5. spam, ru

```
$ curl -s -X POST http://localhost:7860/predict -H 'Content-Type: application/json' -d '{"text": "Поздравляем! Вы выиграли 500 000 рублей в нашей акции. Для получения выигрыша перейдите по ссылке http://priz-online.example.ru и оплатите комиссию 990 руб. Предложение действует только сегодня!"}'
{"is_spam":true,"proba":0.848702,"threshold":0.757994}
```

## Полные запросы и ответы /category

```
$ curl -s -X POST http://localhost:7860/category -H 'Content-Type: application/json' -d '{"text": "Dear PayPal member, your account has been limited. Please verify your billing information at http://paypal-secure.example.com within 24 hours."}'
{"category":"фишинг","scores":{"фишинг":0.9753,"реклама":0.0054,"финансы/мошенничество":0.0053,"знакомства":0.0045,"рассылки/уведомления":0.0044,"другое":0.0032,"предложение работы":0.0018}}
```

```
$ curl -s -X POST http://localhost:7860/category -H 'Content-Type: application/json' -d '{"text": "We are hiring! Work from home and earn $5000 a month, no experience required. Reply with your CV today."}'
{"category":"предложение работы","scores":{"предложение работы":0.9487,"другое":0.0215,"финансы/мошенничество":0.0124,"реклама":0.0062,"фишинг":0.0052,"рассылки/уведомления":0.0034,"знакомства":0.0026}}
```

```
$ curl -s -X POST http://localhost:7860/category -H 'Content-Type: application/json' -d '{"text": "Hot singles in your area want to meet you tonight. Click to see their photos."}'
{"category":"знакомства","scores":{"знакомства":0.9986,"другое":0.0007,"предложение работы":0.0003,"фишинг":0.0001,"рассылки/уведомления":0.0001,"реклама":0.0001,"финансы/мошенничество":0.0001}}
```

```
$ curl -s -X POST http://localhost:7860/category -H 'Content-Type: application/json' -d '{"text": "Refinance your mortgage today at 2.9% APR, approval in minutes, no credit check."}'
{"category":"финансы/мошенничество","scores":{"финансы/мошенничество":0.946,"фишинг":0.0416,"другое":0.0037,"рассылки/уведомления":0.003,"предложение работы":0.0024,"реклама":0.0021,"знакомства":0.0013}}
```

```
$ curl -s -X POST http://localhost:7860/category -H 'Content-Type: application/json' -d '{"text": "Hi Anna, thanks for the slides. Can we move our call to Thursday at 3 pm? Best, Ivan"}'
{"category":"другое","scores":{"другое":0.874,"рассылки/уведомления":0.0491,"предложение работы":0.0275,"реклама":0.0179,"знакомства":0.0128,"финансы/мошенничество":0.0115,"фишинг":0.0072}}
```
