# antispam

Спам-фильтр для писем и классификатор, который раскладывает письмо по 7 категориям. Итоговый проект команды из двух человек по курсу «Применение ML в кибербезопасности», ВШЭ МИЭМ, группа МКБ251.

Демо на Hugging Face: https://huggingface.co/spaces/hexzedels/antispam, Swagger: https://hexzedels-antispam.hf.space/docs

## Модели

Спам ловит TF-IDF по словам и символьным n-граммам плюс LinearSVC. Порог 0,758 подобран на OOF-предсказаниях 5-fold CV под Precision от 0,99 и FPR до 0,5%. Тест курса в подборе не участвовал.

| оценка | порог | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|
| 5-fold CV, 40 769 писем | 0,758 | 0,9914 | 0,9638 | 0,9774 | 0,50% | 0,9987 |
| тест курса, 1 600 писем | 0,758 | 0,9901 | 0,8980 | 0,9418 | 1,50% | 0,9938 |

Категории: реклама, предложение работы, знакомства, финансы/мошенничество, фишинг, рассылки, другое. Модель усредняет вероятности двух LogReg, одна на TF-IDF, вторая на эмбеддингах multilingual-e5-small. Учили на 6 599 письмах со слабыми метками, проверяли на 94 письмах, размеченных руками: macro-F1 0,717.

LogReg, ComplementNB, e5, XLM-R, бейзлайн на SpamBase, leave-one-source-out и разбор ошибок описаны в `reports/experiments.md`.

## Запуск сервиса

Python 3.12. Энкодер e5 (113 МБ) в репозиторий не входит, его собирает `export_e5`, которому нужны torch, transformers, sentence-transformers и onnx.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt torch transformers sentence-transformers onnx
.venv/bin/python -m src.category.export_e5
docker build -f service/Dockerfile -t antispam .
docker run --rm -p 7860:7860 antispam
```

```bash
curl -s -X POST localhost:7860/predict -H 'Content-Type: application/json' \
  -d '{"text": "WIN a FREE iPhone now!!! Click here"}'
```

Ответ вида `{"is_spam": true, "proba": ..., "threshold": 0.758}`. Эндпоинты `/predict`, `/predict_batch`, `/category`, `/health`. Лимиты, коды ошибок и запуск без Docker в `service/README.md`.

## Обучение с нуля

`bash scripts/run_all.sh` скачивает корпуса и проходит весь пайплайн. Шаги, чей результат уже лежит на диске, пропускаются; один шаг можно перезапустить через `FORCE=classic`. Классические модели учатся на CPU за пару минут, а e5, zero-shot разметка и XLM-R требуют GPU с CUDA (мы считали на RTX 4070 Super).

## Данные

В репозитории данных нет, `src/data/download.py` берет их из первоисточников:

- SpamAssassin public corpus
- Enron-Spam (preprocessed)
- Ling-Spam
- SMS Spam Collection (UCI, CC BY 4.0)
- Spambase (UCI, CC BY 4.0), только для табличного бейзлайна
- тестовый набор курса, 1 600 писем `.eml`, только для проверки

После дедупликации осталось 40 769 писем, 37,4% спама. Письма, совпавшие с тестом курса, из обучения убраны. Подробнее в `reports/data.md`.

## Команда

- [@hexzedels](https://github.com/hexzedels): Data Scientist, ML-инженер
- [@Sstany](https://github.com/Sstany): аналитик, Data Engineer
