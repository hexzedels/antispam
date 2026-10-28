# Сервис антиспама (FastAPI + Docker)

Модель из `models/final_classic.joblib` (TF-IDF + LinearSVC, калибровка Платта, порог по вероятности 0,758, подробности в `reports/experiments.md`), обернутая в REST API с демо-страницей.

## Запуск локально

Все команды из папки `final-team/`.

```bash
python3 -m venv service/.venv
service/.venv/bin/pip install -r service/requirements-dev.txt
service/.venv/bin/uvicorn service.app.main:app --port 7860
```

Демо-страница: http://localhost:7860/ , документация API: http://localhost:7860/docs

Тесты:

```bash
cd service && .venv/bin/python -m pytest -q
```

## Запуск в Docker

```bash
docker build -f service/Dockerfile -t antispam .
docker run --rm -p 7860:7860 antispam
```

Контекст сборки это корень проекта, но в образ попадают только код (`service/app`, `src/`) и файлы моделей. Список задан в `service/Dockerfile.dockerignore`, данные и логи обучения в образ не идут. Образ на `python:3.12-slim`, пользователь с uid 1000, порт 7860 (так требует Hugging Face Spaces). Размер образа 663 МБ: numpy/scipy/scikit-learn, onnxruntime и модель категорий, ее файлы занимают 142 МБ (энкодер e5-small в int8 ONNX 113 МБ, токенизатор, классификаторы 13 МБ). Модель спама занимает 6,1 МБ. На CPU спам-модель тратит 2 мс на письмо при обработке по одному и 0,8 мс на письмо в пакете из 1 600 писем, модель категорий около 31 мс на письмо.

## API

| метод | путь | запрос | ответ |
|---|---|---|---|
| GET | `/` | | демо-страница |
| GET | `/health` | | `{"status", "model": {"name", "backend", "threshold"}, "category_loaded", "category_error"}` |
| POST | `/predict` | `{"text": "..."}` | `{"is_spam": true, "proba": 0.98, "threshold": 0.758}` |
| POST | `/predict_batch` | `{"texts": ["...", "..."]}` | `{"results": [{"is_spam", "proba", "threshold"}, ...]}` |
| POST | `/category` | `{"text": "..."}` | `{"category": "фишинг", "scores": {"фишинг": 0.99, ...}}`, 7 классов; если файлов модели категорий нет или они не загрузились, ответ 503 `category model not loaded` |

`proba` это вероятность спама, `is_spam = proba >= threshold`. Пустой текст, текст длиннее 20 000 символов или пакет больше 64 писем дают 422.

Пример:

```bash
curl -s -X POST localhost:7860/predict -H 'Content-Type: application/json' \
  -d '{"text": "WIN a FREE iPhone now!!! Click here"}'
```

## Настройки (переменные окружения)

| переменная | по умолчанию | смысл |
|---|---|---|
| `MODEL_BACKEND` | `classic` | `classic`: joblib-артефакт из `train_classic.py`; `module`: свой модуль с моделью |
| `MODEL_PATH` | `models/final_classic.joblib` | путь к артефакту для `classic` |
| `MODEL_MODULE` | `src.models.predict` | модуль для `module`, должен иметь `load()` и `predict(texts) -> (вероятности, порог)` (или `predict(state, texts)`, где `state` вернул `load()`) |
| `CATEGORY_MODULE` | `src.category.predict` | модуль модели категорий: `load()` и `predict(texts)`, возвращает список `{"category", "scores"}` |
| `CATEGORY_DIR` | `models/category` | папка с файлами модели категорий; если их нет или они не загрузились (например, нет энкодера `models/category/e5/`, пока не запущен `python -m src.category.export_e5`), `/category` отвечает 503 |
| `MAX_TEXT_CHARS`, `MAX_BATCH` | `20000`, `64` | ограничения на вход |

Как поменять модель на другую:

```bash
# локально
MODEL_BACKEND=module service/.venv/bin/uvicorn service.app.main:app --port 7860
# в Docker
docker run --rm -p 7860:7860 -e MODEL_BACKEND=module antispam
```

Такой модуль уже есть: `src/models/predict.py` подключает дообученный XLM-R с порогом 0,8911, подобранным на OOF. Ему нужны torch, transformers и веса в `models/xlmr/` (1,08 ГБ), в образе по умолчанию их нет, поэтому для Docker их надо добавить в `service/requirements.txt` и в белый список `service/Dockerfile.dockerignore`.

Если новой модели нужны свои файлы в `models/` или новые библиотеки, их надо добавить в `service/Dockerfile.dockerignore` (белый список) и в `service/requirements.txt`, иначе в образ они не попадут. Чтобы новая модель была по умолчанию, достаточно поменять `ENV MODEL_BACKEND` в `service/Dockerfile`.

## Hugging Face Space

`service/space/build_space.sh [папка]` собирает папку для Space: `Dockerfile` и `README.md` с заголовком `sdk: docker`, `app_port: 7860` (шаблон в `service/space/README.md`) в корне, рядом код и модель. Эту папку потом пушим в репозиторий Space.
