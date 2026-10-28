import textwrap

from service.app.schemas import MAX_BATCH, MAX_TEXT_CHARS

SPAM = "Congratulations! You won a FREE $1000 gift card. Click here to claim your prize now!!! http://bit.ly/x"
HAM = "Hi Anna, thanks for the slides. Can we move our call to Thursday at 3 pm? Best, Ivan"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"]["backend"] == "classic"
    assert 0 < body["model"]["threshold"] < 1
    assert body["category_loaded"] is False


def test_index_and_docs(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Проверить на спам" in r.text
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/predict", "/predict_batch", "/category"} <= set(paths)


def test_predict_spam_and_ham(client):
    spam = client.post("/predict", json={"text": SPAM}).json()
    ham = client.post("/predict", json={"text": HAM}).json()
    assert set(spam) == {"is_spam", "proba", "threshold"}
    assert spam["is_spam"] is True and ham["is_spam"] is False
    assert 0 <= ham["proba"] < spam["proba"] <= 1
    assert spam["threshold"] == ham["threshold"]


def test_predict_batch_matches_single(client):
    r = client.post("/predict_batch", json={"texts": [SPAM, HAM]})
    assert r.status_code == 200
    res = r.json()["results"]
    assert [x["is_spam"] for x in res] == [True, False]
    single = client.post("/predict", json={"text": SPAM}).json()
    assert abs(res[0]["proba"] - single["proba"]) < 1e-6


def test_validation(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "   \n\t"}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"text": 123}).status_code == 422
    assert client.post("/predict", json={"text": "a" * (MAX_TEXT_CHARS + 1)}).status_code == 422
    assert client.post("/predict", json={"text": "a" * MAX_TEXT_CHARS}).status_code == 200
    assert client.post("/predict_batch", json={"texts": []}).status_code == 422
    assert client.post("/predict_batch", json={"texts": ["ok", " "]}).status_code == 422
    assert client.post("/predict_batch", json={"texts": ["x"] * (MAX_BATCH + 1)}).status_code == 422


def test_category_503_without_model(client):
    r = client.post("/category", json={"text": SPAM})
    assert r.status_code == 503
    assert r.json()["detail"] == "category model not loaded"


def test_category_loads_module(make_client, tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "fake_category.py").write_text(textwrap.dedent("""
        def load():
            return {"labels": ["phishing", "ads"]}

        def predict(state, texts):
            return [{"category": "phishing", "scores": {"phishing": 0.8, "ads": 0.2}} for _ in texts]
    """))
    monkeypatch.syspath_prepend(str(pkg))
    art = tmp_path / "category"
    art.mkdir()
    (art / "model.joblib").write_bytes(b"x")
    with make_client(CATEGORY_MODULE="fake_category", CATEGORY_DIR=str(art)) as c:
        assert c.get("/health").json()["category_loaded"] is True
        r = c.post("/category", json={"text": SPAM})
        assert r.status_code == 200
        assert r.json() == {"category": "phishing", "scores": {"phishing": 0.8, "ads": 0.2}}


def test_category_broken_module_keeps_service_up(make_client, tmp_path):
    art = tmp_path / "category"
    art.mkdir()
    (art / "model.joblib").write_bytes(b"x")
    with make_client(CATEGORY_MODULE="no_such_module_xyz", CATEGORY_DIR=str(art)) as c:
        assert c.post("/category", json={"text": SPAM}).status_code == 503
        h = c.get("/health").json()
        assert h["category_loaded"] is False and "ModuleNotFoundError" in h["category_error"]
        assert c.post("/predict", json={"text": SPAM}).status_code == 200


def test_module_backend(make_client, tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "fake_spam.py").write_text(textwrap.dedent("""
        MODEL_NAME = "fake"

        def load():
            pass

        def predict(texts):
            return [0.9 if "win" in t.lower() else 0.1 for t in texts], 0.5
    """))
    monkeypatch.syspath_prepend(str(pkg))
    with make_client(MODEL_BACKEND="module", MODEL_MODULE="fake_spam") as c:
        assert c.get("/health").json()["model"] == {"name": "fake", "backend": "module", "threshold": 0.5}
        res = c.post("/predict_batch", json={"texts": ["WIN now", "hello"]}).json()["results"]
        assert res == [{"is_spam": True, "proba": 0.9, "threshold": 0.5},
                       {"is_spam": False, "proba": 0.1, "threshold": 0.5}]


def test_category_real_model(make_client):
    from service.app.backends import ROOT

    art = ROOT / "models" / "category"
    if not (art / "model.joblib").exists():
        import pytest
        pytest.skip("category model not trained")
    with make_client(CATEGORY_DIR=str(art)) as c:
        r = c.post("/category", json={"text": "Dear PayPal member, your account has been limited. "
                                              "Please verify your billing information at the link below."})
        assert r.status_code == 200
        body = r.json()
        assert body["category"] in body["scores"]
        assert abs(sum(body["scores"].values()) - 1) < 1e-3
        assert c.get("/health").json()["category_loaded"] is True
