import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from service.app.main import create_app  # noqa: E402


@pytest.fixture
def make_client(monkeypatch, tmp_path):
    """Build a fresh app; category points to an empty tmp dir unless a test overrides it."""
    def make(**env):
        env.setdefault("CATEGORY_DIR", str(tmp_path / "no_category"))
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return TestClient(create_app())
    return make


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c
