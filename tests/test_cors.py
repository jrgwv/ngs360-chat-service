"""CORS allowlist behavior (F07)."""

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


def _load_app(monkeypatch, origin):
    if origin is None:
        monkeypatch.delenv("CLIENT_ORIGIN", raising=False)
    else:
        monkeypatch.setenv("CLIENT_ORIGIN", origin)
    monkeypatch.delenv("ENV_SECRETS", raising=False)
    sys.modules.pop("main", None)
    from config import get_settings

    get_settings.cache_clear()
    return importlib.import_module("main").app


@pytest.fixture(autouse=True)
def _drop_main():
    yield
    sys.modules.pop("main", None)


def test_unset_origin_allows_no_cross_origin(monkeypatch):
    client = TestClient(_load_app(monkeypatch, None))
    resp = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_configured_origin_only(monkeypatch):
    client = TestClient(_load_app(monkeypatch, "https://ngs360.example"))
    ok = client.get("/api/health", headers={"Origin": "https://ngs360.example"})
    assert ok.headers["access-control-allow-origin"] == "https://ngs360.example"
    bad = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_wildcard_refused(monkeypatch):
    with pytest.raises(ValueError):
        _load_app(monkeypatch, "*")
