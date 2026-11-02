"""Startup refusal with a placeholder/weak JWT secret (F02)."""

import importlib
import sys

import pytest

import config
from config import (
    INSECURE_DEFAULT_JWT_SECRET,
    InsecureConfigurationError,
    get_settings,
    validate_security_settings,
)


@pytest.fixture
def no_bypass(monkeypatch):
    monkeypatch.delenv("ALLOW_INSECURE_DEV_SECRET", raising=False)
    monkeypatch.delenv("ENV_SECRETS", raising=False)


def test_default_secret_refused(monkeypatch, no_bypass):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    with pytest.raises(InsecureConfigurationError) as exc:
        validate_security_settings(get_settings())
    assert INSECURE_DEFAULT_JWT_SECRET not in str(exc.value)


def test_short_secret_refused_without_leaking(monkeypatch, no_bypass):
    monkeypatch.setenv("JWT_SECRET_KEY", "shortsecret123")
    with pytest.raises(InsecureConfigurationError) as exc:
        validate_security_settings(get_settings())
    assert "shortsecret123" not in str(exc.value)


def test_strong_secret_accepted(monkeypatch, no_bypass):
    monkeypatch.setenv("JWT_SECRET_KEY", "x" * 40)
    validate_security_settings(get_settings())


def test_bypass_allows_default(monkeypatch):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("ENV_SECRETS", raising=False)
    monkeypatch.setenv("ALLOW_INSECURE_DEV_SECRET", "1")
    validate_security_settings(get_settings())


def test_app_import_refuses_default_secret(monkeypatch, no_bypass):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    sys.modules.pop("main", None)
    try:
        with pytest.raises(InsecureConfigurationError):
            importlib.import_module("main")
    finally:
        sys.modules.pop("main", None)
    assert config  # module stays importable on its own
