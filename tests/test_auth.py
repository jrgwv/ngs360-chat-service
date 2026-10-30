"""JWT verification with PyJWT (F04)."""

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException

from auth import get_auth
from tests.conftest import TEST_SECRET


def _token(secret: str = TEST_SECRET, delta: timedelta = timedelta(minutes=5), **extra) -> str:
    claims = {"sub": "alice", "exp": datetime.now(timezone.utc) + delta, **extra}
    return jwt.encode(claims, secret, algorithm="HS256")


def test_valid_token_accepted():
    token = _token()
    ctx = get_auth(token)
    assert ctx.username == "alice"
    assert ctx.token == token


def test_expired_token_rejected():
    with pytest.raises(HTTPException) as exc:
        get_auth(_token(delta=timedelta(minutes=-5)))
    assert exc.value.status_code == 401


def test_wrong_secret_rejected():
    with pytest.raises(HTTPException) as exc:
        get_auth(_token(secret="some-other-secret-that-is-long-enough-123"))
    assert exc.value.status_code == 401


def test_alg_none_rejected():
    token = jwt.encode({"sub": "alice"}, key=None, algorithm="none")
    with pytest.raises(HTTPException):
        get_auth(token)


def test_missing_sub_rejected():
    token = jwt.encode(
        {"exp": datetime.now(timezone.utc) + timedelta(minutes=5)}, TEST_SECRET, algorithm="HS256"
    )
    with pytest.raises(HTTPException):
        get_auth(token)


def test_garbage_token_rejected():
    with pytest.raises(HTTPException):
        get_auth("not-a-jwt")
