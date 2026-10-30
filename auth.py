"""
JWT verification for the chat service.

This service accepts the same JWT the REST API issues (shared JWT_SECRET_KEY).
It does NOT hit the database and does NOT support the REST API's API-key auth
path — the frontend chat UI sends a JWT. The verified token is forwarded to the
REST API by the agent's tools, where full authorization (including API keys) is
enforced.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt

from config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")


class AuthContext:
    """The authenticated caller plus the raw token to forward downstream."""

    def __init__(self, username: str, token: str) -> None:
        self.username = username
        self.token = token


def get_auth(token: Annotated[str, Depends(oauth2_scheme)]) -> AuthContext:
    settings = get_settings()
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # PyJWT verifies signature and exp by default; pinning ``algorithms``
        # prevents algorithm-confusion (e.g. "none" or RS/HS swaps).
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
    except jwt.PyJWTError:
        raise credentials_exception from None

    username = payload.get("sub")
    if not username:
        raise credentials_exception
    return AuthContext(username=username, token=token)


AuthDep = Annotated[AuthContext, Depends(get_auth)]
