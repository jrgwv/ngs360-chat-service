"""
Configuration for the NGS360 Chat Service.

Resolves each value from (1) an environment variable, then (2) an AWS Secrets
Manager secret named by ENV_SECRETS — the same convention the REST API uses
(core/config.py) so secrets (JWT_SECRET_KEY) are never stored as plaintext env
properties in Elastic Beanstalk. Falls back to a default.

The LLM is Amazon Bedrock; AWS access comes from the default boto3 chain
(ECS task role / instance profile / local profile), never from config.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load a local .env (dev convenience) before settings are read.
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# Placeholder shipped as the JWT secret default. The service refuses to start
# with it (see ``validate_security_settings``).
INSECURE_DEFAULT_JWT_SECRET = "change-this-secret-key-in-production"  # nosec B105 - denied sentinel, not a credential
MIN_JWT_SECRET_LENGTH = 32


class InsecureConfigurationError(RuntimeError):
    """Raised at startup when security-critical settings are unsafe."""


def get_secret(secret_name: str, region_name: str) -> dict | None:
    """Fetch and parse a JSON secret from AWS Secrets Manager."""
    client = boto3.session.Session().client(
        service_name="secretsmanager", region_name=region_name
    )
    try:
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError:
        return None
    return json.loads(response["SecretString"].replace("\n", ""))


class Settings:
    """Env-or-SecretsManager-backed settings (mirrors the REST API pattern)."""

    def __init__(self) -> None:
        self._secret_cache: dict | None = None

    def _get(self, name: str, default: str | None = None) -> str | None:
        env_value = os.getenv(name)
        if env_value:
            return env_value
        if self._secret_cache is None:
            env_secret = os.getenv("ENV_SECRETS")
            if env_secret:
                self._secret_cache = get_secret(
                    env_secret, os.getenv("AWS_REGION", "us-east-1")
                )
        if self._secret_cache:
            value = self._secret_cache.get(name)
            if value is not None:
                return value
        return default

    # Amazon Bedrock model / inference-profile id used via the Converse API.
    @property
    def LLM_MODEL(self) -> str:
        return self._get("LLM_MODEL", default="us.anthropic.claude-sonnet-4-6")

    # Region for the Bedrock runtime client (and Secrets Manager lookups).
    @property
    def AWS_REGION(self) -> str:
        return os.getenv("AWS_REGION") or "us-east-1"

    # Shared with the REST API so its JWTs verify here.
    @property
    def JWT_SECRET_KEY(self) -> str:
        return self._get("JWT_SECRET_KEY", default=INSECURE_DEFAULT_JWT_SECRET)

    @property
    def JWT_ALGORITHM(self) -> str:
        return self._get("JWT_ALGORITHM", default="HS256")

    # REST API base used by the agent's data-lookup tools.
    @property
    def NGS360_API_URL(self) -> str:
        return self._get("NGS360_API_URL", default="http://localhost:3000/api/v1")

    # GA4GH WES base used by the workflow-run tools. Defaults to /ga4gh/wes/v1 on the same host as the REST API.
    @property
    def NGS360_WES_URL(self) -> str:
        explicit = self._get("NGS360_WES_URL")
        if explicit:
            return explicit
        from urllib.parse import urlsplit

        parts = urlsplit(self.NGS360_API_URL)
        return f"{parts.scheme}://{parts.netloc}/ga4gh/wes/v1"

    # CORS allowed origin (the frontend).
    @property
    def CLIENT_ORIGIN(self) -> str | None:
        return self._get("CLIENT_ORIGIN")

    # Saved conversations. Unset means the service keeps no history (the previous behaviour).
    @property
    def CHAT_THREADS_TABLE(self) -> str | None:
        return self._get("CHAT_THREADS_TABLE")

    # Days a conversation is kept after its last message.
    @property
    def CHAT_THREAD_TTL_DAYS(self) -> int:
        return max(1, int(self._get("CHAT_THREAD_TTL_DAYS", default="90") or 90))

    # Most conversations kept per user; starting another removes the oldest.
    @property
    def CHAT_MAX_THREADS(self) -> int:
        return max(1, int(self._get("CHAT_MAX_THREADS", default="200") or 200))


@lru_cache
def get_settings() -> Settings:
    return Settings()


def validate_security_settings(settings: Settings) -> None:
    """Refuse to run with a placeholder or weak JWT secret.

    Set ALLOW_INSECURE_DEV_SECRET=1 to bypass (local dev/test only). The secret
    value itself is never included in the error or logged.
    """
    if os.getenv("ALLOW_INSECURE_DEV_SECRET") == "1":
        return
    secret = settings.JWT_SECRET_KEY or ""
    if secret == INSECURE_DEFAULT_JWT_SECRET:
        raise InsecureConfigurationError(
            "JWT_SECRET_KEY is unset or still the placeholder default; set it "
            "(env or ENV_SECRETS) to the REST API's signing secret. For local "
            "dev only, ALLOW_INSECURE_DEV_SECRET=1 bypasses this check."
        )
    if len(secret) < MIN_JWT_SECRET_LENGTH:
        raise InsecureConfigurationError(
            f"JWT_SECRET_KEY is shorter than {MIN_JWT_SECRET_LENGTH} characters; "
            "use a high-entropy secret. For local dev only, "
            "ALLOW_INSECURE_DEV_SECRET=1 bypasses this check."
        )
