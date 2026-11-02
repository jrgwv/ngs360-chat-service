"""Shared pytest setup: hermetic env, no AWS, no network."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_SECRET = "unit-test-secret-0123456789abcdef-0123456789"

# Set before any app module is imported so module-level settings reads are
# deterministic and never fall through to AWS Secrets Manager.
os.environ["JWT_SECRET_KEY"] = TEST_SECRET
os.environ.pop("ENV_SECRETS", None)
os.environ.setdefault("AWS_REGION", "us-east-1")
# Dummy creds so botocore client construction never consults real profiles.
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ.pop("AWS_PROFILE", None)


@pytest.fixture(autouse=True)
def _fresh_settings():
    from config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
