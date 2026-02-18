"""Shared fixtures for SaborAI tests."""

import os

# Ensure a dummy key exists so Settings() doesn't crash on import
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")

import pytest  # noqa: E402


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)
