"""Shared fixtures for SaborAI tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure a dummy key exists so Settings() doesn't crash on import
os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")

import pytest  # noqa: E402


def _has_real_api_key() -> bool:
    """Return True only when a real OpenAI key is configured."""
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key) and not key.startswith("sk-test")


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked with ``requires_api`` when no real key is set."""
    if _has_real_api_key():
        return
    skip_marker = pytest.mark.skip(reason="No real OPENAI_API_KEY — skipping integration test")
    for item in items:
        if "requires_api" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)


@pytest.fixture
def sample_menu_text() -> str:
    """Minimal realistic menu text for testing."""
    return (
        "RESTAURANTE TESTE\n"
        "Cardápio Principal\n\n"
        "═══════════════════════════════\n"
        "ENTRADAS\n"
        "═══════════════════════════════\n\n"
        "Bruschetta — R$ 24,00\n"
        "Pão artesanal com tomate e manjericão. (Vegano | Sem Lactose)\n\n"
        "Camarão ao Alho — R$ 52,00\n"
        "Camarões salteados com alho e limão. (Contém frutos do mar | Contém glúten)\n\n"
        "═══════════════════════════════\n"
        "PRATOS PRINCIPAIS\n"
        "═══════════════════════════════\n\n"
        "Risoto de Funghi — R$ 68,00\n"
        "Arroz arbóreo com cogumelos e parmesão. (Vegetariano | Contém laticínios | Sem glúten)\n\n"
        "Salmão Grelhado — R$ 89,00\n"
        "Filé de salmão com ervas e purê. (Sem glúten | Contém peixe)\n\n"
        "Pizza Vegana — R$ 44,00\n"
        "Massa integral com legumes grelhados. (Vegano | Contém glúten)\n\n"
        "═══════════════════════════════\n"
        "SOBREMESAS\n"
        "═══════════════════════════════\n\n"
        "Sorbet de Manga — R$ 18,00\n"
        "Sorvete artesanal de manga. (Vegano | Sem glúten)\n\n"
        "Petit Gâteau — R$ 28,00\n"
        "Bolo de chocolate com sorvete. (Contém glúten | Contém laticínios | Contém ovo)\n"
    )


@pytest.fixture
def sample_menu_file(sample_menu_text: str, tmp_path: Path) -> Path:
    """Write sample menu to a temp .txt file and return its path."""
    path = tmp_path / "test_menu.txt"
    path.write_text(sample_menu_text, encoding="utf-8")
    return path
