"""Integration tests for SaborAI API."""

from io import BytesIO

import pytest


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ingest_invalid_file_type(api_client):
    resp = api_client.post(
        "/ingest/file",
        files={"file": ("menu.csv", BytesIO(b"a,b,c"), "text/csv")},
        data={"menu_name": "Bad Format"},
    )
    assert resp.status_code == 400


@pytest.mark.requires_api
def test_ingest_text(api_client):
    resp = api_client.post(
        "/ingest/text",
        json={
            "menu_name": "Test Menu",
            "text": (
                "Pizza Margherita R$40 - molho de tomate, mussarela e manjericao. "
                "(Vegetariano | Contem gluten | Contem laticinios)\n"
                "Pizza Vegana R$44 - molho de tomate e legumes. "
                "(Vegano | Contem gluten)"
            ),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ingested"
    assert data["menu_name"] == "Test Menu"
    assert data["total_chunks"] >= 1


@pytest.mark.requires_api
def test_query_recommendation(api_client):
    api_client.post(
        "/ingest/text",
        json={
            "menu_name": "Test Menu",
            "text": "Pizza Margherita R$40. Pizza Vegana R$44. Salada Caesar R$30. Brownie R$18.",
        },
    )
    resp = api_client.post(
        "/query",
        json={"query": "Monte um combo por ate R$90 para um casal"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "agents_used" in data
    assert len(data["response"]) > 50
