"""Integration tests for SaborAI API."""

from io import BytesIO

import pytest


# ── Health ────────────────────────────────────────────────────────────────────


def test_health(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["app"] == "SaborAI"
    assert "version" in body


# ── Ingestion validation ─────────────────────────────────────────────────────


def test_ingest_invalid_file_type(api_client):
    resp = api_client.post(
        "/ingest/file",
        files={"file": ("menu.csv", BytesIO(b"a,b,c"), "text/csv")},
        data={"menu_name": "Bad Format"},
    )
    assert resp.status_code == 400


def test_ingest_file_requires_menu_name(api_client):
    """menu_name is a required form field."""
    resp = api_client.post(
        "/ingest/file",
        files={"file": ("menu.txt", BytesIO(b"Pizza R$30"), "text/plain")},
    )
    assert resp.status_code == 422  # validation error


def test_ingest_text_empty_body(api_client):
    resp = api_client.post("/ingest/text", json={})
    assert resp.status_code == 422


def test_query_empty_body(api_client):
    resp = api_client.post("/query", json={})
    assert resp.status_code == 422


# ── Ingestion + Query (require real API key) ─────────────────────────────────


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


@pytest.mark.requires_api
def test_query_returns_latency(api_client):
    """The async endpoint now returns latency_ms."""
    api_client.post(
        "/ingest/text",
        json={"menu_name": "Latency Test", "text": "Pizza R$40."},
    )
    resp = api_client.post(
        "/query",
        json={"query": "Quais opções de pizza existem?"},
    )
    assert resp.status_code == 200
    assert "latency_ms" in resp.json()
