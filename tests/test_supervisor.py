"""Unit tests for supervisor routing logic and agent orchestration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")


class TestRoutingLogic:
    """Test the routing prompt parsing — mocks the LLM to test logic only."""

    def _make_supervisor(self):
        from agents.supervisor import SupervisorAgent

        sup = SupervisorAgent()
        return sup

    @patch("agents.supervisor.ChatOpenAI")
    def test_routes_nutrition_query(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '["nutrition"]'
        mock_llm.invoke.return_value = mock_resp
        mock_llm_cls.return_value = mock_llm

        from agents.supervisor import SupervisorAgent

        sup = SupervisorAgent()
        sup.llm = mock_llm
        result = sup._route("Quais pratos são veganos?")
        assert result == ["nutrition"]

    @patch("agents.supervisor.ChatOpenAI")
    def test_routes_multi_agent_query(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '["nutrition", "quality"]'
        mock_llm.invoke.return_value = mock_resp
        mock_llm_cls.return_value = mock_llm

        from agents.supervisor import SupervisorAgent

        sup = SupervisorAgent()
        sup.llm = mock_llm
        result = sup._route("Opções sem glúten e avalie as descrições")
        assert set(result) == {"nutrition", "quality"}

    @patch("agents.supervisor.ChatOpenAI")
    def test_handles_markdown_fenced_json(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '```json\n["recommendation"]\n```'
        mock_llm.invoke.return_value = mock_resp
        mock_llm_cls.return_value = mock_llm

        from agents.supervisor import SupervisorAgent

        sup = SupervisorAgent()
        sup.llm = mock_llm
        result = sup._route("Monte um combo por R$60")
        assert result == ["recommendation"]

    @patch("agents.supervisor.ChatOpenAI")
    def test_defaults_on_parse_failure(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "invalid json!!!"
        mock_llm.invoke.return_value = mock_resp
        mock_llm_cls.return_value = mock_llm

        from agents.supervisor import SupervisorAgent

        sup = SupervisorAgent()
        sup.llm = mock_llm
        result = sup._route("anything")
        assert result == ["recommendation"]

    @patch("agents.supervisor.ChatOpenAI")
    def test_filters_unknown_agents(self, mock_llm_cls):
        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = '["nutrition", "unknown_agent", "quality"]'
        mock_llm.invoke.return_value = mock_resp
        mock_llm_cls.return_value = mock_llm

        from agents.supervisor import SupervisorAgent

        sup = SupervisorAgent()
        sup.llm = mock_llm
        result = sup._route("test")
        assert result == ["nutrition", "quality"]


class TestSupervisorRun:
    """Test the full run flow with mocked agents."""

    @patch("agents.supervisor.ChatOpenAI")
    def test_run_returns_expected_structure(self, mock_llm_cls):
        mock_llm = MagicMock()

        # Route response
        route_resp = MagicMock()
        route_resp.content = '["recommendation"]'

        # Consolidation response
        consolidate_resp = MagicMock()
        consolidate_resp.content = "Aqui está seu combo ideal!"

        mock_llm.invoke.side_effect = [route_resp, consolidate_resp]
        mock_llm_cls.return_value = mock_llm

        from agents.supervisor import SupervisorAgent

        sup = SupervisorAgent()
        sup.llm = mock_llm

        # Mock the recommendation agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Combo: Pizza + Salada = R$70"
        sup.agents["recommendation"] = mock_agent

        result = sup.run("Monte um combo por R$80")
        assert result["query"] == "Monte um combo por R$80"
        assert result["agents_used"] == ["recommendation"]
        assert "response" in result
        assert "latency_ms" in result
        assert result["latency_ms"] > 0

    @patch("agents.supervisor.ChatOpenAI")
    def test_run_handles_agent_failure(self, mock_llm_cls):
        mock_llm = MagicMock()

        route_resp = MagicMock()
        route_resp.content = '["nutrition"]'

        consolidate_resp = MagicMock()
        consolidate_resp.content = "Houve um erro ao consultar o agente."

        mock_llm.invoke.side_effect = [route_resp, consolidate_resp]
        mock_llm_cls.return_value = mock_llm

        from agents.supervisor import SupervisorAgent

        sup = SupervisorAgent()
        sup.llm = mock_llm

        # Mock agent that raises
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("ChromaDB not initialized")
        sup.agents["nutrition"] = mock_agent

        result = sup.run("Pratos veganos?")
        assert "Erro no agente nutrition" in result["agent_outputs"]["nutrition"]
        assert "response" in result  # still consolidates
