"""Unit tests for agent configuration and initialization."""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")

import pytest


class TestAgentInit:
    """Test that agents initialize correctly with expected configuration."""

    def test_nutrition_agent_temperature(self):
        from agents.nutrition import NutritionAgent

        agent = NutritionAgent()
        assert agent.llm.temperature == 0

    def test_recommendation_agent_temperature(self):
        from agents.recommendation import RecommendationAgent

        agent = RecommendationAgent()
        assert agent.llm.temperature == 0.2

    def test_quality_agent_temperature(self):
        from agents.quality import QualityAgent

        agent = QualityAgent()
        assert agent.llm.temperature == 0.1

    def test_supervisor_has_all_agents(self):
        from agents.supervisor import SupervisorAgent

        sup = SupervisorAgent()
        assert set(sup.agents.keys()) == {"nutrition", "recommendation", "quality"}

    def test_all_agents_have_run_and_arun(self):
        """Every agent must expose both sync run() and async arun()."""
        from agents.nutrition import NutritionAgent
        from agents.quality import QualityAgent
        from agents.recommendation import RecommendationAgent

        for AgentCls in [NutritionAgent, RecommendationAgent, QualityAgent]:
            agent = AgentCls()
            assert callable(getattr(agent, "run", None)), f"{AgentCls.__name__} missing run()"
            assert callable(getattr(agent, "arun", None)), f"{AgentCls.__name__} missing arun()"

    def test_supervisor_has_arun(self):
        from agents.supervisor import SupervisorAgent

        sup = SupervisorAgent()
        assert callable(getattr(sup, "arun", None))


class TestSettings:
    """Test configuration loading."""

    def test_settings_defaults(self):
        from api.settings import settings

        assert settings.openai_model == "gpt-4o-mini"
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0
        assert settings.retriever_k > 0
        assert settings.log_level in ("DEBUG", "INFO", "WARNING", "ERROR")

    def test_chroma_persist_dir_is_set(self):
        from api.settings import settings

        assert settings.chroma_persist_dir
        assert "chroma" in settings.chroma_persist_dir


class TestEvalSuiteDefinition:
    """Test that the eval suite is well-formed."""

    def test_default_suite_has_cases(self):
        from evals.runner import DEFAULT_SUITE

        assert len(DEFAULT_SUITE) >= 4

    def test_each_case_has_required_fields(self):
        from evals.runner import DEFAULT_SUITE

        required = {"id", "query", "expected_agents", "expected_keywords"}
        for case in DEFAULT_SUITE:
            assert required.issubset(case.keys()), f"Case {case.get('id')} missing fields"

    def test_expected_agents_are_valid(self):
        from evals.runner import DEFAULT_SUITE

        valid_agents = {"nutrition", "recommendation", "quality"}
        for case in DEFAULT_SUITE:
            for agent in case["expected_agents"]:
                assert agent in valid_agents, f"Invalid agent '{agent}' in case {case['id']}"
