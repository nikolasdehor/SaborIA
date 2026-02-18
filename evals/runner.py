"""
Eval framework for SaborAI.

Metrics evaluated per test case:
  - relevance      : does the answer address the question? (LLM-as-judge)
  - groundedness   : is the answer grounded in the retrieved context? (LLM-as-judge)
  - latency_ms     : wall-clock time in milliseconds
  - agent_routing  : did the supervisor route to the expected agent(s)?

Results are stored as a structured report dict and also persisted to
data/eval_results/ as JSON for experiment tracking.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.supervisor import SupervisorAgent
from api.settings import settings

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Test cases ────────────────────────────────────────────────────────────────

DEFAULT_SUITE: list[dict] = [
    {
        "id": "nutrition_01",
        "query": "Quais pratos são adequados para veganos?",
        "expected_agents": ["nutrition"],
        "expected_keywords": ["vegano", "vegan", "sem carne", "plant"],
    },
    {
        "id": "recommendation_01",
        "query": "Monte um combo completo por até R$60 para um casal.",
        "expected_agents": ["recommendation"],
        "expected_keywords": ["entrada", "prato", "sobremesa", "R$"],
    },
    {
        "id": "quality_01",
        "query": "Avalie a qualidade das descrições do cardápio e sugira melhorias.",
        "expected_agents": ["quality"],
        "expected_keywords": ["score", "melhoria", "descrição", "conversão"],
    },
    {
        "id": "multi_01",
        "query": (
            "Preciso de opções sem glúten e também quero saber se o cardápio está bem descrito."
        ),
        "expected_agents": ["nutrition", "quality"],
        "expected_keywords": ["glúten", "score"],
    },
]

# ── Judge LLM ─────────────────────────────────────────────────────────────────

JUDGE_RELEVANCE = """You are an evaluation judge. Score the answer's RELEVANCE to
the question on a scale from 0.0 to 1.0 where:
  1.0 = fully addresses the question
  0.5 = partially addresses
  0.0 = irrelevant

Reply ONLY with a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""

JUDGE_GROUNDEDNESS = """You are an evaluation judge. Score the answer's GROUNDEDNESS
in the provided context on a scale from 0.0 to 1.0 where:
  1.0 = every claim is supported by context
  0.5 = some unsupported claims
  0.0 = hallucinated / not grounded

Reply ONLY with a JSON object: {{"score": <float>, "reason": "<one sentence>"}}"""


class ScoreResult(TypedDict):
    score: float
    reason: str


def _llm_judge(system: str, query: str, answer: str, context: str = "") -> ScoreResult:
    judge = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )
    user_content = f"Question: {query}\n\nAnswer: {answer}"
    if context:
        user_content += f"\n\nContext: {context[:2000]}"

    resp = judge.invoke([SystemMessage(content=system), HumanMessage(content=user_content)])
    raw = resp.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"score": 0.0, "reason": f"parse error: {raw[:100]}"}


# ── Runner ────────────────────────────────────────────────────────────────────


def run_evals(suite: str = "default") -> dict:
    test_cases = DEFAULT_SUITE  # extend with more suites later
    supervisor = SupervisorAgent()

    case_results = []
    for tc in test_cases:
        logger.info("Running eval case: %s", tc["id"])
        t0 = time.perf_counter()
        output = supervisor.run(tc["query"])
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = output["response"]
        agents_used = output["agents_used"]

        # Routing accuracy
        expected = set(tc["expected_agents"])
        actual = set(agents_used)
        routing_score = len(expected & actual) / max(len(expected), 1)

        # Keyword presence (simple heuristic)
        answer_lower = answer.lower()
        keyword_hits = sum(1 for kw in tc["expected_keywords"] if kw.lower() in answer_lower)
        keyword_score = keyword_hits / max(len(tc["expected_keywords"]), 1)

        # LLM-as-judge
        relevance = _llm_judge(JUDGE_RELEVANCE, tc["query"], answer)
        groundedness = _llm_judge(JUDGE_GROUNDEDNESS, tc["query"], answer)

        case_results.append(
            {
                "id": tc["id"],
                "query": tc["query"],
                "agents_used": agents_used,
                "latency_ms": round(latency_ms, 1),
                "scores": {
                    "relevance": relevance,
                    "groundedness": groundedness,
                    "routing_accuracy": round(routing_score, 2),
                    "keyword_coverage": round(keyword_score, 2),
                },
                "answer_preview": answer[:300],
            }
        )

    # Aggregate
    def _extract_score(metric_value) -> float:
        """Extract numeric score from either a dict {'score': float} or a plain float."""
        if isinstance(metric_value, dict):
            return metric_value.get("score", 0.0)
        return float(metric_value)

    aggregated = {}
    for metric in ["relevance", "groundedness", "routing_accuracy", "keyword_coverage"]:
        scores = [_extract_score(r["scores"][metric]) for r in case_results]
        aggregated[f"avg_{metric}"] = round(sum(scores) / len(scores), 3)
    aggregated["avg_latency_ms"] = round(
        sum(r["latency_ms"] for r in case_results) / len(case_results), 1
    )

    report = {
        "suite": suite,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(case_results),
        "aggregated": aggregated,
        "cases": case_results,
    }

    # Persist for experiment tracking
    out_path = RESULTS_DIR / f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Eval report saved to %s", out_path)

    return report
