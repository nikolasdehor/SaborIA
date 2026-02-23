"""
LLM Experiment Runner — Compare models, prompts, temperatures and chunk sizes.

This script runs controlled experiments to measure how different configurations
affect the quality of multi-agent responses. Results are saved as JSON and can
be visualized in the dashboard.

Usage:
    python -m experiments.compare_models
    python -m experiments.compare_models --models gpt-4o-mini gpt-4o --temperatures 0 0.2 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from api.settings import settings

logging.basicConfig(level="INFO", format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/experiment_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Test queries for benchmarking ─────────────────────────────────────────────

BENCHMARK_QUERIES = [
    {
        "id": "routing_simple",
        "query": "Quais pratos são adequados para veganos?",
        "expected_domain": "nutrition",
        "expected_keywords": ["vegano", "vegan"],
    },
    {
        "id": "routing_multi",
        "query": "Monte um combo vegano por até R$60 e avalie a qualidade das descrições.",
        "expected_domain": "multi",
        "expected_keywords": ["combo", "R$", "qualidade"],
    },
    {
        "id": "reasoning_negation",
        "query": "Quais pratos NÃO contêm glúten nem laticínios?",
        "expected_domain": "nutrition",
        "expected_keywords": ["sem glúten", "sem laticínios"],
    },
    {
        "id": "budget_constraint",
        "query": "Quero um jantar completo (entrada + principal + sobremesa) por até R$80.",
        "expected_domain": "recommendation",
        "expected_keywords": ["entrada", "principal", "sobremesa", "R$"],
    },
]

# ── Evaluation via LLM-as-judge ───────────────────────────────────────────────

JUDGE_PROMPT = """You are an evaluation judge. Score the following answer on a
scale from 0.0 to 1.0 across these dimensions:

1. relevance: Does the answer address the question?
2. coherence: Is the answer well-structured and clear?
3. completeness: Does it cover all aspects of the query?

Reply ONLY with a JSON object:
{{"relevance": <float>, "coherence": <float>, "completeness": <float>}}"""


class ExperimentConfig(BaseModel):
    model: str
    temperature: float
    chunk_size: int = 1024
    chunk_overlap: int = 128
    retriever_k: int = 6


class ExperimentResult(BaseModel):
    config: ExperimentConfig
    query_id: str
    query: str
    answer: str
    latency_ms: float
    scores: dict[str, float]
    keyword_hits: int
    keyword_total: int


def _judge_answer(query: str, answer: str, judge_model: str = "gpt-4o-mini") -> dict[str, float]:
    """Use an LLM as judge to score answer quality."""
    judge = ChatOpenAI(
        model=judge_model,
        openai_api_key=settings.openai_api_key,
        temperature=0,
    )
    resp = judge.invoke([
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=f"Question: {query}\n\nAnswer: {answer}"),
    ])
    raw = resp.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"relevance": 0.0, "coherence": 0.0, "completeness": 0.0}


def _run_single_query(
    config: ExperimentConfig,
    benchmark: dict,
) -> ExperimentResult:
    """Run a single query with the given config and evaluate it."""
    llm = ChatOpenAI(
        model=config.model,
        openai_api_key=settings.openai_api_key,
        temperature=config.temperature,
    )

    # Simple direct query (not full pipeline — isolates LLM behavior)
    t0 = time.perf_counter()
    resp = llm.invoke([
        SystemMessage(
            content=(
                "Voce e um assistente de restaurante. Responda de forma clara "
                "e detalhada em portugues brasileiro. Use texto puro."
            )
        ),
        HumanMessage(content=benchmark["query"]),
    ])
    latency_ms = (time.perf_counter() - t0) * 1000
    answer = resp.content

    # Score
    scores = _judge_answer(benchmark["query"], answer)

    # Keyword hits
    answer_lower = answer.lower()
    hits = sum(1 for kw in benchmark["expected_keywords"] if kw.lower() in answer_lower)

    return ExperimentResult(
        config=config,
        query_id=benchmark["id"],
        query=benchmark["query"],
        answer=answer[:500],
        latency_ms=round(latency_ms, 1),
        scores=scores,
        keyword_hits=hits,
        keyword_total=len(benchmark["expected_keywords"]),
    )


def run_experiment(
    models: list[str],
    temperatures: list[float],
    chunk_sizes: list[int] | None = None,
) -> dict:
    """Run a full experiment grid and return a structured report."""
    chunk_sizes = chunk_sizes or [1024]
    configs = [
        ExperimentConfig(model=m, temperature=t, chunk_size=cs)
        for m, t, cs in product(models, temperatures, chunk_sizes)
    ]

    logger.info(
        "Running experiment: %d configs x %d queries = %d total runs",
        len(configs),
        len(BENCHMARK_QUERIES),
        len(configs) * len(BENCHMARK_QUERIES),
    )

    all_results: list[dict] = []
    for cfg in configs:
        logger.info("Config: model=%s temp=%.1f chunk=%d", cfg.model, cfg.temperature, cfg.chunk_size)
        for bm in BENCHMARK_QUERIES:
            try:
                result = _run_single_query(cfg, bm)
                all_results.append(result.model_dump())
                logger.info(
                    "  %s: rel=%.2f coh=%.2f comp=%.2f lat=%.0fms",
                    bm["id"],
                    result.scores.get("relevance", 0),
                    result.scores.get("coherence", 0),
                    result.scores.get("completeness", 0),
                    result.latency_ms,
                )
            except Exception as exc:
                logger.error("  %s: FAILED — %s", bm["id"], exc)
                all_results.append({
                    "config": cfg.model_dump(),
                    "query_id": bm["id"],
                    "error": str(exc),
                })

    # Aggregate by config
    summary: list[dict] = []
    for cfg in configs:
        cfg_results = [r for r in all_results if r.get("config") == cfg.model_dump() and "scores" in r]
        if not cfg_results:
            continue

        avg_scores = {}
        for metric in ["relevance", "coherence", "completeness"]:
            values = [r["scores"].get(metric, 0) for r in cfg_results]
            avg_scores[f"avg_{metric}"] = round(sum(values) / len(values), 3)

        latencies = [r["latency_ms"] for r in cfg_results]
        avg_scores["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)

        kw_hits = sum(r["keyword_hits"] for r in cfg_results)
        kw_total = sum(r["keyword_total"] for r in cfg_results)
        avg_scores["keyword_coverage"] = round(kw_hits / max(kw_total, 1), 3)

        summary.append({
            "config": cfg.model_dump(),
            "n_queries": len(cfg_results),
            "aggregated": avg_scores,
        })

    report = {
        "experiment": "model_comparison",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_configs": len(configs),
        "n_queries": len(BENCHMARK_QUERIES),
        "summary": summary,
        "detailed_results": all_results,
    }

    # Persist
    out_path = RESULTS_DIR / f"experiment_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Experiment report saved to %s", out_path)

    # Print comparison table
    _print_summary(summary)

    return report


def _print_summary(summary: list[dict]) -> None:
    """Print a human-readable comparison table."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Experiment Results", show_header=True, header_style="bold cyan")
        table.add_column("Model")
        table.add_column("Temp")
        table.add_column("Relevance", justify="right")
        table.add_column("Coherence", justify="right")
        table.add_column("Completeness", justify="right")
        table.add_column("KW Coverage", justify="right")
        table.add_column("Latency (ms)", justify="right")

        for s in sorted(summary, key=lambda x: -x["aggregated"]["avg_relevance"]):
            cfg = s["config"]
            agg = s["aggregated"]
            table.add_row(
                cfg["model"],
                f"{cfg['temperature']:.1f}",
                f"{agg['avg_relevance']:.3f}",
                f"{agg['avg_coherence']:.3f}",
                f"{agg['avg_completeness']:.3f}",
                f"{agg['keyword_coverage']:.3f}",
                f"{agg['avg_latency_ms']:.0f}",
            )

        console.print(table)
    except ImportError:
        for s in summary:
            cfg = s["config"]
            agg = s["aggregated"]
            print(
                f"  {cfg['model']} (temp={cfg['temperature']}): "
                f"rel={agg['avg_relevance']:.3f} "
                f"coh={agg['avg_coherence']:.3f} "
                f"comp={agg['avg_completeness']:.3f} "
                f"kw={agg['keyword_coverage']:.3f} "
                f"lat={agg['avg_latency_ms']:.0f}ms"
            )


def main():
    parser = argparse.ArgumentParser(description="Run LLM comparison experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini", "gpt-4o"],
        help="Models to compare",
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0, 0.2, 0.5],
        help="Temperature values to test",
    )
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[512, 1024],
        help="Chunk sizes to test",
    )
    args = parser.parse_args()

    run_experiment(
        models=args.models,
        temperatures=args.temperatures,
        chunk_sizes=args.chunk_sizes,
    )


if __name__ == "__main__":
    main()
