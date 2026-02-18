"""
SaborAI Evals Dashboard — Streamlit app for visualizing evaluation results.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS_DIR = Path("data/eval_results")

st.set_page_config(page_title="SaborAI Evals", page_icon="🍽️", layout="wide")


# ── Data Loading ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def load_eval_results() -> list[dict]:
    """Load all eval result JSON files, sorted by timestamp."""
    if not RESULTS_DIR.exists():
        return []
    files = sorted(RESULTS_DIR.glob("eval_*.json"))
    results = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            data["_file"] = f.name
            results.append(data)
        except Exception:
            continue
    return results


def _extract_score(metric_value) -> float:
    """Extract numeric score from either a dict {'score': float} or a plain float."""
    if isinstance(metric_value, dict):
        return metric_value.get("score", 0.0)
    return float(metric_value)


def results_to_trends_df(results: list[dict]) -> pd.DataFrame:
    """Convert list of eval reports to a DataFrame for trend analysis."""
    rows = []
    for r in results:
        agg = r.get("aggregated", {})
        rows.append(
            {
                "timestamp": r.get("timestamp", ""),
                "n_cases": r.get("n_cases", 0),
                "avg_relevance": agg.get("avg_relevance", 0),
                "avg_groundedness": agg.get("avg_groundedness", 0),
                "avg_routing_accuracy": agg.get("avg_routing_accuracy", 0),
                "avg_keyword_coverage": agg.get("avg_keyword_coverage", 0),
                "avg_latency_ms": agg.get("avg_latency_ms", 0),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
    return df


def cases_to_df(report: dict) -> pd.DataFrame:
    """Extract per-case details from a single report."""
    rows = []
    for case in report.get("cases", []):
        scores = case.get("scores", {})
        rows.append(
            {
                "Case ID": case["id"],
                "Query": case["query"][:80],
                "Agents Used": ", ".join(case.get("agents_used", [])),
                "Latency (ms)": case.get("latency_ms", 0),
                "Relevance": _extract_score(scores.get("relevance", 0)),
                "Groundedness": _extract_score(scores.get("groundedness", 0)),
                "Routing Acc.": scores.get("routing_accuracy", 0),
                "Keyword Cov.": scores.get("keyword_coverage", 0),
            }
        )
    return pd.DataFrame(rows)


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("SaborAI Evals Dashboard")
st.markdown("Evaluation metrics for the multi-agent RAG system.")

results = load_eval_results()

if not results:
    st.warning(
        "No evaluation results found in `data/eval_results/`. "
        "Run an evaluation first via the API or the button below."
    )
else:
    # ── Latest Run Summary ────────────────────────────────────────────────
    latest = results[-1]
    agg = latest["aggregated"]

    st.header("Latest Run")
    st.caption(f"Timestamp: {latest['timestamp']}  |  Cases: {latest['n_cases']}")

    cols = st.columns(5)
    metrics = [
        ("Relevance", "avg_relevance"),
        ("Groundedness", "avg_groundedness"),
        ("Routing Acc.", "avg_routing_accuracy"),
        ("Keyword Cov.", "avg_keyword_coverage"),
        ("Avg Latency", "avg_latency_ms"),
    ]

    for col, (label, key) in zip(cols, metrics):
        value = agg[key]
        fmt = f"{value:.0f} ms" if "latency" in key else f"{value:.2f}"

        # Show delta vs previous run if available
        if len(results) > 1:
            prev_value = results[-2]["aggregated"][key]
            delta = value - prev_value
            delta_fmt = f"{delta:+.0f} ms" if "latency" in key else f"{delta:+.3f}"
            # For latency, lower is better
            delta_color = "inverse" if "latency" in key else "normal"
            col.metric(label, fmt, delta=delta_fmt, delta_color=delta_color)
        else:
            col.metric(label, fmt)

    # ── Per-Case Detail ───────────────────────────────────────────────────
    st.header("Per-Case Breakdown")
    case_df = cases_to_df(latest)

    st.dataframe(
        case_df.style.format(
            {
                "Latency (ms)": "{:.0f}",
                "Relevance": "{:.2f}",
                "Groundedness": "{:.2f}",
                "Routing Acc.": "{:.2f}",
                "Keyword Cov.": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Radar chart per case
    score_metrics = ["Relevance", "Groundedness", "Routing Acc.", "Keyword Cov."]
    fig_radar = go.Figure()
    for _, row in case_df.iterrows():
        fig_radar.add_trace(
            go.Scatterpolar(
                r=[row[m] for m in score_metrics],
                theta=score_metrics,
                fill="toself",
                name=row["Case ID"],
            )
        )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Score Profile per Test Case",
        height=450,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Trend Over Time ───────────────────────────────────────────────────
    if len(results) > 1:
        st.header("Trends Over Time")
        trend_df = results_to_trends_df(results)

        metric_cols = [
            "avg_relevance",
            "avg_groundedness",
            "avg_routing_accuracy",
            "avg_keyword_coverage",
        ]
        melted = trend_df.melt(
            id_vars=["timestamp"],
            value_vars=metric_cols,
            var_name="metric",
            value_name="score",
        )
        # Clean metric names for display
        melted["metric"] = (
            melted["metric"].str.replace("avg_", "").str.replace("_", " ").str.title()
        )

        col_left, col_right = st.columns(2)

        with col_left:
            fig_trend = px.line(
                melted,
                x="timestamp",
                y="score",
                color="metric",
                title="Quality Metrics Over Time",
                markers=True,
            )
            fig_trend.update_yaxes(range=[0, 1])
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, use_container_width=True)

        with col_right:
            fig_latency = px.bar(
                trend_df,
                x="timestamp",
                y="avg_latency_ms",
                title="Average Latency per Run (ms)",
            )
            fig_latency.update_layout(height=400)
            st.plotly_chart(fig_latency, use_container_width=True)

    # ── Historical Results ────────────────────────────────────────────────
    with st.expander("All Evaluation Runs"):
        for r in reversed(results):
            a = r["aggregated"]
            st.markdown(
                f"**{r['timestamp']}** — "
                f"Rel: {a['avg_relevance']:.2f} | "
                f"Gnd: {a['avg_groundedness']:.2f} | "
                f"Rout: {a['avg_routing_accuracy']:.2f} | "
                f"KW: {a['avg_keyword_coverage']:.2f} | "
                f"Lat: {a['avg_latency_ms']:.0f}ms"
            )

# ── Run New Evaluation ────────────────────────────────────────────────────

st.divider()
st.header("Run New Evaluation")
st.markdown("Trigger an evaluation run against the current ingested menus.")

if st.button("Run Evals", type="primary"):
    with st.spinner("Running evaluation suite... This may take 1-2 minutes."):
        try:
            from evals.runner import run_evals

            report = run_evals()
            st.success(f"Evaluation complete! {report['n_cases']} cases evaluated.")
            st.json(report["aggregated"])
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
