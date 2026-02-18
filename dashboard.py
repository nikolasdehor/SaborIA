"""
SaborAI Dashboard — Streamlit app for menu analysis and evaluation visualization.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS_DIR = Path("data/eval_results")

st.set_page_config(page_title="SaborAI", page_icon="🍽️", layout="wide")

st.title("SaborAI")
st.markdown("Sistema multi-agente com RAG para analise inteligente de cardapios.")

tab_menu, tab_evals = st.tabs(["Cardapio & Consultas", "Avaliacoes (Evals)"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Menu Upload & Query
# ══════════════════════════════════════════════════════════════════════════════

with tab_menu:
    col_upload, col_query = st.columns([1, 1])

    # ── Upload / Ingest ───────────────────────────────────────────────────
    with col_upload:
        st.header("Ingerir Cardapio")
        st.markdown("Suba um arquivo PDF ou TXT de cardapio para o sistema analisar.")

        upload_method = st.radio(
            "Metodo de entrada",
            ["Upload de arquivo (PDF/TXT)", "Colar texto"],
            horizontal=True,
        )

        menu_name = st.text_input("Nome do restaurante", placeholder="Ex: Bella Terra")

        if upload_method == "Upload de arquivo (PDF/TXT)":
            uploaded_file = st.file_uploader(
                "Arraste ou selecione o cardapio",
                type=["pdf", "txt"],
            )

            can_ingest = uploaded_file and menu_name
            if st.button("Ingerir Cardapio", type="primary", disabled=not can_ingest):
                with st.spinner("Processando cardapio..."):
                    try:
                        from ingestion.pipeline import ingest_file

                        suffix = Path(uploaded_file.name).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        result = ingest_file(tmp_path, menu_name)
                        Path(tmp_path).unlink(missing_ok=True)

                        st.success(
                            f"Cardapio **{result['menu_name']}** ingerido! "
                            f"{result['total_chunks']} chunks criados."
                        )
                    except Exception as e:
                        st.error(f"Erro na ingestao: {e}")

        else:
            menu_text = st.text_area(
                "Cole o texto do cardapio",
                height=200,
                placeholder="ENTRADAS\nSalada Caesar R$25...",
            )

            if st.button("Ingerir Texto", type="primary", disabled=not menu_text or not menu_name):
                with st.spinner("Processando texto..."):
                    try:
                        from ingestion.pipeline import ingest_text

                        result = ingest_text(menu_text, menu_name)
                        st.success(
                            f"Cardapio **{result['menu_name']}** ingerido! "
                            f"{result['total_chunks']} chunks criados."
                        )
                    except Exception as e:
                        st.error(f"Erro na ingestao: {e}")

    # ── Query ─────────────────────────────────────────────────────────────
    with col_query:
        st.header("Consultar Cardapio")
        st.markdown("Faca perguntas sobre o cardapio ingerido.")

        query = st.text_area(
            "Sua pergunta",
            height=100,
            placeholder="Ex: Monte um combo vegano por ate R$60 para um casal.",
        )

        example_queries = [
            "Quais pratos sao adequados para veganos?",
            "Monte um combo completo por ate R$60 para um casal.",
            "Avalie a qualidade das descricoes do cardapio e sugira melhorias.",
            "Quais pratos nao contem gluten nem laticinios?",
        ]

        st.markdown("**Exemplos rapidos:**")
        example_cols = st.columns(2)
        for i, ex in enumerate(example_queries):
            with example_cols[i % 2]:
                if st.button(ex[:50] + "...", key=f"ex_{i}", use_container_width=True):
                    query = ex

        if st.button("Enviar Consulta", type="primary", disabled=not query):
            with st.spinner("Consultando agentes..."):
                try:
                    from agents.supervisor import SupervisorAgent

                    supervisor = SupervisorAgent()
                    result = supervisor.run(query)

                    st.subheader("Resposta")
                    st.markdown(result["response"])

                    with st.expander("Detalhes da execucao"):
                        st.markdown(f"**Agentes utilizados:** {', '.join(result['agents_used'])}")
                        for agent_name, output in result.get("agent_outputs", {}).items():
                            st.markdown(f"---\n**{agent_name.upper()}:**")
                            st.markdown(output[:500])
                except Exception as e:
                    st.error(f"Erro na consulta: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evals Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab_evals:
    # ── Data Loading ─────────────────────────────────────────────────────

    @st.cache_data(ttl=30)
    def load_eval_results() -> list[dict]:
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
        if isinstance(metric_value, dict):
            return metric_value.get("score", 0.0)
        return float(metric_value)

    def results_to_trends_df(results: list[dict]) -> pd.DataFrame:
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

    # ── Evals UI ─────────────────────────────────────────────────────────

    results = load_eval_results()

    if not results:
        st.warning(
            "Nenhum resultado de avaliacao encontrado. Rode uma avaliacao pelo botao abaixo."
        )
    else:
        latest = results[-1]
        agg = latest["aggregated"]

        st.header("Ultimo Run")
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

            if len(results) > 1:
                prev_value = results[-2]["aggregated"][key]
                delta = value - prev_value
                delta_fmt = f"{delta:+.0f} ms" if "latency" in key else f"{delta:+.3f}"
                delta_color = "inverse" if "latency" in key else "normal"
                col.metric(label, fmt, delta=delta_fmt, delta_color=delta_color)
            else:
                col.metric(label, fmt)

        # Per-Case Detail
        st.header("Breakdown por Caso")
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

        # Radar chart
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
            title="Score Profile por Test Case",
            height=450,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Trends
        if len(results) > 1:
            st.header("Tendencias")
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
                    title="Metricas de Qualidade ao Longo do Tempo",
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
                    title="Latencia Media por Run (ms)",
                )
                fig_latency.update_layout(height=400)
                st.plotly_chart(fig_latency, use_container_width=True)

        # History
        with st.expander("Historico de Runs"):
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

    # Run New Evaluation
    st.divider()
    st.header("Rodar Nova Avaliacao")

    if st.button("Run Evals", type="primary"):
        with st.spinner("Rodando suite de avaliacao... Pode levar 1-2 minutos."):
            try:
                from evals.runner import run_evals

                report = run_evals()
                st.success(f"Avaliacao completa! {report['n_cases']} casos avaliados.")
                st.json(report["aggregated"])
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Avaliacao falhou: {e}")
