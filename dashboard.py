"""
SaborAI Dashboard — Streamlit app for menu analysis and evaluation visualization.

Run with:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load .env locally (no-op if file missing)

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

RESULTS_DIR = Path("data/eval_results")

# ── Page Config (must be the first Streamlit command) ─────────────────────────

st.set_page_config(page_title="SaborAI", page_icon="🍽️", layout="wide")

# On Streamlit Cloud, .env doesn't exist — bridge st.secrets into env vars.
if "OPENAI_API_KEY" not in os.environ:
    for _key in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        try:
            os.environ[_key] = st.secrets[_key]
        except Exception:
            pass

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-top: 0;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1D23 0%, #22262E 100%);
        border: 1px solid #2D3139;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        color: #9CA3AF !important;
        font-size: 0.85rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(59, 130, 246, 0.15);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .stButton > button {
        border-radius: 8px;
        transition: all 0.2s;
    }

    .section-sep {
        border-top: 1px solid #2D3139;
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<p class="hero-title">SaborAI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    "Sistema multi-agente com RAG para análise inteligente de cardápios"
    "</p>",
    unsafe_allow_html=True,
)

tab_menu, tab_evals = st.tabs(["🍽️ Cardápio & Consultas", "📊 Avaliações (Evals)"])

CHART_COLORS = ["#FF4B4B", "#FF8E53", "#3B82F6", "#10B981", "#A78BFA"]


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Menu Upload & Query
# ══════════════════════════════════════════════════════════════════════════════

with tab_menu:
    col_upload, col_query = st.columns([1, 1], gap="large")

    # ── Upload / Ingest ───────────────────────────────────────────────────
    with col_upload:
        st.markdown("### 📤 Ingerir Cardápio")
        st.caption("Suba um arquivo PDF ou TXT de cardápio para o sistema analisar.")

        upload_method = st.radio(
            "Método de entrada",
            ["📁 Upload de arquivo (PDF/TXT)", "📋 Colar texto"],
            horizontal=True,
        )

        menu_name = st.text_input(
            "Nome do restaurante",
            placeholder="Ex: Bella Terra",
        )

        if upload_method.startswith("📁"):
            uploaded_file = st.file_uploader(
                "Arraste ou selecione o cardápio",
                type=["pdf", "txt"],
            )

            can_ingest = uploaded_file and menu_name
            if st.button(
                "🚀 Ingerir Cardápio",
                type="primary",
                disabled=not can_ingest,
                use_container_width=True,
            ):
                with st.spinner("Processando cardápio..."):
                    try:
                        from ingestion.pipeline import ingest_file

                        suffix = Path(uploaded_file.name).suffix
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        result = ingest_file(tmp_path, menu_name)
                        Path(tmp_path).unlink(missing_ok=True)

                        st.session_state["ingested_menu"] = menu_name
                        st.success(
                            f"Cardápio **{result['menu_name']}** ingerido com sucesso! "
                            f"**{result['total_chunks']}** chunks criados."
                        )
                    except Exception as e:
                        st.error(f"Erro na ingestão: {e}")

        else:
            menu_text = st.text_area(
                "Cole o texto do cardápio",
                height=200,
                placeholder="ENTRADAS\nSalada Caesar — R$25\nBruschetta — R$18\n...",
            )

            if st.button(
                "🚀 Ingerir Texto",
                type="primary",
                disabled=not menu_text or not menu_name,
                use_container_width=True,
            ):
                with st.spinner("Processando texto..."):
                    try:
                        from ingestion.pipeline import ingest_text

                        result = ingest_text(menu_text, menu_name)
                        st.session_state["ingested_menu"] = menu_name
                        st.success(
                            f"Cardápio **{result['menu_name']}** ingerido com sucesso! "
                            f"**{result['total_chunks']}** chunks criados."
                        )
                    except Exception as e:
                        st.error(f"Erro na ingestão: {e}")

    # ── Query ─────────────────────────────────────────────────────────────
    with col_query:
        st.markdown("### 🔍 Consultar Cardápio")
        st.caption("Faça perguntas sobre o cardápio ingerido.")

        query = st.text_area(
            "Sua pergunta",
            height=100,
            placeholder="Ex: Monte um combo vegano por até R$60 para um casal.",
        )

        example_queries = [
            ("🌱", "Quais pratos são adequados para veganos?"),
            ("💰", "Monte um combo completo por até R$60 para um casal."),
            ("✨", "Avalie a qualidade das descrições e sugira melhorias."),
            ("🚫", "Quais pratos não contêm glúten nem laticínios?"),
        ]

        st.markdown("**Exemplos rápidos:**")
        example_cols = st.columns(2)
        selected_example = None
        for i, (icon, ex) in enumerate(example_queries):
            with example_cols[i % 2]:
                if st.button(
                    f"{icon} {ex[:45]}...",
                    key=f"ex_{i}",
                    use_container_width=True,
                ):
                    selected_example = ex

        send_clicked = st.button(
            "🚀 Enviar Consulta",
            type="primary",
            disabled=not query and not selected_example,
            use_container_width=True,
        )

        effective_query = selected_example or query

        if send_clicked or selected_example:
            if not effective_query:
                st.warning("Digite uma pergunta ou selecione um exemplo.")
            else:
                with st.spinner("Consultando agentes especializados..."):
                    try:
                        from agents.supervisor import SupervisorAgent

                        active_menu = st.session_state.get("ingested_menu")
                        supervisor = SupervisorAgent()
                        result = supervisor.run(effective_query, menu_name=active_menu)

                        st.markdown("---")
                        st.markdown("#### 💬 Resposta")
                        st.info(result["response"])

                        agents_used = result["agents_used"]
                        badges = " ".join(
                            f'<span class="status-badge">{a}</span>' for a in agents_used
                        )
                        st.markdown(
                            f"**Agentes utilizados:** {badges}",
                            unsafe_allow_html=True,
                        )

                        with st.expander("Ver detalhes por agente"):
                            for agent_name, output in result.get("agent_outputs", {}).items():
                                st.markdown(f"**{agent_name.upper()}**")
                                st.info(output)
                    except Exception as e:
                        st.error(f"Erro na consulta: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evals Dashboard
# ══════════════════════════════════════════════════════════════════════════════

with tab_evals:

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
                    "Caso": case["id"],
                    "Query": case["query"][:80],
                    "Agentes": ", ".join(case.get("agents_used", [])),
                    "Latência (ms)": case.get("latency_ms", 0),
                    "Relevância": _extract_score(scores.get("relevance", 0)),
                    "Fundamentação": _extract_score(scores.get("groundedness", 0)),
                    "Roteamento": scores.get("routing_accuracy", 0),
                    "Cobertura KW": scores.get("keyword_coverage", 0),
                }
            )
        return pd.DataFrame(rows)

    # ── Evals UI ─────────────────────────────────────────────────────────

    results = load_eval_results()

    if not results:
        st.info(
            "Nenhum resultado de avaliação encontrado. "
            "Ingira um cardápio e rode uma avaliação pelo botão abaixo."
        )
    else:
        latest = results[-1]
        agg = latest["aggregated"]

        st.markdown("### 🏆 Último Run")
        st.caption(f"Timestamp: {latest['timestamp']}  ·  Casos avaliados: {latest['n_cases']}")

        cols = st.columns(5)
        metrics = [
            ("Relevância", "avg_relevance", "🎯"),
            ("Fundamentação", "avg_groundedness", "📌"),
            ("Roteamento", "avg_routing_accuracy", "🔀"),
            ("Cobertura KW", "avg_keyword_coverage", "🔑"),
            ("Latência Média", "avg_latency_ms", "⚡"),
        ]

        for col, (label, key, icon) in zip(cols, metrics):
            value = agg[key]
            fmt = f"{value:.0f} ms" if "latency" in key else f"{value:.2f}"

            if len(results) > 1:
                prev_value = results[-2]["aggregated"][key]
                delta = value - prev_value
                delta_fmt = f"{delta:+.0f} ms" if "latency" in key else f"{delta:+.3f}"
                delta_color = "inverse" if "latency" in key else "normal"
                col.metric(
                    f"{icon} {label}",
                    fmt,
                    delta=delta_fmt,
                    delta_color=delta_color,
                )
            else:
                col.metric(f"{icon} {label}", fmt)

        st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)

        # Per-Case Detail
        st.markdown("### 📋 Detalhamento por Caso")
        case_df = cases_to_df(latest)

        st.dataframe(
            case_df.style.format(
                {
                    "Latência (ms)": "{:.0f}",
                    "Relevância": "{:.2f}",
                    "Fundamentação": "{:.2f}",
                    "Roteamento": "{:.2f}",
                    "Cobertura KW": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # Radar chart
        score_metrics = ["Relevância", "Fundamentação", "Roteamento", "Cobertura KW"]
        fig_radar = go.Figure()
        for idx, (_, row) in enumerate(case_df.iterrows()):
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=[row[m] for m in score_metrics],
                    theta=score_metrics,
                    fill="toself",
                    name=row["Caso"],
                    line=dict(color=CHART_COLORS[idx % len(CHART_COLORS)]),
                    fillcolor=CHART_COLORS[idx % len(CHART_COLORS)],
                    opacity=0.6,
                )
            )
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1]),
                bgcolor="rgba(0,0,0,0)",
            ),
            title="Perfil de Scores por Caso de Teste",
            height=480,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Trends
        if len(results) > 1:
            st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
            st.markdown("### 📈 Tendências")
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
                    title="Métricas de Qualidade ao Longo do Tempo",
                    markers=True,
                    color_discrete_sequence=CHART_COLORS,
                )
                fig_trend.update_yaxes(range=[0, 1])
                fig_trend.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#FAFAFA"),
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            with col_right:
                fig_latency = px.bar(
                    trend_df,
                    x="timestamp",
                    y="avg_latency_ms",
                    title="Latência Média por Run (ms)",
                    color_discrete_sequence=["#FF4B4B"],
                )
                fig_latency.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#FAFAFA"),
                )
                st.plotly_chart(fig_latency, use_container_width=True)

        # History
        with st.expander("📜 Histórico de Runs"):
            for r in reversed(results):
                a = r["aggregated"]
                st.markdown(
                    f"**{r['timestamp']}** — "
                    f"Rel: {a['avg_relevance']:.2f} · "
                    f"Fund: {a['avg_groundedness']:.2f} · "
                    f"Rout: {a['avg_routing_accuracy']:.2f} · "
                    f"KW: {a['avg_keyword_coverage']:.2f} · "
                    f"Lat: {a['avg_latency_ms']:.0f}ms"
                )

    # Run New Evaluation
    st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
    st.markdown("### 🧪 Rodar Nova Avaliação")

    if st.button("🚀 Run Evals", type="primary"):
        with st.spinner("Rodando suite de avaliação... Pode levar 1-2 minutos."):
            try:
                from evals.runner import run_evals

                report = run_evals()
                st.success(f"Avaliação completa! **{report['n_cases']}** casos avaliados.")
                st.json(report["aggregated"])
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Avaliação falhou: {e}")
