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

    /* ── Hero banner ────────────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid rgba(255, 75, 75, 0.15);
        border-radius: 20px;
        padding: 2.5rem 3rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,75,75,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,142,83,0.06) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8E53 50%, #FFBD59 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        letter-spacing: -1px;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #9CA3AF;
        margin-top: 0;
        margin-bottom: 1.2rem;
        position: relative;
        z-index: 1;
    }
    .hero-chips {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    .hero-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.08);
        color: #D1D5DB;
    }
    .chip-rag     { background: rgba(59,130,246,0.12); border-color: rgba(59,130,246,0.25); }
    .chip-agents  { background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.25); }
    .chip-evals   { background: rgba(167,139,250,0.12); border-color: rgba(167,139,250,0.25); }
    .chip-stream  { background: rgba(255,75,75,0.12); border-color: rgba(255,75,75,0.25); }

    /* ── Agent cards ────────────────────────────────────────────────── */
    .agent-cards {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 1rem 0 1.5rem;
    }
    .agent-card {
        background: linear-gradient(145deg, #1A1D23 0%, #1E2229 100%);
        border: 1px solid #2D3139;
        border-radius: 14px;
        padding: 18px 16px;
        text-align: center;
        transition: border-color 0.25s, transform 0.2s;
    }
    .agent-card:hover {
        border-color: rgba(255,142,83,0.4);
        transform: translateY(-2px);
    }
    .agent-icon { font-size: 1.8rem; margin-bottom: 6px; }
    .agent-name {
        font-size: 0.82rem;
        font-weight: 700;
        color: #E5E7EB;
        margin-bottom: 3px;
    }
    .agent-desc {
        font-size: 0.72rem;
        color: #6B7280;
        line-height: 1.35;
    }

    /* ── Section panels ─────────────────────────────────────────────── */
    .panel {
        background: linear-gradient(145deg, #13151A 0%, #181B22 100%);
        border: 1px solid #2D3139;
        border-radius: 16px;
        padding: 1.6rem 1.5rem 1.2rem;
    }
    .panel-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }
    .panel-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    .panel-icon-upload { background: rgba(59,130,246,0.15); }
    .panel-icon-query  { background: rgba(255,75,75,0.15); }
    .panel-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F3F4F6;
        margin: 0;
    }
    .panel-desc {
        font-size: 0.85rem;
        color: #6B7280;
        margin-bottom: 1rem;
        padding-left: 48px;
    }

    /* ── How it works ───────────────────────────────────────────────── */
    .flow-steps {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        margin: 0.8rem 0 1.6rem;
        flex-wrap: wrap;
    }
    .flow-step {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 16px;
        background: rgba(255,255,255,0.03);
        border: 1px solid #2D3139;
        border-radius: 10px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #D1D5DB;
    }
    .flow-arrow {
        color: #4B5563;
        font-size: 1rem;
    }

    /* ── Metrics / badges ───────────────────────────────────────────── */
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

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
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

    .menu-loaded {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(16,185,129,0.12);
        color: #10B981;
        border: 1px solid rgba(16,185,129,0.3);
        margin-bottom: 0.8rem;
    }

    .stButton > button {
        border-radius: 10px;
        transition: all 0.2s;
        font-weight: 600;
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

st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-title">🍽️ SaborAI</p>
        <p class="hero-subtitle">
            Sistema multi-agente com RAG para análise inteligente de cardápios de restaurantes
        </p>
        <div class="hero-chips">
            <span class="hero-chip chip-rag">🔎 RAG com ChromaDB</span>
            <span class="hero-chip chip-agents">🤖 3 Agentes Especialistas</span>
            <span class="hero-chip chip-evals">📊 Eval Framework</span>
            <span class="hero-chip chip-stream">⚡ Async Paralelo</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Agent cards ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="agent-cards">
        <div class="agent-card">
            <div class="agent-icon">🥗</div>
            <div class="agent-name">NutritionAgent</div>
            <div class="agent-desc">Restrições alimentares, alérgenos, calorias e dietas especiais</div>
        </div>
        <div class="agent-card">
            <div class="agent-icon">🎯</div>
            <div class="agent-name">RecommendationAgent</div>
            <div class="agent-desc">Combos personalizados, sugestões por orçamento e ocasião</div>
        </div>
        <div class="agent-card">
            <div class="agent-icon">✍️</div>
            <div class="agent-name">QualityAgent</div>
            <div class="agent-desc">Qualidade das descrições, UX writing e conversão</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── How it works ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="flow-steps">
        <span class="flow-step">📄 Upload Cardápio</span>
        <span class="flow-arrow">→</span>
        <span class="flow-step">🔪 Chunking + Embed</span>
        <span class="flow-arrow">→</span>
        <span class="flow-step">🧠 Routing LLM</span>
        <span class="flow-arrow">→</span>
        <span class="flow-step">🤖 Agentes em Paralelo</span>
        <span class="flow-arrow">→</span>
        <span class="flow-step">💬 Resposta Consolidada</span>
    </div>
    """,
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
        st.markdown(
            '<div class="panel-header">'
            '<div class="panel-icon panel-icon-upload">📤</div>'
            '<p class="panel-title">Ingerir Cardápio</p>'
            "</div>"
            '<p class="panel-desc">Suba um arquivo PDF/TXT ou cole o texto do cardápio.</p>',
            unsafe_allow_html=True,
        )

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
        st.markdown(
            '<div class="panel-header">'
            '<div class="panel-icon panel-icon-query">🔍</div>'
            '<p class="panel-title">Consultar Cardápio</p>'
            "</div>"
            '<p class="panel-desc">Faça perguntas sobre o cardápio ingerido.</p>',
            unsafe_allow_html=True,
        )

        # Show loaded-menu indicator
        active_menu = st.session_state.get("ingested_menu")
        if active_menu:
            st.markdown(
                f'<span class="menu-loaded">✅ Cardápio ativo: {active_menu}</span>',
                unsafe_allow_html=True,
            )

        # Session state for query processing
        if "processing" not in st.session_state:
            st.session_state.processing = False
        if "pending_query" not in st.session_state:
            st.session_state.pending_query = None
        if "last_result" not in st.session_state:
            st.session_state.last_result = None

        is_busy = st.session_state.processing

        query = st.text_area(
            "Sua pergunta",
            height=100,
            placeholder="Ex: Monte um combo vegano por até R$60 para um casal.",
            disabled=is_busy,
        )

        example_queries = [
            ("🌱", "Quais pratos são adequados para veganos?"),
            ("💰", "Monte um combo completo por até R$60 para um casal."),
            ("✨", "Avalie a qualidade das descrições e sugira melhorias."),
            ("🚫", "Quais pratos não contêm glúten nem laticínios?"),
        ]

        st.markdown("**Exemplos rápidos:**")
        example_cols = st.columns(2)
        for i, (icon, ex) in enumerate(example_queries):
            with example_cols[i % 2]:
                if st.button(
                    f"{icon} {ex[:45]}...",
                    key=f"ex_{i}",
                    use_container_width=True,
                    disabled=is_busy,
                ):
                    st.session_state.pending_query = ex
                    st.session_state.processing = True
                    st.session_state.last_result = None
                    st.rerun()

        if st.button(
            "🚀 Enviar Consulta",
            type="primary",
            disabled=is_busy or not query,
            use_container_width=True,
        ):
            st.session_state.pending_query = query
            st.session_state.processing = True
            st.session_state.last_result = None
            st.rerun()

        # Process pending query (runs after rerun — buttons already disabled)
        if st.session_state.processing and st.session_state.pending_query:
            effective_query = st.session_state.pending_query
            st.caption(f"Consultando: *{effective_query}*")
            with st.spinner("Consultando agentes especializados..."):
                try:
                    from agents.supervisor import SupervisorAgent

                    active_menu = st.session_state.get("ingested_menu")
                    supervisor = SupervisorAgent()
                    result = supervisor.run(effective_query, menu_name=active_menu)
                    st.session_state.last_result = result
                except Exception as e:
                    st.session_state.last_result = {"error": str(e)}
                finally:
                    st.session_state.processing = False
                    st.session_state.pending_query = None
            st.rerun()

        # Display results (persists across reruns)
        if st.session_state.last_result:
            result = st.session_state.last_result
            if "error" in result:
                st.error(f"Erro na consulta: {result['error']}")
            else:
                st.markdown("---")
                st.markdown("#### 💬 Resposta")
                st.info(result["response"])

                agents_used = result["agents_used"]
                latency = result.get("latency_ms")
                badges = " ".join(f'<span class="status-badge">{a}</span>' for a in agents_used)
                latency_badge = (
                    f' <span class="status-badge" style="background:rgba(255,75,75,0.12);'
                    f'color:#FF8E53;border-color:rgba(255,142,83,0.3);">'
                    f'⚡ {latency:.0f}ms</span>'
                    if latency
                    else ""
                )
                st.markdown(
                    f"**Agentes utilizados:** {badges}{latency_badge}",
                    unsafe_allow_html=True,
                )

                with st.expander("Ver detalhes por agente"):
                    for agent_name, output in result.get("agent_outputs", {}).items():
                        st.markdown(f"**{agent_name.upper()}**")
                        st.info(output)


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
