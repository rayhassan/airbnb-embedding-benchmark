"""
utils/display.py
----------------
Shared Streamlit UI components used across all pages.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.indexes import INDEX_DEFINITIONS
from core.metrics import ScoreStats, SearchResult, SearchRun, overlap_matrix, recall_at_k, score_stats


# ---------------------------------------------------------------------------
# Colour / theme helpers
# ---------------------------------------------------------------------------

STATUS_COLOURS = {
    "ready": "green",
    "building": "orange",
    "missing": "red",
    "error": "red",
}

STATUS_ICONS = {
    "ready": "✅",
    "building": "⏳",
    "missing": "❌",
    "error": "⚠️",
}


# ---------------------------------------------------------------------------
# Connection / status banner
# ---------------------------------------------------------------------------

def render_connection_status(ok: bool, detail: str) -> None:
    if ok:
        st.success(f"MongoDB Atlas connected — {detail}")
    else:
        st.error(f"Cannot connect to MongoDB Atlas: {detail}")


def render_setup_summary(benchmark_count: int, embedding_count: int, index_statuses: dict[str, str]) -> None:
    """Compact 3-column status display shown on the home page."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Benchmark docs", f"{benchmark_count:,}")
    with c2:
        st.metric("Docs with embeddings", f"{embedding_count:,}")
    with c3:
        ready = sum(1 for s in index_statuses.values() if s == "ready")
        st.metric("Indexes ready", f"{ready} / {len(index_statuses)}")

    for name, status in index_statuses.items():
        label = INDEX_DEFINITIONS[name]["label"]
        icon = STATUS_ICONS.get(status, "?")
        colour = STATUS_COLOURS.get(status, "grey")
        st.markdown(f":{colour}[{icon} **{label}** — `{name}` — {status}]")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_cost(usd: float) -> str:
    """
    Format a cost value so it always fits in a narrow card.
    - >= $0.01      → 4 dp   e.g. "$0.0500"
    - >= $0.000001  → 6 dp   e.g. "$0.000084"
    - smaller       → scientific  e.g. "$8.40e-8"
    """
    if usd >= 0.01:
        return f"${usd:.4f}"
    if usd >= 0.000001:
        return f"${usd:.6f}"
    return f"${usd:.2e}"


def _fmt_cost_per_million(usd: float) -> str:
    """Cost scaled to 1M queries for context."""
    per_m = usd * 1_000_000
    if per_m >= 1:
        return f"${per_m:.2f}/1M"
    return f"${per_m:.4f}/1M"


_METRIC_CSS = """
<style>
.mc-card {
    background: #f8f9fb;
    border: 1px solid #e0e3e9;
    border-radius: 6px;
    padding: 8px 10px 6px 10px;
    text-align: center;
    min-width: 0;
}
.mc-label {
    font-size: 0.70rem;
    color: #6b7280;
    font-weight: 500;
    letter-spacing: 0.02em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 2px;
}
.mc-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #111827;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.mc-sub {
    font-size: 0.65rem;
    color: #9ca3af;
    margin-top: 1px;
    white-space: nowrap;
}
</style>
"""

def _metric_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="mc-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="mc-card">'
        f'<div class="mc-label">{label}</div>'
        f'<div class="mc-value">{value}</div>'
        f'{sub_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Metrics bar
# ---------------------------------------------------------------------------

def render_metrics_bar(run: SearchRun) -> None:
    """
    Horizontal metric row using compact custom HTML cards so values
    are never clipped regardless of decimal places or column width.
    Shows: embed latency | DB latency | total latency | tokens | cost per query | cost per 1M
    """
    st.markdown(_METRIC_CSS, unsafe_allow_html=True)

    cost_str = _fmt_cost(run.estimated_cost_usd)
    per_m_str = _fmt_cost_per_million(run.estimated_cost_usd)

    cards = [
        _metric_card("Embed latency", f"{run.query_embed_ms:.0f} ms"),
        _metric_card("DB latency",    f"{run.db_query_ms:.0f} ms"),
        _metric_card("Total latency", f"{run.latency_ms:.0f} ms"),
        _metric_card("Tokens",        f"{run.tokens_used:,}"),
        _metric_card("Cost / query",  cost_str, sub=per_m_str),
    ]

    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Result card
# ---------------------------------------------------------------------------

def render_result_card(result: SearchResult, rank: int) -> None:
    score_val = result.score
    score_str = f"{score_val:.4f}"
    price_str = f"${result.price:,.2f}" if result.price is not None else "N/A"
    beds_str = f"{int(result.bedrooms)} bed" if result.bedrooms else "?"

    with st.expander(
        f"#{rank}  **{result.name or 'Unnamed listing'}**  —  score: `{score_str}`",
        expanded=(rank <= 3),
    ):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f"**Neighbourhood:** {result.neighbourhood or '—'}")
            st.markdown(f"**Type:** {result.property_type} / {result.room_type} / {beds_str}")
            st.caption(result.listing_text_snippet + ("…" if len(result.listing_text_snippet) == 300 else ""))
        with col_b:
            st.markdown(_METRIC_CSS, unsafe_allow_html=True)
            st.markdown(
                _metric_card("Price / night", price_str) +
                "<div style='margin-top:6px'></div>" +
                _metric_card("Similarity", score_str),
                unsafe_allow_html=True,
            )


def render_results_list(run: SearchRun) -> None:
    if not run.results:
        st.info("No results returned. Check that the index is READY and embeddings exist.")
        return
    for i, res in enumerate(run.results, start=1):
        render_result_card(res, i)


# ---------------------------------------------------------------------------
# Score distribution chart
# ---------------------------------------------------------------------------

def render_score_chart(runs: dict[str, SearchRun], title: str = "Score distribution") -> None:
    """
    Box/violin chart comparing score distributions across multiple runs.
    Each run becomes one trace keyed by its label.
    """
    fig = go.Figure()
    for label, run in runs.items():
        scores = [r.score for r in run.results]
        fig.add_trace(go.Box(
            y=scores,
            name=label,
            boxpoints="all",
            jitter=0.4,
            pointpos=-1.5,
        ))
    fig.update_layout(
        title=title,
        yaxis_title="Similarity score",
        showlegend=True,
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Latency comparison bar chart
# ---------------------------------------------------------------------------

def render_latency_chart(runs: dict[str, SearchRun], title: str = "Latency breakdown") -> None:
    labels = list(runs.keys())
    embed_ms = [runs[l].query_embed_ms for l in labels]
    db_ms = [runs[l].db_query_ms for l in labels]

    fig = go.Figure(data=[
        go.Bar(name="Embed (Voyage API)", x=labels, y=embed_ms, marker_color="#636EFA"),
        go.Bar(name="DB query (MongoDB)", x=labels, y=db_ms, marker_color="#EF553B"),
    ])
    fig.update_layout(
        barmode="stack",
        title=title,
        yaxis_title="Milliseconds",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Recall / overlap table
# ---------------------------------------------------------------------------

def render_recall_table(runs: dict[str, SearchRun], k: int) -> None:
    """
    Render a symmetric recall@k matrix.  Diagonal is always 1.0.
    """
    labels = list(runs.keys())
    matrix = overlap_matrix(runs, k)

    rows = []
    for la in labels:
        row = {"": la}
        for lb in labels:
            row[lb] = f"{matrix.get((la, lb), 1.0):.2f}"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("")
    st.markdown(f"**Recall@{k}** (fraction of top-{k} results shared between runs)")
    st.dataframe(df, use_container_width=True)


# ---------------------------------------------------------------------------
# Embedding vector preview
# ---------------------------------------------------------------------------

def render_embedding_preview(values: list[float], label: str, n_dims: int = 32) -> None:
    """
    Bar chart of the first n_dims dimensions of an embedding vector.
    """
    preview = values[:n_dims]
    fig = go.Figure(go.Bar(
        x=list(range(len(preview))),
        y=preview,
        marker_color=["#1f77b4" if v >= 0 else "#d62728" for v in preview],
    ))
    fig.update_layout(
        title=f"{label} — first {n_dims} of {len(values)} dimensions",
        xaxis_title="Dimension index",
        yaxis_title="Value",
        height=250,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Query display
# ---------------------------------------------------------------------------

def render_query_box(query_text: str, model: str, index_name: str) -> None:
    """Show the query text alongside the model and index used."""
    st.markdown(
        f"""
        <div style="
            background:#f0f2f6;
            border-left:4px solid #1f77b4;
            padding:10px 16px;
            border-radius:4px;
            margin-bottom:8px;
        ">
        <b>Query:</b> {query_text}<br>
        <small><b>Query model:</b> {model} &nbsp;|&nbsp; <b>Index:</b> {index_name}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Score summary table (used in quantization page)
# ---------------------------------------------------------------------------

def render_score_summary_table(runs: dict[str, SearchRun]) -> None:
    rows = []
    for label, run in runs.items():
        stats = score_stats(run.results)
        rows.append({
            "Configuration": label,
            "Results": stats.count,
            "Top score": f"{stats.top:.4f}",
            "Mean score": f"{stats.mean:.4f}",
            "Std dev": f"{stats.std:.4f}",
            "Embed ms": f"{run.query_embed_ms:.0f}",
            "DB ms": f"{run.db_query_ms:.0f}",
            "Total ms": f"{run.latency_ms:.0f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Cost comparison table
# ---------------------------------------------------------------------------

def render_cost_table(runs: dict[str, SearchRun]) -> None:
    rows = []
    for label, run in runs.items():
        rows.append({
            "Configuration": label,
            "Query model": run.model,
            "Tokens": run.tokens_used,
            "Cost per query": _fmt_cost(run.estimated_cost_usd),
            "Cost / 1M queries": f"${run.estimated_cost_usd * 1_000_000:.2f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
