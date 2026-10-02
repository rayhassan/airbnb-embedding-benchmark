"""
pages/3_Quantization.py
-----------------------
Atlas-side quantization impact benchmark:

  Same query vector (voyage-4-large) → three indexes

  vs_none    — float32, cosine   (baseline / ground truth)
  vs_scalar  — int8,    cosine   (~4× smaller index)
  vs_binary  — 1-bit,   euclidean (~32× smaller index)

Key metrics: latency, score magnitude, recall@k vs float32 baseline.

Note on binary scores: binary quantization uses euclidean similarity, which
produces scores on a different scale to cosine.  Scores are not comparable
across configurations — only the result set overlap (recall@k) is a fair
cross-configuration metric.
"""

import streamlit as st

st.set_page_config(page_title="Quantization", page_icon="📐", layout="wide")

import pandas as pd
import plotly.graph_objects as go

from core.embeddings import DOCUMENT_MODEL, embed_query
from core.indexes import INDEX_DEFINITIONS, get_index_status
from core.metrics import full_search_run, recall_at_k, score_stats
from utils.display import (
    render_embedding_preview,
    render_latency_chart,
    render_metrics_bar,
    render_query_box,
    render_result_card,
    render_score_chart,
    render_score_summary_table,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Quantization Impact")
st.markdown(
    """
    The **same** `voyage-4-large` query vector is sent to three Atlas vector indexes
    that differ only in their Atlas-side quantization setting.  The underlying document
    vectors (`emb_large`) are unchanged — only the index representation differs.

    | Index | Atlas quantization | Similarity | Storage saving |
    |---|---|---|---|
    | `vs_none` | none (float32) | cosine | 1× (baseline) |
    | `vs_scalar` | scalar (int8) | cosine | ~4× |
    | `vs_binary` | binary (1-bit) | euclidean | ~32× |

    > **Score comparability warning:** `vs_binary` uses euclidean similarity (required
    > by Atlas for binary indexes), so its scores are on a different scale to cosine.
    > Use **Recall@k** — not raw scores — to compare result quality across configurations.
    """
)

# ---------------------------------------------------------------------------
# Readiness check
# ---------------------------------------------------------------------------

statuses = {name: get_index_status(name) for name in INDEX_DEFINITIONS}
not_ready = [n for n, s in statuses.items() if s != "ready"]

if not_ready:
    st.warning(
        f"The following indexes are not ready: {', '.join(not_ready)}. "
        "Complete the Setup page before running this benchmark."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Search controls")

    num_candidates = st.slider(
        "numCandidates",
        min_value=10,
        max_value=500,
        value=150,
        step=10,
        help="Candidates evaluated per ANN search.  Applied to all three indexes.",
    )
    limit = st.slider("Results (limit)", min_value=3, max_value=25, value=10, step=1)

    st.markdown("---")
    st.subheader("Optional pre-filters")
    st.caption("Applied identically to all three indexes.")

    @st.cache_data(ttl=300)
    def get_filter_options():
        from core.db import benchmark_col
        col = benchmark_col()
        return sorted(col.distinct("property_type")), sorted(col.distinct("room_type"))

    prop_types, room_types = get_filter_options()
    selected_prop = st.multiselect("Property type", prop_types)
    selected_room = st.multiselect("Room type", room_types)
    bed_range = st.slider("Bedrooms (max)", 0, 10, 10)

    pre_filter = None
    clauses = []
    if selected_prop:
        clauses.append({"property_type": {"$in": selected_prop}})
    if selected_room:
        clauses.append({"room_type": {"$in": selected_room}})
    if bed_range < 10:
        clauses.append({"bedrooms": {"$lte": bed_range}})
    if clauses:
        pre_filter = {"$and": clauses} if len(clauses) > 1 else clauses[0]

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

from core.sample_queries import SAMPLE_QUERIES

st.markdown("**Try a sample query**")
sample_options = ["— type your own —"] + [q["label"] for q in SAMPLE_QUERIES]
selected_sample = st.selectbox(
    "Sample queries",
    sample_options,
    label_visibility="collapsed",
)

prefill = ""
if selected_sample != "— type your own —":
    match = next(q for q in SAMPLE_QUERIES if q["label"] == selected_sample)
    prefill = match["query"]
    st.caption(f"_{match['why']}_")

query_text = st.text_input(
    "Search query",
    value=prefill,
    placeholder="e.g. quiet apartment with a view for remote work",
    help="Embedded once with voyage-4-large; the same vector is sent to all three indexes.",
)

run_search = st.button("Search", type="primary", disabled=not query_text.strip())

if not query_text.strip():
    st.info("Enter a query above or pick a sample to start.")
    st.stop()

if not run_search:
    st.stop()

# ---------------------------------------------------------------------------
# Embed query ONCE with voyage-4-large
# ---------------------------------------------------------------------------

with st.spinner(f"Embedding query with `{DOCUMENT_MODEL}`…"):
    er = embed_query(query_text, model=DOCUMENT_MODEL)

st.success(
    f"Query embedded — {er.tokens} tokens, {er.latency_ms:.0f} ms, "
    f"${er.estimated_cost_usd:.6f}"
)
st.caption(
    "The same vector is now sent to vs_none, vs_scalar, and vs_binary simultaneously."
)

# ---------------------------------------------------------------------------
# Run searches against all three indexes
# ---------------------------------------------------------------------------

index_keys = list(INDEX_DEFINITIONS.keys())  # vs_none, vs_scalar, vs_binary
runs = {}

with st.spinner("Querying all three indexes…"):
    for idx_name in index_keys:
        runs[idx_name] = full_search_run(
            query_text=query_text,
            query_vector=er.vector,
            query_embed_ms=er.latency_ms,
            tokens_used=er.tokens,
            query_model=DOCUMENT_MODEL,
            index_name=idx_name,
            num_candidates=num_candidates,
            limit=limit,
            pre_filter=pre_filter,
        )

# ---------------------------------------------------------------------------
# Per-index columns
# ---------------------------------------------------------------------------

st.markdown("---")

cols = st.columns(3)
for col_ui, idx_name in zip(cols, index_keys):
    defn = INDEX_DEFINITIONS[idx_name]
    run = runs[idx_name]
    with col_ui:
        st.subheader(defn["label"])
        st.caption(defn["description"])
        render_query_box(query_text, DOCUMENT_MODEL, idx_name)

        if defn["quantization"] == "binary":
            st.warning(
                "Binary index uses **euclidean** similarity — scores are not "
                "comparable to cosine scores above. Use Recall@k for accuracy comparison.",
                icon="⚠️",
            )

        render_metrics_bar(run)
        st.markdown("**Results**")
        for i, res in enumerate(run.results, start=1):
            render_result_card(res, i)

# ---------------------------------------------------------------------------
# Aggregate analysis tabs
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Aggregate analysis")

tab_scores, tab_latency, tab_recall, tab_storage = st.tabs(
    ["Scores", "Latency", "Recall@k vs baseline", "Storage estimate"]
)

labeled_runs = {INDEX_DEFINITIONS[k]["label"]: runs[k] for k in index_keys}

with tab_scores:
    st.info(
        "Binary scores (euclidean) are not on the same scale as cosine scores. "
        "The chart below is informational — use Recall@k for cross-configuration comparison."
    )
    render_score_chart(labeled_runs, title="Score distribution per quantization level")
    render_score_summary_table(labeled_runs)

with tab_latency:
    st.markdown(
        "Embed latency is identical across all three runs (same API call, vector reused). "
        "The DB latency difference reflects Atlas index compression effects."
    )
    render_latency_chart(labeled_runs, title="Latency per quantization level")

with tab_recall:
    baseline_ids = [r.id for r in runs["vs_none"].results]
    recall_rows = []
    for idx_name in index_keys:
        ids = [r.id for r in runs[idx_name].results]
        rk = recall_at_k(baseline_ids, ids, limit)
        defn = INDEX_DEFINITIONS[idx_name]
        recall_rows.append({
            "Index": defn["label"],
            "Quantization": defn["quantization"],
            "Similarity": defn["similarity"],
            f"Recall@{limit} vs float32": f"{rk:.2f}",
            f"Docs lost from top-{limit}": int(round((1 - rk) * limit)),
        })
    st.dataframe(pd.DataFrame(recall_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Pairwise overlap across all three configurations**")

    # Build pairwise overlap table manually
    pair_rows = []
    for i, ka in enumerate(index_keys):
        for kb in index_keys[i:]:
            ids_a = [r.id for r in runs[ka].results]
            ids_b = [r.id for r in runs[kb].results]
            r = recall_at_k(ids_a, ids_b, limit)
            pair_rows.append({
                "Config A": INDEX_DEFINITIONS[ka]["label"],
                "Config B": INDEX_DEFINITIONS[kb]["label"],
                f"Overlap@{limit}": f"{r:.2f}",
            })
    st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)

with tab_storage:
    st.markdown(
        """
        Estimated index storage for 14,000 documents × 1024 dimensions:

        | Configuration | Bits/value | Bytes/vector | Total index size (est.) |
        |---|---|---|---|
        | float32 (none) | 32 | 4,096 | ~54 MB |
        | scalar (int8)  | 8  | 1,024 | ~14 MB |
        | binary (1-bit) | 1  | 128   | ~1.8 MB |

        *Actual Atlas index sizes include HNSW graph overhead and will be larger.*
        *The compression ratios (4× scalar, ~32× binary) are accurate for the vector data itself.*
        """
    )

    # Visual bar chart of relative sizes
    fig = go.Figure(go.Bar(
        x=["float32 (none)", "scalar (int8)", "binary (1-bit)"],
        y=[54, 14, 1.8],
        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
        text=["54 MB", "14 MB", "1.8 MB"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Estimated vector data size (14K docs, 1024 dims)",
        yaxis_title="MB",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Representative document
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Representative document")
st.markdown(
    "Top result from `vs_none` (float32 baseline). Scores shown from all three indexes "
    "for the same document."
)

if runs["vs_none"].results:
    top_id = runs["vs_none"].results[0].id

    from core.db import EMBEDDING_FIELD, benchmark_col

    raw = benchmark_col().find_one(
        {"_id": top_id},
        {"name": 1, "listing_text": 1, EMBEDDING_FIELD: 1,
         "property_type": 1, "room_type": 1, "bedrooms": 1},
    )
    if raw:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**{raw.get('name', '')}**")
            st.caption(
                f"{raw.get('property_type', '')} / {raw.get('room_type', '')} / "
                f"{raw.get('bedrooms', '?')} bed"
            )
            st.text_area(
                "listing_text",
                value=raw.get("listing_text", ""),
                height=180,
                label_visibility="visible",
                disabled=True,
            )

        with c2:
            emb = raw.get(EMBEDDING_FIELD)
            if emb:
                render_embedding_preview(emb, f"{EMBEDDING_FIELD} (stored float32)", n_dims=32)

            st.markdown("**Score from each quantization index**")
            for idx_name in index_keys:
                match = next((r for r in runs[idx_name].results if r.id == top_id), None)
                defn = INDEX_DEFINITIONS[idx_name]
                score_str = f"{match.score:.4f} ({defn['similarity']})" if match else "not in top-k"
                st.text(f"  {defn['quantization']:8s}  {score_str}")
else:
    st.info("No results from vs_none baseline.")
