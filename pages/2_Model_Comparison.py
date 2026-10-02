"""
pages/2_Model_Comparison.py
---------------------------
Asymmetric retrieval benchmark:

  Same query text → three query models → same index (vs_none)

  voyage-4-large  (symmetric baseline — same model as document embedding)
  voyage-4        (mid-tier asymmetric)
  voyage-4-lite   (cheapest/fastest asymmetric)

The query vector is computed independently for each model; all three are
sent to the same vs_none index.  Score distributions, latency breakdown,
recall@k vs the large-model baseline, and cost tables are shown side-by-side.
"""

import streamlit as st

st.set_page_config(page_title="Model Comparison", page_icon="🔍", layout="wide")

from core.embeddings import QUERY_MODELS, embed_query
from core.indexes import INDEX_DEFINITIONS, get_index_status
from core.metrics import full_search_run, recall_at_k, score_stats
from utils.display import (
    render_cost_table,
    render_latency_chart,
    render_metrics_bar,
    render_query_box,
    render_recall_table,
    render_result_card,
    render_score_chart,
    render_score_summary_table,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Model Comparison — Asymmetric Retrieval")
st.markdown(
    """
    All three Voyage 4-series query models search the **same** `vs_none` index
    (documents embedded with `voyage-4-large`).  Because all v4 models share a
    vector space, the query vector from a cheaper model is directly comparable
    to the indexed document vectors — no reindexing required.
    """
)

# ---------------------------------------------------------------------------
# Readiness check
# ---------------------------------------------------------------------------

index_name = "vs_none"
index_status = get_index_status(index_name)

if index_status != "ready":
    st.warning(
        f"Index `{index_name}` is **{index_status}**. "
        "Complete the Setup page before running comparisons."
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
        help="Number of ANN candidates Atlas evaluates before returning top-k. "
             "Higher = better recall, slower query.",
    )
    limit = st.slider(
        "Results (limit)",
        min_value=3,
        max_value=25,
        value=10,
        step=1,
    )

    st.markdown("---")
    st.subheader("Optional pre-filters")
    st.caption("Leave blank to search all listings.")

    # Fetch distinct values for filter selectors (cached per session)
    @st.cache_data(ttl=300)
    def get_filter_options():
        from core.db import benchmark_col
        col = benchmark_col()
        prop_types = sorted(col.distinct("property_type"))
        room_types = sorted(col.distinct("room_type"))
        return prop_types, room_types

    prop_types, room_types = get_filter_options()

    selected_prop = st.multiselect("Property type", prop_types)
    selected_room = st.multiselect("Room type", room_types)
    bed_range = st.slider("Bedrooms (max)", 0, 10, 10)

    pre_filter: dict | None = None
    filter_clauses = []
    if selected_prop:
        filter_clauses.append({"property_type": {"$in": selected_prop}})
    if selected_room:
        filter_clauses.append({"room_type": {"$in": selected_room}})
    if bed_range < 10:
        filter_clauses.append({"bedrooms": {"$lte": bed_range}})
    if filter_clauses:
        pre_filter = {"$and": filter_clauses} if len(filter_clauses) > 1 else filter_clauses[0]

    st.markdown("---")
    show_raw_docs = st.checkbox("Show raw document in result cards", value=False)

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

from core.sample_queries import SAMPLE_QUERIES

# Sample query picker — pre-fills the text input
st.markdown("**Try a sample query**")
sample_options = ["— type your own —"] + [q["label"] for q in SAMPLE_QUERIES]
selected_sample = st.selectbox(
    "Sample queries",
    sample_options,
    label_visibility="collapsed",
)

# Resolve pre-fill text from selected sample
prefill = ""
if selected_sample != "— type your own —":
    match = next(q for q in SAMPLE_QUERIES if q["label"] == selected_sample)
    prefill = match["query"]
    st.caption(f"_{match['why']}_")

query_text = st.text_input(
    "Search query",
    value=prefill,
    placeholder="e.g. cosy studio near the beach with great reviews",
    help="Enter any natural-language description of what you are looking for.",
)

run_search = st.button("Search", type="primary", disabled=not query_text.strip())

if not query_text.strip():
    st.info("Enter a query above or pick a sample to start the comparison.")
    st.stop()

if not run_search:
    st.stop()

# ---------------------------------------------------------------------------
# Embed with all three models (in sequence — Voyage API calls are serial)
# ---------------------------------------------------------------------------

model_keys = list(QUERY_MODELS.keys())
embed_results = {}

with st.spinner("Embedding query with all three models…"):
    for model_key in model_keys:
        embed_results[model_key] = embed_query(query_text, model=model_key)

# ---------------------------------------------------------------------------
# Run searches (one per model)
# ---------------------------------------------------------------------------

runs = {}
with st.spinner("Querying MongoDB Atlas…"):
    for model_key in model_keys:
        er = embed_results[model_key]
        runs[model_key] = full_search_run(
            query_text=query_text,
            query_vector=er.vector,
            query_embed_ms=er.latency_ms,
            tokens_used=er.tokens,
            query_model=model_key,
            index_name=index_name,
            num_candidates=num_candidates,
            limit=limit,
            pre_filter=pre_filter,
        )

# ---------------------------------------------------------------------------
# Display — per-model columns
# ---------------------------------------------------------------------------

st.markdown("---")

cols = st.columns(3)
for col_ui, model_key in zip(cols, model_keys):
    defn = QUERY_MODELS[model_key]
    run = runs[model_key]
    with col_ui:
        st.subheader(defn["label"])
        st.caption(defn["description"])
        render_query_box(query_text, model_key, index_name)
        render_metrics_bar(run)
        st.markdown("**Results**")
        if run.results:
            for i, res in enumerate(run.results, start=1):
                render_result_card(res, i)
        else:
            st.info("No results.")

# ---------------------------------------------------------------------------
# Aggregate charts
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Aggregate analysis")

tab_scores, tab_latency, tab_recall, tab_cost = st.tabs(
    ["Score distribution", "Latency breakdown", "Recall@k", "Cost comparison"]
)

with tab_scores:
    labeled_runs = {QUERY_MODELS[k]["label"]: runs[k] for k in model_keys}
    render_score_chart(labeled_runs, title="Score distribution by query model → vs_none index")
    render_score_summary_table(labeled_runs)

with tab_latency:
    render_latency_chart(labeled_runs, title="Latency: Voyage embed + MongoDB query")

with tab_recall:
    st.markdown(
        """
        **Recall@k** measures what fraction of the top-k results from the
        `voyage-4-large` baseline appear in each other model's top-k.
        A recall of 1.0 = identical result sets; 0.5 = half the docs match.
        """
    )
    # Recall of each asymmetric model vs the large baseline
    baseline_ids = [r.id for r in runs["voyage-4-large"].results]
    recall_rows = []
    for model_key in model_keys:
        ids = [r.id for r in runs[model_key].results]
        rk = recall_at_k(baseline_ids, ids, limit)
        recall_rows.append({
            "Query model": QUERY_MODELS[model_key]["label"],
            f"Recall@{limit} vs large baseline": f"{rk:.2f}",
        })
    import pandas as pd
    st.dataframe(pd.DataFrame(recall_rows), use_container_width=True, hide_index=True)

    st.markdown("**Pairwise recall matrix**")
    render_recall_table(
        {QUERY_MODELS[k]["label"]: runs[k] for k in model_keys},
        k=limit,
    )

with tab_cost:
    st.markdown(
        "Costs shown are **query-side only** (Voyage API embed call). "
        "Document indexing cost was paid once during Setup."
    )
    render_cost_table(labeled_runs)

# ---------------------------------------------------------------------------
# Representative document section
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Representative document")
st.markdown(
    "The top result from `voyage-4-large` (the symmetric baseline) is shown "
    "below with its raw embedding preview."
)

if runs["voyage-4-large"].results:
    top_doc_id = runs["voyage-4-large"].results[0].id
    from core.db import EMBEDDING_FIELD, benchmark_col
    from utils.display import render_embedding_preview

    raw = benchmark_col().find_one(
        {"_id": top_doc_id},
        {"name": 1, "listing_text": 1, EMBEDDING_FIELD: 1, "property_type": 1,
         "room_type": 1, "bedrooms": 1, "price": 1, "address": 1},
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
                render_embedding_preview(emb, label=f"{EMBEDDING_FIELD} ({len(emb)} dims)", n_dims=32)

            # Show scores this doc received from each model
            st.markdown("**Scores received from each query model**")
            for model_key in model_keys:
                match = next((r for r in runs[model_key].results if r.id == top_doc_id), None)
                score_str = f"{match.score:.4f}" if match else "not in top-k"
                st.text(f"  {QUERY_MODELS[model_key]['label']:20s}  {score_str}")
else:
    st.info("No results to display a representative document.")
