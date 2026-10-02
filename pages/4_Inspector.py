"""
pages/4_Inspector.py
--------------------
Document inspector: browse individual listings, inspect their embedding,
and optionally run a live vector search to see where they rank.

Useful for:
  - Verifying embedding content (what went into the vector?)
  - Spot-checking that a specific listing is retrievable
  - Exploring the raw field values alongside the embedding preview
"""

import streamlit as st

st.set_page_config(page_title="Doc Inspector", page_icon="🔎", layout="wide")

import pandas as pd
import plotly.graph_objects as go

from core.db import EMBEDDING_FIELD, LISTING_TEXT_FIELD, benchmark_col, embedding_count
from core.embeddings import DOCUMENT_MODEL, QUERY_MODELS, embed_query
from core.indexes import INDEX_DEFINITIONS, get_index_status
from core.metrics import run_vector_search, score_stats
from utils.display import render_embedding_preview, render_result_card

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Document Inspector")
st.markdown(
    "Browse any listing in the benchmark collection.  View its `listing_text`, "
    "its `emb_large` embedding, and optionally run a live vector search using "
    "the document's own embedding as the query vector."
)

# ---------------------------------------------------------------------------
# Readiness check
# ---------------------------------------------------------------------------

emb_count = embedding_count()
if emb_count == 0:
    st.warning("No documents with embeddings found. Complete the Setup page first.")
    st.stop()

# ---------------------------------------------------------------------------
# Document selector
# ---------------------------------------------------------------------------

col = benchmark_col()

@st.cache_data(ttl=120, show_spinner=False)
def load_names() -> list[str]:
    return sorted(
        d["name"]
        for d in col.find({EMBEDDING_FIELD: {"$exists": True}}, {"name": 1}, limit=500)
        if d.get("name")
    )

names = load_names()

search_name = st.text_input("Filter by listing name (substring)", placeholder="Cosy…")
filtered = [n for n in names if search_name.lower() in n.lower()] if search_name else names

if not filtered:
    st.info("No listings match that filter.")
    st.stop()

selected_name = st.selectbox("Select listing", filtered)

# ---------------------------------------------------------------------------
# Fetch and display the selected document
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_doc(name: str) -> dict | None:
    return col.find_one({"name": name})

doc = fetch_doc(selected_name)

if not doc:
    st.error("Document not found.")
    st.stop()

emb: list[float] | None = doc.get(EMBEDDING_FIELD)
listing_text: str = doc.get(LISTING_TEXT_FIELD, "")

# ---------------------------------------------------------------------------
# Layout: top row — listing text + metadata
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader(doc.get("name", ""))

meta_cols = st.columns(4)
price_val = doc.get("price")
if hasattr(price_val, "to_decimal"):
    price_val = float(price_val.to_decimal())
meta_cols[0].metric("Property type", doc.get("property_type", "—"))
meta_cols[1].metric("Room type", doc.get("room_type", "—"))
meta_cols[2].metric("Bedrooms", str(doc.get("bedrooms", "—")))
meta_cols[3].metric("Price / night", f"${price_val:,.2f}" if price_val else "—")

addr = doc.get("address") or {}
neighbourhood = (
    addr.get("neighbourhood_cleansed")
    or addr.get("suburb")
    or addr.get("market")
    or addr.get("country")
    or "—"
)
st.caption(f"Neighbourhood: {neighbourhood}")

tab_text, tab_embed, tab_raw, tab_search = st.tabs(
    ["listing_text", "Embedding", "Raw document", "Live search"]
)

# ---------------------------------------------------------------------------
# Tab 1 — listing_text
# ---------------------------------------------------------------------------

with tab_text:
    st.markdown("**`listing_text`** — composite free-text field used for embedding")

    if listing_text:
        char_count = len(listing_text)
        # Rough token estimate: ~4 chars/token on English text
        token_estimate = char_count // 4
        st.caption(f"{char_count:,} characters, ~{token_estimate:,} tokens")
        st.text_area(
            label="listing_text content",
            value=listing_text,
            height=300,
            label_visibility="collapsed",
            disabled=True,
        )

        # Show which source fields contributed
        st.markdown("**Source fields included in `listing_text`:**")
        source_fields = [
            "name", "summary", "description", "space",
            "neighborhood_overview", "notes", "transit",
            "access", "interaction", "house_rules",
        ]
        rows = []
        for f in source_fields:
            v = doc.get(f, "")
            rows.append({
                "Field": f,
                "Present": "yes" if v else "no",
                "Length (chars)": len(v) if v else 0,
                "Preview": (v[:80] + "…") if len(v or "") > 80 else (v or ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No `listing_text` field found. Re-run Setup Step 1.")

# ---------------------------------------------------------------------------
# Tab 2 — Embedding
# ---------------------------------------------------------------------------

with tab_embed:
    if not emb:
        st.warning("No embedding found. Complete Setup Step 2.")
    else:
        st.markdown(
            f"**`{EMBEDDING_FIELD}`** — generated by `{DOCUMENT_MODEL}`, "
            f"`input_type='document'`"
        )

        dim_col, byte_col, model_col = st.columns(3)
        dim_col.metric("Dimensions", f"{len(emb):,}")
        byte_col.metric("Storage (float32)", f"{len(emb) * 4:,} bytes")
        model_col.metric("Generated by", DOCUMENT_MODEL)

        # Dimension preview slider
        n_preview = st.slider(
            "Dimensions to preview",
            min_value=8,
            max_value=min(256, len(emb)),
            value=32,
            step=8,
        )
        render_embedding_preview(emb, label=EMBEDDING_FIELD, n_dims=n_preview)

        # Statistics
        import numpy as np
        arr = emb
        st.markdown("**Vector statistics**")
        stat_cols = st.columns(4)
        stat_cols[0].metric("Min", f"{min(arr):.6f}")
        stat_cols[1].metric("Max", f"{max(arr):.6f}")
        mean_val = sum(arr) / len(arr)
        stat_cols[2].metric("Mean", f"{mean_val:.6f}")
        norm = sum(v ** 2 for v in arr) ** 0.5
        stat_cols[3].metric("L2 norm", f"{norm:.6f}")

        # Full values table (paginated via expander)
        with st.expander("All 1024 values (first 50 shown)"):
            rows = [{"Index": i, "Value": f"{v:.8f}"} for i, v in enumerate(emb[:50])]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab 3 — Raw document
# ---------------------------------------------------------------------------

with tab_raw:
    st.markdown("Full document as stored in MongoDB (embedding field excluded for readability).")

    display_doc = {k: v for k, v in doc.items() if k not in (EMBEDDING_FIELD,)}

    # Convert non-serialisable types
    def clean_for_display(obj):
        from bson import ObjectId, Decimal128
        from datetime import datetime
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, Decimal128):
            return float(obj.to_decimal())
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: clean_for_display(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_display(i) for i in obj]
        return obj

    st.json(clean_for_display(display_doc), expanded=True)

# ---------------------------------------------------------------------------
# Tab 4 — Live search (use this doc's own embedding as the query vector)
# ---------------------------------------------------------------------------

with tab_search:
    st.markdown(
        "Use this document's `emb_large` vector as the query vector — effectively "
        "asking: *'find me listings most similar to this one.'*"
    )

    if not emb:
        st.warning("No embedding for this document.")
    else:
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            selected_index = st.selectbox(
                "Index to search",
                options=list(INDEX_DEFINITIONS.keys()),
                format_func=lambda k: INDEX_DEFINITIONS[k]["label"],
            )
        with s_col2:
            s_limit = st.slider("Results", 3, 20, 5, key="inspector_limit")
            s_candidates = st.slider(
                "numCandidates", 10, 500, 150, 10, key="inspector_candidates"
            )

        exclude_self = st.checkbox("Exclude this document from results", value=True)

        if st.button("Find similar listings", type="primary"):
            import time
            t0 = time.perf_counter()
            results, db_ms = run_vector_search(
                query_vector=emb,
                index_name=selected_index,
                num_candidates=s_candidates,
                limit=s_limit + (1 if exclude_self else 0),
            )
            total_ms = (time.perf_counter() - t0) * 1_000

            # Remove self from results if requested
            if exclude_self:
                doc_id_str = str(doc["_id"])
                results = [r for r in results if r.id != doc_id_str][:s_limit]

            st.success(f"Query completed in {db_ms:.0f} ms (total {total_ms:.0f} ms)")

            idx_defn = INDEX_DEFINITIONS[selected_index]
            if idx_defn["quantization"] == "binary":
                st.warning(
                    "Binary index uses euclidean similarity — scores are not comparable "
                    "to cosine-based scores.",
                    icon="⚠️",
                )

            for i, res in enumerate(results, start=1):
                render_result_card(res, i)
