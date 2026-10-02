"""
pages/1_Setup.py
----------------
Three-step setup wizard:
  Step 1 — Copy listingsAndReviews → listings_benchmark (adds listing_text field)
  Step 2 — Generate voyage-4-large embeddings and write emb_large to each doc
  Step 3 — Create the three Atlas Vector Search indexes

The page is designed to be idempotent: each step checks its own preconditions
and skips work already done.
"""

import time

import streamlit as st
from pymongo import UpdateOne

st.set_page_config(page_title="Setup", page_icon="⚙️", layout="wide")

from core.db import (
    BENCHMARK_COLLECTION,
    EMBEDDING_FIELD,
    LISTING_TEXT_FIELD,
    SOURCE_COLLECTION,
    SOURCE_DB,
    benchmark_col,
    benchmark_count,
    build_listing_text,
    embedding_count,
    get_client,
    source_col,
)
from core.embeddings import (
    BATCH_SIZE,
    DOCUMENT_MODEL,
    embed_documents_batch,
    estimate_setup_cost,
)
from core.indexes import INDEX_DEFINITIONS, create_all_indexes, create_index, get_index_status
from utils.display import render_embedding_preview

st.title("Setup")
st.markdown(
    "Prepare the benchmark dataset in three steps. Each step is **idempotent** — "
    "safe to re-run if it was interrupted."
)

# ---------------------------------------------------------------------------
# Step 1 — Copy collection
# ---------------------------------------------------------------------------

st.header("Step 1 — Copy collection")

src_count = source_col().estimated_document_count()
bm_count = benchmark_count()

st.markdown(
    f"Source: `{SOURCE_DB}.{SOURCE_COLLECTION}` — **{src_count:,} documents**  \n"
    f"Target: `{SOURCE_DB}.{BENCHMARK_COLLECTION}` — **{bm_count:,} documents**"
)

if bm_count >= src_count > 0:
    st.success(f"Collection already copied ({bm_count:,} docs). Nothing to do.")
else:
    st.info(
        "This step copies each document and adds a `listing_text` field by concatenating "
        "the free-text fields (name, summary, description, space, neighborhood_overview, "
        "notes, transit, access, interaction, house_rules).  Single-value fields "
        "(price, bedrooms, amenities, etc.) are kept as-is for CRUD/filter use."
    )
    if st.button("Copy collection", type="primary"):
        col = get_client()[SOURCE_DB]
        progress = st.progress(0, text="Copying documents…")
        status_text = st.empty()

        # Use a cursor + bulk_write to add listing_text during copy.
        # $out would overwrite the collection but can't add computed fields easily.
        target = benchmark_col()
        target.drop()  # start fresh

        cursor = source_col().find({}, batch_size=500)
        batch = []
        processed = 0

        for doc in cursor:
            doc[LISTING_TEXT_FIELD] = build_listing_text(doc)
            batch.append(doc)
            if len(batch) >= 500:
                target.insert_many(batch, ordered=False)
                processed += len(batch)
                batch = []
                pct = min(processed / src_count, 1.0)
                progress.progress(pct, text=f"Copied {processed:,} / {src_count:,}")
                status_text.text(f"Last inserted: {processed:,} docs")

        if batch:
            target.insert_many(batch, ordered=False)
            processed += len(batch)

        # Index _id for fast lookups in later steps
        target.create_index("_id")

        progress.progress(1.0, text="Done")
        status_text.text(f"Copied {processed:,} documents.")
        st.success(f"Collection copied: {processed:,} documents with `listing_text` field.")
        st.rerun()

# ---------------------------------------------------------------------------
# Step 2 — Generate embeddings
# ---------------------------------------------------------------------------

st.header("Step 2 — Generate document embeddings")

bm_count = benchmark_count()
emb_count = embedding_count()

col_a, col_b = st.columns(2)
col_a.metric("Benchmark docs", f"{bm_count:,}")
col_b.metric("Docs with embeddings", f"{emb_count:,}")

if bm_count == 0:
    st.warning("Complete Step 1 first.")
elif emb_count >= bm_count:
    st.success(f"All {emb_count:,} documents already have `{EMBEDDING_FIELD}`. Nothing to do.")
else:
    remaining = bm_count - emb_count
    estimate = estimate_setup_cost(remaining)

    st.markdown(
        f"""
        **Model:** `{DOCUMENT_MODEL}` (document-side, `input_type="document"`)  
        **Documents to embed:** {remaining:,}  
        **Estimated tokens:** ~{estimate['total_tokens']:,}  
        **Estimated cost:** ~${estimate['estimated_cost_usd']:.2f} USD  
        **Batch size:** {BATCH_SIZE} docs/call  
        **Output:** 1024-dimensional float32 vectors stored as `{EMBEDDING_FIELD}`
        """
    )

    confirmed = st.checkbox(
        f"I understand this will call the Voyage AI API and cost approximately "
        f"${estimate['estimated_cost_usd']:.2f} USD.",
        key="embed_confirm",
    )

    if confirmed and st.button("Generate embeddings", type="primary"):
        col = benchmark_col()

        # Fetch only docs that don't yet have the embedding field
        cursor = col.find(
            {EMBEDDING_FIELD: {"$exists": False}},
            {LISTING_TEXT_FIELD: 1, "_id": 1},
            batch_size=BATCH_SIZE,
        )
        docs = list(cursor)
        total = len(docs)

        progress = st.progress(0, text="Starting…")
        status_text = st.empty()
        error_box = st.empty()

        completed = 0
        skipped = 0
        for i in range(0, total, BATCH_SIZE):
            chunk = docs[i : i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1

            # Pair each doc with its text, substituting a minimal placeholder
            # for docs whose listing_text is empty/missing so Voyage never
            # receives an empty string (which it rejects with a 400 error).
            PLACEHOLDER = "listing"
            pairs = []   # (doc_id, text, is_placeholder)
            for d in chunk:
                raw = d.get(LISTING_TEXT_FIELD, "") or ""
                text = raw.strip()
                if text:
                    pairs.append((d["_id"], text, False))
                else:
                    pairs.append((d["_id"], PLACEHOLDER, True))

            ids_to_embed  = [p[0] for p in pairs]
            texts_to_embed = [p[1] for p in pairs]

            try:
                result = embed_documents_batch(texts_to_embed, model=DOCUMENT_MODEL)
            except Exception as exc:  # noqa: BLE001
                error_box.error(f"Voyage API error at batch {batch_num}: {exc}")
                st.stop()

            # Write embeddings back; mark placeholder docs with a flag so they
            # can be identified later if needed.
            ops = []
            for (doc_id, _text, is_placeholder), vec in zip(pairs, result.vectors):
                update: dict = {EMBEDDING_FIELD: vec}
                if is_placeholder:
                    update["emb_large_placeholder"] = True
                    skipped += 1
                ops.append(UpdateOne({"_id": doc_id}, {"$set": update}))
            col.bulk_write(ops, ordered=False)

            completed += len(chunk)
            pct = min(completed / total, 1.0)
            progress.progress(pct, text=f"Embedded {completed:,} / {total:,} docs")
            status_text.text(
                f"Batch {batch_num} — "
                f"{result.total_tokens:,} tokens — "
                f"${result.estimated_cost_usd:.4f}"
                + (f" ({skipped} placeholder docs so far)" if skipped else "")
            )

        progress.progress(1.0, text="Done")
        st.success(f"Embeddings written for {completed:,} documents.")
        st.rerun()

# ---------------------------------------------------------------------------
# Step 3 — Create indexes
# ---------------------------------------------------------------------------

st.header("Step 3 — Create Atlas Vector Search indexes")

emb_count = embedding_count()

if emb_count == 0:
    st.warning("Complete Step 2 first — indexes need documents with the `emb_large` field.")
else:
    statuses = {name: get_index_status(name) for name in INDEX_DEFINITIONS}

    for name, defn in INDEX_DEFINITIONS.items():
        status = statuses[name]
        icon = {"ready": "✅", "building": "⏳", "missing": "❌", "error": "⚠️"}.get(status, "?")
        st.markdown(
            f"{icon} **{defn['label']}** (`{name}`) — "
            f"{defn['description']} — status: **{status}**"
        )

    all_ready = all(s == "ready" for s in statuses.values())
    any_missing = any(s in ("missing", "error") for s in statuses.values())

    if all_ready:
        st.success("All three indexes are READY.")
    else:
        if any_missing:
            if st.button("Create missing indexes", type="primary"):
                for name, status in statuses.items():
                    if status in ("missing", "error"):
                        with st.spinner(f"Creating `{name}`…"):
                            try:
                                create_index(name)
                                st.info(
                                    f"`{name}` creation initiated — Atlas builds indexes "
                                    "asynchronously. Refresh in ~1–2 minutes."
                                )
                            except Exception as exc:  # noqa: BLE001
                                st.error(f"Failed to create `{name}`: {exc}")

        if any(s == "building" for s in statuses.values()):
            st.info("One or more indexes are still building. Refresh the page to check status.")
            if st.button("Refresh status"):
                st.rerun()

# ---------------------------------------------------------------------------
# Representative document viewer
# ---------------------------------------------------------------------------

st.header("Representative document viewer")

emb_count = embedding_count()
if emb_count == 0:
    st.info("No documents with embeddings yet. Complete Steps 1 & 2 first.")
else:
    col = benchmark_col()

    # Fetch a few names for the selectbox
    sample_names = [
        d["name"]
        for d in col.find(
            {EMBEDDING_FIELD: {"$exists": True}},
            {"name": 1},
            limit=200,
        )
        if d.get("name")
    ]

    selected_name = st.selectbox("Pick a listing", sorted(set(sample_names)))

    if selected_name:
        doc = col.find_one({"name": selected_name})
        if doc:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.subheader(doc.get("name", ""))
                st.caption(
                    f"{doc.get('property_type', '')} / {doc.get('room_type', '')} / "
                    f"{doc.get('bedrooms', '?')} bed"
                )
                st.markdown("**`listing_text` (free-text composite field)**")
                st.text_area(
                    label="listing_text",
                    value=doc.get(LISTING_TEXT_FIELD, ""),
                    height=200,
                    label_visibility="collapsed",
                )

            with c2:
                emb = doc.get(EMBEDDING_FIELD)
                if emb:
                    st.markdown(
                        f"**`{EMBEDDING_FIELD}`** — {len(emb):,} dimensions, "
                        f"float32, {len(emb) * 4:,} bytes"
                    )
                    render_embedding_preview(emb, label=EMBEDDING_FIELD, n_dims=32)
                    with st.expander("Raw first 10 values"):
                        for i, v in enumerate(emb[:10]):
                            st.text(f"  [{i:4d}]  {v:.8f}")
                else:
                    st.info("This document does not have an embedding yet.")

            with st.expander("Raw document (excluding embedding field)"):
                display_doc = {k: v for k, v in doc.items() if k != EMBEDDING_FIELD}
                st.json(display_doc, expanded=False)
