"""
core/db.py
----------
Cached MongoClient singleton and collection helpers.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError

load_dotenv(override=True)

SOURCE_DB = "sample_airbnb"
SOURCE_COLLECTION = "listingsAndReviews"
BENCHMARK_COLLECTION = "listings_benchmark"

EMBEDDING_FIELD = "emb_large"          # single document embedding field
LISTING_TEXT_FIELD = "listing_text"    # concatenated free-text field


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_client() -> MongoClient:
    uri = os.getenv("MONGODB_URI", "")
    if not uri:
        raise EnvironmentError(
            "MONGODB_URI is not set. Copy .env.template → .env and fill in your credentials."
        )
    return MongoClient(
        uri,
        maxPoolSize=5,                  # Streamlit is low-concurrency
        serverSelectionTimeoutMS=5_000,
        socketTimeoutMS=60_000,         # vector search can take a moment on large collections
        connectTimeoutMS=10_000,
    )


def ping() -> tuple[bool, str]:
    """
    Return (ok, cluster_name).
    Extracts the Atlas cluster name from the 'hello' command response —
    specifically the hostname in 'me', e.g.:
      'sandbox-shard-00-02.fgkzb.mongodb.net:27017' → 'sandbox'
    Falls back to the MONGODB_URI hostname stem if the command fails.
    """
    try:
        client = get_client()
        info = client.admin.command("ping")
        ok = info.get("ok") == 1.0

        try:
            hello = client.admin.command("hello")
            # 'me' is the hostname of the node we're connected to, e.g.:
            # 'sandbox-shard-00-02.fgkzb.mongodb.net:27017'
            me = hello.get("me", "")
            host = me.split(":")[0]          # drop port
            cluster = host.split(".")[0]     # first label, e.g. 'sandbox-shard-00-02'
            # Strip the shard/node suffix: 'sandbox-shard-00-02' → 'sandbox'
            parts = cluster.split("-")
            # Walk back from the end dropping numeric and 'shard' tokens
            while parts and (parts[-1].isdigit() or parts[-1] == "shard"):
                parts.pop()
            cluster_name = "-".join(parts) if parts else cluster
        except Exception:  # noqa: BLE001
            # Fallback: parse cluster name from the URI host stem
            uri = os.getenv("MONGODB_URI", "")
            try:
                from urllib.parse import urlparse
                host = urlparse(uri).hostname or ""
                cluster_name = host.split(".")[0] if host else "Atlas"
            except Exception:  # noqa: BLE001
                cluster_name = "Atlas"

        return ok, cluster_name
    except ServerSelectionTimeoutError as exc:
        return False, str(exc)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------

def source_col() -> Collection:
    return get_client()[SOURCE_DB][SOURCE_COLLECTION]


def benchmark_col() -> Collection:
    return get_client()[SOURCE_DB][BENCHMARK_COLLECTION]


# ---------------------------------------------------------------------------
# Setup-state queries
# ---------------------------------------------------------------------------

def benchmark_count() -> int:
    """Number of documents in the benchmark collection."""
    try:
        return benchmark_col().estimated_document_count()
    except Exception:  # noqa: BLE001
        return 0


def embedding_count() -> int:
    """Number of docs in benchmark collection that already have the embedding field."""
    try:
        return benchmark_col().count_documents({EMBEDDING_FIELD: {"$exists": True}})
    except Exception:  # noqa: BLE001
        return 0


def index_statuses() -> dict[str, str]:
    """
    Return a dict of {index_name: status} for all known benchmark indexes.
    Status is one of: 'ready', 'building', 'missing'.
    """
    from core.indexes import INDEX_DEFINITIONS

    try:
        col = benchmark_col()
        existing = {idx["name"]: idx for idx in col.list_search_indexes()}
    except Exception:  # noqa: BLE001
        existing = {}

    result = {}
    for name in INDEX_DEFINITIONS:
        if name not in existing:
            result[name] = "missing"
        elif existing[name].get("status") == "READY":
            result[name] = "ready"
        else:
            result[name] = "building"
    return result


def build_listing_text(doc: dict) -> str:
    """
    Concatenate free-text fields from a listingsAndReviews document into a
    single string suitable for embedding.  Single-value fields (price,
    bedrooms, amenities, etc.) are intentionally excluded — they are better
    served by CRUD/filter operations.
    """
    parts = [
        doc.get("name", ""),
        doc.get("summary", ""),
        doc.get("description", ""),
        doc.get("space", ""),
        doc.get("neighborhood_overview", ""),
        doc.get("notes", ""),
        doc.get("transit", ""),
        doc.get("access", ""),
        doc.get("interaction", ""),
        doc.get("house_rules", ""),
    ]
    return " ".join(p.strip() for p in parts if p and p.strip())
