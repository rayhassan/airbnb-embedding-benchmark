"""
core/indexes.py
---------------
Atlas Vector Search index definitions and creation helpers.

Three indexes, all on the same field (emb_large), differing only in
Atlas-side quantization:

  vs_none    — full float32 precision  (cosine similarity)
  vs_scalar  — scalar int8 quantization (cosine similarity, ~4× smaller)
  vs_binary  — binary quantization      (euclidean similarity, ~32× smaller)

Filter fields added to each index: property_type, room_type, bedrooms
(enables optional pre-filtering in the UI without post-filter cost).
"""

from __future__ import annotations

import time
from typing import Any

from core.db import EMBEDDING_FIELD, benchmark_col

# ---------------------------------------------------------------------------
# Index definitions
# ---------------------------------------------------------------------------

_FILTER_FIELDS: list[dict] = [
    {"type": "filter", "path": "property_type"},
    {"type": "filter", "path": "room_type"},
    {"type": "filter", "path": "bedrooms"},
]

# Binary quantization MUST use euclidean (per Atlas docs).
# none and scalar both use cosine.
INDEX_DEFINITIONS: dict[str, dict[str, Any]] = {
    "vs_none": {
        "label": "None (float32 baseline)",
        "description": "Full float32 precision — highest accuracy, largest storage",
        "quantization": "none",
        "similarity": "cosine",
        "color": "#1f77b4",
        "fields": [
            {
                "type": "vector",
                "path": EMBEDDING_FIELD,
                "numDimensions": 1024,
                "similarity": "cosine",
                # quantization omitted → defaults to none
            },
            *_FILTER_FIELDS,
        ],
    },
    "vs_scalar": {
        "label": "Scalar (int8, ~4× smaller)",
        "description": "Atlas compresses float32 → int8 at query time — good accuracy/speed balance",
        "quantization": "scalar",
        "similarity": "cosine",
        "color": "#ff7f0e",
        "fields": [
            {
                "type": "vector",
                "path": EMBEDDING_FIELD,
                "numDimensions": 1024,
                "similarity": "cosine",
                "quantization": "scalar",
            },
            *_FILTER_FIELDS,
        ],
    },
    "vs_binary": {
        "label": "Binary (1-bit, ~32× smaller)",
        "description": "Atlas compresses to single bits — max compression, euclidean similarity required",
        "quantization": "binary",
        "similarity": "euclidean",
        "color": "#2ca02c",
        "fields": [
            {
                "type": "vector",
                "path": EMBEDDING_FIELD,
                "numDimensions": 1024,
                "similarity": "euclidean",
                "quantization": "binary",
            },
            *_FILTER_FIELDS,
        ],
    },
}

# ---------------------------------------------------------------------------
# Creation helpers
# ---------------------------------------------------------------------------

def create_index(name: str) -> None:
    """Create a single vector search index by name. Idempotent — skips if already present."""
    if name not in INDEX_DEFINITIONS:
        raise ValueError(f"Unknown index '{name}'. Available: {list(INDEX_DEFINITIONS)}")

    col = benchmark_col()
    definition = INDEX_DEFINITIONS[name]

    # Check if already exists
    existing = {idx["name"] for idx in col.list_search_indexes()}
    if name in existing:
        return  # already exists — nothing to do

    col.create_search_index(
        model={
            "name": name,
            "type": "vectorSearch",
            "definition": {"fields": definition["fields"]},
        }
    )


def create_all_indexes() -> None:
    """Create all three vector search indexes."""
    for name in INDEX_DEFINITIONS:
        create_index(name)


def drop_index(name: str) -> None:
    """Drop a vector search index by name (for re-creation)."""
    col = benchmark_col()
    col.drop_search_index(name)


def get_index_status(name: str) -> str:
    """
    Return one of: 'ready', 'building', 'missing', 'error'.
    Polls the list_search_indexes cursor.
    """
    try:
        col = benchmark_col()
        for idx in col.list_search_indexes():
            if idx["name"] == name:
                status = idx.get("status", "").upper()
                if status == "READY":
                    return "ready"
                return "building"
        return "missing"
    except Exception:  # noqa: BLE001
        return "error"


def wait_for_index(name: str, timeout_s: int = 300, poll_s: int = 5) -> bool:
    """
    Block until the named index reaches READY status or timeout expires.
    Returns True if ready, False if timed out.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if get_index_status(name) == "ready":
            return True
        time.sleep(poll_s)
    return False
