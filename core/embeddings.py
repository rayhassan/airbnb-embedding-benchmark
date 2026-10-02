"""
core/embeddings.py
------------------
Voyage AI embedding wrapper.

Key design decisions:
- Documents are ALWAYS embedded with voyage-4-large (input_type="document").
  This happens once during Setup.
- Queries are embedded at runtime with whichever model is selected by the user
  (input_type="query"), enabling asymmetric retrieval comparisons.
- All voyage-4 series models share the same vector space, so a query vector
  from voyage-4-lite is valid against an index built from voyage-4-large vectors.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import voyageai
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

DOCUMENT_MODEL = "voyage-4-large"

QUERY_MODELS: dict[str, dict] = {
    "voyage-4-large": {
        "label": "voyage-4-large",
        "description": "Best quality — same model as index (symmetric)",
        "dims": 1024,
        # USD per 1M tokens (source: Voyage AI pricing, mid-2026)
        "price_per_1m": 0.12,
        "color": "#1f77b4",
    },
    "voyage-4": {
        "label": "voyage-4",
        "description": "Mid-tier — optimised for quality/cost balance (asymmetric)",
        "dims": 1024,
        "price_per_1m": 0.08,
        "color": "#ff7f0e",
    },
    "voyage-4-lite": {
        "label": "voyage-4-lite",
        "description": "Cheapest/fastest — latency & cost optimised (asymmetric)",
        "dims": 1024,
        "price_per_1m": 0.05,
        "color": "#2ca02c",
    },
}

DOCUMENT_MODEL_PRICE_PER_1M = QUERY_MODELS[DOCUMENT_MODEL]["price_per_1m"]
EMBEDDING_DIMS = 1024          # fixed output dimension for all models used here
BATCH_SIZE = 100               # Voyage recommends ≤128 texts per call; 100 is safe
MAX_CHARS_PER_DOC = 40_000     # ~10K tokens; Voyage truncates at 32K tokens anyway


# ---------------------------------------------------------------------------
# Internal client factory (not cached — voyageai.Client is lightweight)
# ---------------------------------------------------------------------------

def _client() -> voyageai.Client:
    api_key = os.getenv("VOYAGE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "VOYAGE_API_KEY is not set. Copy .env.template → .env and add your key."
        )
    return voyageai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QueryEmbeddingResult:
    vector: list[float]
    tokens: int
    latency_ms: float
    model: str
    estimated_cost_usd: float


@dataclass
class BatchEmbeddingResult:
    vectors: list[list[float]]
    total_tokens: int
    latency_ms: float
    model: str
    estimated_cost_usd: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_query(text: str, model: str = DOCUMENT_MODEL) -> QueryEmbeddingResult:
    """
    Embed a single query string.  Uses input_type='query' so Voyage prepends
    the retrieval prompt — important for asymmetric retrieval quality.
    """
    if model not in QUERY_MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(QUERY_MODELS)}")

    client = _client()
    t0 = time.perf_counter()
    result = client.embed(
        [text],
        model=model,
        input_type="query",
        output_dimension=EMBEDDING_DIMS,
        truncation=True,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    tokens = result.total_tokens
    cost = _compute_cost(tokens, model)

    return QueryEmbeddingResult(
        vector=result.embeddings[0],
        tokens=tokens,
        latency_ms=elapsed_ms,
        model=model,
        estimated_cost_usd=cost,
    )


def embed_documents_batch(
    texts: list[str],
    model: str = DOCUMENT_MODEL,
    progress_callback=None,
) -> BatchEmbeddingResult:
    """
    Embed a list of document strings in batches of BATCH_SIZE.

    progress_callback(completed: int, total: int) is called after each batch
    if provided — use this to drive a Streamlit progress bar.

    Always uses input_type='document'.
    """
    if model not in QUERY_MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {list(QUERY_MODELS)}")

    # Voyage rejects empty strings with a 400 error — replace any that slipped through
    _FALLBACK = "listing"
    texts = [t.strip() if t and t.strip() else _FALLBACK for t in texts]

    client = _client()
    all_vectors: list[list[float]] = []
    total_tokens = 0
    t0 = time.perf_counter()

    for i in range(0, len(texts), BATCH_SIZE):
        batch = [t[:MAX_CHARS_PER_DOC] for t in texts[i : i + BATCH_SIZE]]
        result = client.embed(
            batch,
            model=model,
            input_type="document",
            output_dimension=EMBEDDING_DIMS,
            truncation=True,
        )
        all_vectors.extend(result.embeddings)
        total_tokens += result.total_tokens

        if progress_callback:
            progress_callback(min(i + BATCH_SIZE, len(texts)), len(texts))

    elapsed_ms = (time.perf_counter() - t0) * 1_000
    cost = _compute_cost(total_tokens, model)

    return BatchEmbeddingResult(
        vectors=all_vectors,
        total_tokens=total_tokens,
        latency_ms=elapsed_ms,
        model=model,
        estimated_cost_usd=cost,
    )


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def _compute_cost(tokens: int, model: str) -> float:
    price = QUERY_MODELS[model]["price_per_1m"]
    return (tokens / 1_000_000) * price


def estimate_setup_cost(n_docs: int, avg_tokens_per_doc: int = 700) -> dict:
    """
    Return a cost estimate dict for the Setup page pre-flight warning.
    Only voyage-4-large is used for document embedding.
    """
    total_tokens = n_docs * avg_tokens_per_doc
    cost = _compute_cost(total_tokens, DOCUMENT_MODEL)
    return {
        "model": DOCUMENT_MODEL,
        "n_docs": n_docs,
        "avg_tokens_per_doc": avg_tokens_per_doc,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost,
    }
