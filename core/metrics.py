"""
core/metrics.py
---------------
Latency measurement, token cost calculation, recall@k, and score statistics.
Also contains the vector search runner that wraps pymongo aggregate() calls
and returns results + timing in one shot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.db import EMBEDDING_FIELD, benchmark_col
from core.embeddings import QUERY_MODELS


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """One document returned from a $vectorSearch query."""
    id: str
    name: str
    score: float
    property_type: str
    room_type: str
    bedrooms: Optional[float]
    price: Optional[float]
    neighbourhood: str
    listing_text_snippet: str  # first 300 chars of listing_text
    raw_doc: dict = field(default_factory=dict, repr=False)


@dataclass
class SearchRun:
    """Full result of a single $vectorSearch execution."""
    results: list[SearchResult]
    latency_ms: float
    query_embed_ms: float       # time spent calling Voyage API
    db_query_ms: float          # time spent in MongoDB aggregate()
    tokens_used: int
    estimated_cost_usd: float   # embedding API cost only
    model: str
    index_name: str
    num_candidates: int
    limit: int
    query_text: str


@dataclass
class ScoreStats:
    count: int
    top: float
    mean: float
    std: float
    minimum: float


# ---------------------------------------------------------------------------
# Score statistics
# ---------------------------------------------------------------------------

def score_stats(results: list[SearchResult]) -> ScoreStats:
    if not results:
        return ScoreStats(0, 0.0, 0.0, 0.0, 0.0)
    scores = [r.score for r in results]
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    std = variance ** 0.5
    return ScoreStats(
        count=n,
        top=max(scores),
        mean=round(mean, 4),
        std=round(std, 4),
        minimum=min(scores),
    )


# ---------------------------------------------------------------------------
# Recall / overlap
# ---------------------------------------------------------------------------

def recall_at_k(ids_a: list[str], ids_b: list[str], k: int) -> float:
    """
    Fraction of the top-k results from set A that appear anywhere in set B.
    Used to measure how much quality is lost when switching query model or
    index quantization vs. a baseline.

    recall@k = |top-k(A) ∩ top-k(B)| / k
    """
    if k == 0:
        return 0.0
    set_a = set(ids_a[:k])
    set_b = set(ids_b[:k])
    return len(set_a & set_b) / k


def overlap_matrix(
    runs: dict[str, SearchRun], k: int
) -> dict[tuple[str, str], float]:
    """
    Compute pairwise recall@k between all run pairs.
    Returns {(label_a, label_b): recall} for all (a, b) combinations.
    """
    labels = list(runs.keys())
    result = {}
    for i, la in enumerate(labels):
        for lb in labels[i:]:
            ids_a = [r.id for r in runs[la].results]
            ids_b = [r.id for r in runs[lb].results]
            r = recall_at_k(ids_a, ids_b, k)
            result[(la, lb)] = r
            result[(lb, la)] = r
    return result


# ---------------------------------------------------------------------------
# Token cost
# ---------------------------------------------------------------------------

def token_cost_usd(tokens: int, model: str) -> float:
    price = QUERY_MODELS[model]["price_per_1m"]
    return (tokens / 1_000_000) * price


# ---------------------------------------------------------------------------
# MongoDB vector search runner
# ---------------------------------------------------------------------------

def run_vector_search(
    query_vector: list[float],
    index_name: str,
    num_candidates: int,
    limit: int,
    pre_filter: Optional[dict] = None,
) -> tuple[list[SearchResult], float]:
    """
    Execute a $vectorSearch aggregation and return (results, db_latency_ms).
    """
    vector_search_stage: dict[str, Any] = {
        "$vectorSearch": {
            "index": index_name,
            "path": EMBEDDING_FIELD,
            "queryVector": query_vector,
            "numCandidates": num_candidates,
            "limit": limit,
        }
    }
    if pre_filter:
        vector_search_stage["$vectorSearch"]["filter"] = pre_filter

    pipeline = [
        vector_search_stage,
        {
            "$project": {
                "_id": 1,
                "name": 1,
                "property_type": 1,
                "room_type": 1,
                "bedrooms": 1,
                "price": 1,
                "address": 1,
                "listing_text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    col = benchmark_col()
    t0 = time.perf_counter()
    raw = list(col.aggregate(pipeline))
    db_ms = (time.perf_counter() - t0) * 1_000

    results = []
    for doc in raw:
        addr = doc.get("address") or {}
        neighbourhood = (
            addr.get("neighbourhood_cleansed")
            or addr.get("suburb")
            or addr.get("market")
            or addr.get("country")
            or ""
        )
        price_val = doc.get("price")
        if hasattr(price_val, "to_decimal"):
            price_val = float(price_val.to_decimal())

        listing_text = doc.get("listing_text", "")
        results.append(
            SearchResult(
                id=str(doc["_id"]),
                name=doc.get("name", ""),
                score=round(doc.get("score", 0.0), 4),
                property_type=doc.get("property_type", ""),
                room_type=doc.get("room_type", ""),
                bedrooms=doc.get("bedrooms"),
                price=price_val,
                neighbourhood=neighbourhood,
                listing_text_snippet=listing_text[:300],
                raw_doc=doc,
            )
        )

    return results, db_ms


def full_search_run(
    query_text: str,
    query_vector: list[float],
    query_embed_ms: float,
    tokens_used: int,
    query_model: str,
    index_name: str,
    num_candidates: int,
    limit: int,
    pre_filter: Optional[dict] = None,
) -> SearchRun:
    """
    Orchestrate a complete search run given a pre-computed query vector.
    Separates embedding latency from DB latency so they can be shown independently.
    """
    results, db_ms = run_vector_search(
        query_vector=query_vector,
        index_name=index_name,
        num_candidates=num_candidates,
        limit=limit,
        pre_filter=pre_filter,
    )
    cost = token_cost_usd(tokens_used, query_model)

    return SearchRun(
        results=results,
        latency_ms=query_embed_ms + db_ms,
        query_embed_ms=round(query_embed_ms, 1),
        db_query_ms=round(db_ms, 1),
        tokens_used=tokens_used,
        estimated_cost_usd=cost,
        model=query_model,
        index_name=index_name,
        num_candidates=num_candidates,
        limit=limit,
        query_text=query_text,
    )
