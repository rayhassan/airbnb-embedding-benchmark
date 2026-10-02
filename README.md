# Airbnb Embedding Benchmark

A Streamlit app for benchmarking **Voyage AI embedding models** and **MongoDB Atlas Vector Search** quantization strategies, built on the `sample_airbnb.listingsAndReviews` dataset.

The app demonstrates two independent axes of comparison:

- **Asymmetric retrieval** — documents indexed once with `voyage-4-large`; queries issued with `voyage-4-large`, `voyage-4`, and `voyage-4-lite` against the same index, showing the quality/cost trade-off of using a cheaper query model against a premium corpus index.
- **Atlas-side quantization** — the same float32 document vectors indexed three ways (`none`, `scalar`, `binary`), comparing latency, recall@k, and storage savings at each compression level.

---

## Architecture

### Core principle

Documents are embedded **once** using `voyage-4-large` and stored in a single field (`emb_large`). All voyage-4 series models share the same vector space, so query vectors from any model in the series are geometrically compatible with the indexed document vectors — no re-embedding required.

```
Document embedding (Setup, once)
  voyage-4-large → emb_large (1024-dim float32)

Atlas indexes (all on the same field)
  vs_none    → float32,  cosine     (baseline)
  vs_scalar  → int8,     cosine     (~4× smaller)
  vs_binary  → 1-bit,    euclidean  (~32× smaller)

Query embedding (runtime, per search)
  voyage-4-large  → vs_none  (symmetric baseline)
  voyage-4        → vs_none  (asymmetric, mid-tier)
  voyage-4-lite   → vs_none  (asymmetric, cheapest)
```

### Embedding field

The `listing_text` field is built by concatenating the free-text fields from each listing:

```
name · summary · description · space · neighborhood_overview · notes · transit · access · interaction · house_rules
```

Single-value fields (`price`, `bedrooms`, `amenities`, etc.) are left as-is for CRUD/filter use and are not embedded.

### Project structure

```
airbnb-embedding-benchmark/
├── app.py                      # Home page — connection status, setup summary
├── pages/
│   ├── 1_Setup.py              # Copy collection, generate embeddings, create indexes
│   ├── 2_Model_Comparison.py   # Asymmetric retrieval: 3 query models → vs_none
│   ├── 3_Quantization.py       # Quantization: voyage-4-large → none/scalar/binary
│   └── 4_Inspector.py          # Browse any listing, inspect embedding, find similars
├── core/
│   ├── db.py                   # Cached MongoClient, collection helpers
│   ├── embeddings.py           # Voyage AI wrapper (embed_query / embed_documents_batch)
│   ├── indexes.py              # Atlas Vector Search index definitions and creation
│   ├── metrics.py              # Latency, cost, recall@k, score stats, search runner
│   └── sample_queries.py       # Six pre-built demo queries
├── utils/
│   └── display.py              # Shared Streamlit UI components
├── pyproject.toml              # uv project manifest
├── uv.lock                     # Reproducible dependency lockfile
├── .python-version             # Pins Python 3.13
└── .env.template               # Credential template
```

---

## Prerequisites

| Requirement | Notes |
|---|---|
| [MongoDB Atlas](https://www.mongodb.com/atlas) account | Free tier (M0) is sufficient for setup; M10+ recommended for production-scale vector search |
| `sample_airbnb` dataset loaded | Available via **Load Sample Dataset** in the Atlas UI |
| [Voyage AI](https://www.voyageai.com) API key | Free tier available at [dash.voyageai.com](https://dash.voyageai.com) |
| [uv](https://docs.astral.sh/uv/) | Fast Python package manager — `curl -LsSf https://astral.sh/uv/install.sh \| sh` |

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/airbnb-embedding-benchmark.git
cd airbnb-embedding-benchmark
```

### 2. Configure credentials

```bash
cp .env.template .env
```

Edit `.env`:

```env
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
VOYAGE_API_KEY=pa-...
```

> **Important:** If your password contains special characters (e.g. `$`, `@`, `/`), percent-encode them using [`urllib.parse.quote_plus`](https://docs.python.org/3/library/urllib.parse.html#urllib.parse.quote_plus).  For example, `p@ss$word` becomes `p%40ss%24word`.

> **Important:** Do not export `MONGODB_URI` in your shell profile. If the variable is already set in your environment with a raw (un-encoded) password, `load_dotenv(override=True)` will replace it with the correctly encoded value from `.env` — but an already-exported broken value takes precedence in some shells. Use `unset MONGODB_URI` in your terminal session if you see connection errors after editing `.env`.

### 3. Install dependencies and run

```bash
# Install all dependencies into an isolated .venv (downloads Python 3.13 if needed)
uv sync

# Launch the app
uv run streamlit run app.py

# With auto-reload on save (development)
uv run streamlit run app.py --server.runOnSave true
```

---

## Setup walkthrough

Open the **Setup** page and complete the three steps in order:

### Step 1 — Copy collection

Copies `sample_airbnb.listingsAndReviews` → `sample_airbnb.listings_benchmark`, adding the computed `listing_text` field. The operation is idempotent — safe to re-run.

### Step 2 — Generate embeddings

Embeds each listing's `listing_text` using `voyage-4-large` (`input_type="document"`) in batches of 100. Results are written back to the collection as `emb_large` (1024-dim float32).

**Cost estimate:** ~14,000 documents × ~700 tokens ≈ 9.8M tokens ≈ **~$1.18 USD** at `voyage-4-large` pricing. The page displays an estimate and requires a confirmation checkbox before calling the API.

Documents with no usable free-text are assigned a minimal placeholder vector and tagged with `emb_large_placeholder: true`.

### Step 3 — Create indexes

Creates three Atlas Vector Search indexes on `emb_large`, each with `property_type`, `room_type`, and `bedrooms` as pre-filter fields:

| Index | Quantization | Similarity | Storage (est.) |
|---|---|---|---|
| `vs_none` | none (float32) | cosine | ~54 MB |
| `vs_scalar` | scalar (int8) | cosine | ~14 MB |
| `vs_binary` | binary (1-bit) | euclidean | ~1.8 MB |

Atlas builds indexes asynchronously. Status polls automatically — refresh until all three show **READY**.

---

## Pages

### Home

Connection status (shows the Atlas cluster name), setup progress summary, and an architecture reference panel.

### Model Comparison

Enter a query (or pick one of six built-in demo queries). The same query is embedded independently with all three models and run against `vs_none`:

| Panel | Query model | Index | Type |
|---|---|---|---|
| Left | `voyage-4-large` | `vs_none` | Symmetric baseline |
| Centre | `voyage-4` | `vs_none` | Asymmetric |
| Right | `voyage-4-lite` | `vs_none` | Asymmetric |

**Metrics per panel:** embed latency, DB latency, total latency, token count, cost per query, cost per 1M queries.

**Analysis tabs:** score distribution, latency breakdown, recall@k vs baseline, cost comparison table.

**Controls (sidebar):** `numCandidates` slider (10–500, default 150), results limit, optional pre-filters by property type, room type, and bedroom count.

### Quantization

The query is embedded **once** with `voyage-4-large`, then the same vector is sent to all three indexes:

| Panel | Index | Quantization | Similarity |
|---|---|---|---|
| Left | `vs_none` | none | cosine |
| Centre | `vs_scalar` | scalar | cosine |
| Right | `vs_binary` | binary | euclidean |

> Scores from `vs_binary` use euclidean similarity and are **not directly comparable** to cosine scores — only recall@k is a fair cross-configuration accuracy metric.

**Analysis tabs:** score distribution, latency breakdown, recall@k vs float32 baseline, storage size estimate chart.

### Inspector

Browse any listing by name. For each listing the page shows:

- The `listing_text` composite field with per-source-field breakdown
- The `emb_large` vector with an adjustable dimension preview chart and statistics (min, max, mean, L2 norm)
- The full raw document (minus the embedding field)
- **Find similar listings** — uses the document's own `emb_large` vector as the query to retrieve nearest neighbours from any of the three indexes

---

## Demo queries

Six pre-built queries are available as a selectbox on the Model Comparison and Quantization pages:

| Query | Intent | Fields exercised |
|---|---|---|
| Beachside retreat | Coastal listing with transport + cafes | `neighbourhood`, `transit` |
| City-centre remote work | Urban flat, desk, metro access | `transit`, `space` |
| Family beach holiday | Large home, pool/garden, parking | bedroom count, amenities |
| Boutique harbour views | Premium/unique — villa, boat, views | rare property types |
| Budget solo traveller | Private room, lively area, friendly host | `notes`, `interaction`, `house_rules` |
| Peaceful rural escape | Countryside/nature, quiet, disconnect | `neighborhood_overview`, `description` |

---

## Dependencies

Managed by [uv](https://docs.astral.sh/uv/). See `pyproject.toml` for version constraints and `uv.lock` for exact pinned versions.

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pymongo` | MongoDB driver |
| `voyageai` | Voyage AI embedding API client |
| `python-dotenv` | `.env` credential loading |
| `pandas` | Tabular data for comparison tables |
| `numpy` | Vector statistics |
| `plotly` | Interactive charts |

---

## Cost reference

All costs are for **query-side embedding only** (Voyage API). Document embedding is a one-time Setup cost.

| Model | Price (per 1M tokens) | Typical query cost | Cost per 1M queries |
|---|---|---|---|
| `voyage-4-large` | $0.12 | ~$0.000084 | ~$84 |
| `voyage-4` | $0.08 | ~$0.000056 | ~$56 |
| `voyage-4-lite` | $0.05 | ~$0.000035 | ~$35 |

*Assumes ~700 tokens per query. Voyage pricing as of mid-2026 — verify at [docs.voyageai.com](https://docs.voyageai.com).*

---

## Key design decisions

**One embedding field, three indexes.** Documents are embedded once with the best available model. Atlas-side quantization compresses the index representation without touching the stored float32 values — all three indexes point at the same `emb_large` field.

**Shared vector space enables asymmetric retrieval.** All `voyage-4` series models (`voyage-4-large`, `voyage-4`, `voyage-4-lite`) produce vectors in the same geometric space. A query vector from `voyage-4-lite` is directly comparable to document vectors from `voyage-4-large` — making budget query models a legitimate production option against a premium index.

**`input_type` matters.** Query embeddings use `input_type="query"` and document embeddings use `input_type="document"`. Voyage prepends different prompts for each, improving retrieval quality in asymmetric scenarios.

**Binary quantization requires euclidean similarity.** `vs_binary` uses `euclidean` rather than `cosine` — this is a hard Atlas requirement for binary-quantized indexes. Scores from binary indexes are on a different scale and should not be compared directly to cosine scores; use recall@k instead.

**`load_dotenv(override=True)`.** All modules use `override=True` so that the correctly encoded `.env` values always take precedence over any stale shell-exported variables.

---

## License

MIT
