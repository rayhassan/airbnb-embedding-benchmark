# AGENTS.md

Instructions for AI coding agents working in this repository.
`CLAUDE.md` agents: this file is your source of truth — use `@AGENTS.md` to import it.

---

## Project Domain

This is a **Streamlit benchmarking application** for comparing vector search quality and cost
on MongoDB Atlas using the `sample_airbnb.listingsAndReviews` dataset (~14,000 listings).

It benchmarks two independent axes:

1. **Asymmetric retrieval (Model Comparison):** Documents are embedded once with
   `voyage-4-large`. At query time, three different Voyage AI models (`voyage-4-large`,
   `voyage-4`, `voyage-4-lite`) embed the same query and search the *same* float32 index
   (`vs_none`). This tests whether cheaper models produce acceptably similar results, since
   all `voyage-4`-series models share a vector space.

2. **Atlas-side quantization (Quantization):** A single `voyage-4-large` query vector is
   sent to *three different indexes* (`vs_none` float32 / cosine, `vs_scalar` int8 / cosine,
   `vs_binary` 1-bit / euclidean). The page measures recall@k, score drift, and latency.

Key invariants:
- The embedding field on every document is `emb_large` (1024 dims, float32, `voyage-4-large`).
- The three Atlas Vector Search indexes are named `vs_none`, `vs_scalar`, and `vs_binary`.
- Binary indexes **must** use euclidean distance; scalar and none indexes use cosine.
- `listing_text` is a synthetic field built from 10 source fields; never modify its construction
  logic in `core/db.py::build_listing_text()` without updating the Setup page copy step.
- Two external credentials are required: `MONGODB_URI` and `VOYAGE_API_KEY` — loaded from
  `.env` via `python-dotenv`. Never hardcode credentials or commit `.env`.

---

## Project Setup and Management

- **Python version:** 3.13 (pinned in `.python-version`). Do not use syntax from newer versions.
- **Package manager:** `uv` with `pyproject.toml`. Never use `pip`, `pip install`, or
  `requirements.txt`.
- **Add a dependency:** `uv add <package>` — ask before adding any new dependency.
- **Run the app:** `uv run streamlit run app.py`
- **Sync the environment:** `uv sync`
- **Git workflow:** Branch from `main` as `feature/<name>`. Use Conventional Commits
  (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`). Stage changes for review — do not
  commit to `main` or push without being asked.

---

## Project Structure

```
app.py                    # Home page — entry point; must call st.set_page_config() first
Embedding_Benchmark.py    # Legacy duplicate of app.py — do not modify
pages/
  1_Setup.py              # 3-step wizard: copy collection, embed docs, create indexes
  2_Model_Comparison.py   # Asymmetric retrieval: 3 query models vs 1 index
  3_Quantization.py       # Quantization: 1 vector vs 3 indexes
  4_Inspector.py          # Browse listing, inspect embedding, find similar docs
core/
  db.py                   # MongoClient singleton, collection helpers, build_listing_text()
  embeddings.py           # Voyage AI wrapper, QUERY_MODELS registry, embed_query/batch
  indexes.py              # INDEX_DEFINITIONS, create/drop/status helpers
  metrics.py              # SearchResult/SearchRun dataclasses, run_vector_search, recall@k
  sample_queries.py       # 6 pre-built demo queries with 'why' annotations
utils/
  display.py              # All shared Streamlit UI components (charts, tables, cards)
```

Rules:
- `core/` modules must not import from `streamlit` directly (except `db.py` which uses
  `@st.cache_resource`). Keep business logic free of UI dependencies.
- `utils/display.py` is the only place to add shared UI components. Do not duplicate
  rendering logic across page files.
- Do not create new packages or top-level files without being asked.
- `Embedding_Benchmark.py` is a legacy alias — never edit it; always edit `app.py`.

---

## Coding Conventions

- **Type hints:** Add type annotations to all public functions and methods, including return
  types. Use `from __future__ import annotations` if needed for forward references.
- **Dataclasses:** Use `@dataclass` for structured results (`SearchResult`, `SearchRun`).
  Prefer dataclasses over plain dicts for data passed between functions.
- **Path handling:** Use `pathlib.Path` for any file I/O. Never use `os.path`.
- **String formatting:** Use f-strings. Never use `str.format()` or `%` formatting.
- **Error handling:** Follow EAFP — use `try/except` rather than pre-checking conditions.
- **Docstrings:** Write Google-style docstrings for every public function and method.
- **Imports:** Standard library first, then third-party, then local `core/` and `utils/`
  imports. One blank line between groups.
- **Constants:** Define at module level in `UPPER_SNAKE_CASE`. No magic numbers in
  functions — name them.
- **Comprehensions:** Prefer list/dict comprehensions and generators over explicit loops
  where idiomatic. Avoid comprehensions that span more than two lines.
- **No mutable defaults:** Never use mutable objects (list, dict) as default argument values.

---

## Streamlit Conventions

- **`st.set_page_config()` must be the first Streamlit call** in every page file, before any
  other `st.*` call or import that triggers Streamlit.
- **Cache resources** (MongoClient, API clients) with `@st.cache_resource`.
- **Cache pure data** (filter options, listing names, fetched documents) with
  `@st.cache_data(ttl=<seconds>)`. Choose TTL based on how stale the data can be.
- **Guard clauses:** Use `st.stop()` to halt rendering when a precondition fails
  (e.g., index not ready, query empty, connection failed). Never use a deeply nested
  `if/else` tree for page-level guards.
- **`st.rerun()`** only after a state-mutating action (setup step completed, doc written).
  Do not call it inside a loop or on every render.
- **Progress:** Use `st.progress()` for batch operations. Wrap all external API calls and
  slow DB queries in `st.spinner("...")`.
- **Unsafe HTML:** Custom CSS and metric card HTML may use `st.markdown(..., unsafe_allow_html=True)`.
  Inject shared styles once per page render, not inside a loop.
- **Multi-page layout:** Follow the existing `pages/` convention with numeric prefixes for
  sidebar ordering (`1_`, `2_`, etc.). Do not switch to `st.navigation()` / `st.Page()` API
  without being asked — the project uses the legacy directory convention.
- **Sidebar:** Put search controls (sliders, multiselects) in `st.sidebar`. Keep the main
  area for results and analysis.
- **Charts:** Use `plotly` for all charts — box plots, stacked bar, bar. Always pass
  `use_container_width=True` to `st.plotly_chart()`.
- **Columns:** Use `st.columns` for side-by-side benchmark panels (typically 3 for model
  comparison, 2 for representative doc sections).
- **Tabs:** Use `st.tabs` for analysis sections (score distribution, latency, recall, cost).
- **Primary buttons:** All action buttons that trigger writes or API calls use `type="primary"`.

---

## MongoDB and Vector Search Conventions

- **Aggregation pipelines:** Build `$vectorSearch` pipelines in `core/metrics.py::run_vector_search()`.
  Do not construct pipeline dicts in page files.
- **Pre-filters:** Pass Atlas pre-filters as a single `{"$and": [...]}` dict. Always validate
  that the filter fields (`property_type`, `room_type`, `bedrooms`) are included in the index
  filter definition before adding new filter fields.
- **Score projection:** Always project `score: {"$meta": "vectorSearchScore"}` on vector
  search results.
- **BSON types:** When displaying raw documents in the UI, clean BSON types before passing
  to `st.json()`: convert `ObjectId` → `str`, `Decimal128` → `float`, `datetime` → ISO string.
- **Batch writes:** Use `UpdateOne` operations with `bulk_write()` for embedding updates.
  Batch size: 500 for document copies, 100 for embedding inserts.
- **Connection:** Use the `@st.cache_resource`-cached `get_client()` from `core/db.py`.
  Never instantiate `MongoClient` directly in a page or utility file.

---

## Quality Gates

A task is complete only when all of the following pass:

- `uv run ruff format --check .` — no formatting changes needed.
- `uv run ruff check .` — no linting errors.
- `uv run mypy core/ utils/ app.py` — no type errors (strict mode is not enforced, but
  obvious annotation gaps must be fixed).
- Manual smoke test: `uv run streamlit run app.py` starts without import errors and the
  home page renders the connection status widget.

There are currently no automated tests in this project. If adding tests, place them in
`tests/` and add `pythonpath = ["."]` to a `[tool.pytest.ini_options]` section in
`pyproject.toml`.

---

## Constraints

- **Never commit `.env`** or any file containing credentials. The `.gitignore` already covers
  this — do not remove those rules.
- **Ask before adding an external dependency.** Every new package increases the attack surface
  and install time.
- **Preserve existing function signatures** in `core/` modules. Pages depend on them directly;
  a signature change breaks multiple call sites.
- **Do not modify `Embedding_Benchmark.py`** — it is a legacy alias kept for compatibility.
- **Do not restructure pages** from the `pages/` directory convention to the newer
  `st.navigation()` API without explicit instruction.
- **Do not embed credentials** in any Python file, config file, or Streamlit secrets file
  checked into version control.
- **Do not run the Voyage AI embedding API** (batch or query) without user confirmation when
  the operation will incur cost — follow the pattern in `pages/1_Setup.py` (checkbox gate).
- **Binary index distance must be euclidean.** Atlas requires this for 1-bit quantized
  indexes. Do not change `vs_binary` to cosine.
- Keep existing docstrings and type hints intact when modifying functions. Update them if the
  behaviour changes.

---

## Ignore

Treat everything in `.gitignore` as off-limits to read or edit. Additionally, never open:

- `.env` — live credentials
- `uv.lock` — machine-generated lockfile; update only via `uv add` / `uv sync`
- `.venv/` — managed virtual environment
- `__pycache__/` — bytecode cache
- Any file inside `.agents/` or `.claude/` skill trees (symlinks into installed packages)
