# PRD: Vector Search for Construction Item Specifications

## Overview

Add semantic search capability to the construction specification extraction pipeline using **Qwen Embedding** via the existing DashScope API. Users should be able to query items like `"lantai kamar mandi keramik"` or `"Toto toilet Gedung SPPG"` and get ranked results across all extracted items from any processed PDF.

Project metadata (project name, building name, etc.) is **denormalized into every item row** and included in the embedding — there is no separate projects table.

---

## Repository Context

```
hamilton-ai-construction/
├── requirements.txt                          # current deps (openai, dotenv, PyMuPDF, Pillow)
├── schema/
│   └── database.sql                          # EMPTY — you will populate this
├── items_specification/
│   ├── items_specification_extraction.py     # main orchestrator — READ THIS
│   ├── stage2_page_extraction.py             # per-page LLM extraction — READ THIS
│   ├── merge_results.py                      # merges per-page JSONs into result.json — READ THIS
│   └── pdf_to_images.py                      # renders PDF pages to base64
└── output/
    └── 1772212680/
        └── result.json                       # SAMPLE OUTPUT — READ THIS for data shape
```

**Read all files listed above before starting implementation.**

---

## Existing Data Shape

`result.json` (produced by the extraction pipeline) looks like:

```json
{
  "nama_proyek": "Pembangunan Gedung SPPG 1 TA 2025",
  "nama_site": null,
  "lokasi": null,
  "nama_bangunan": "Gedung SPPG 1",
  "jenis_pekerjaan": "Gedung SPPG",
  "items": [
    {
      "header1": "Arsitektur",
      "header2": "A. Lantai",
      "nama_barang": "R. Konsultasi Ahli Gizi",
      "spesifikasi_barang": {
        "spesifikasi_general": "HT 60x60, Polished/Unpolished",
        "spesifikasi_merek": "Nirogranite, Granito, Romangranit"
      }
    }
  ]
}
```

> **Note:** The `header1` and `header2` fields were added in the latest version of `stage2_page_extraction.py`. Older `result.json` files in `output/` may not have them — the implementation must handle `null` gracefully.

---

## API / Credentials

- **Embedding model:** `text-embedding-v3` (Qwen, via DashScope)
- **Base URL:** `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- **API Key env var:** `DASHSCOPE_API_KEY` (loaded via `python-dotenv` from a `.env` file)
- **Client:** use the existing `openai` Python package (already installed), pointing to the DashScope base URL — same pattern already used in `stage2_page_extraction.py`

Embedding call example (follow existing client pattern):

```python
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
response = client.embeddings.create(
    model="text-embedding-v3",
    input=["your text here"],
)
vector = response.data[0].embedding
```

---

## Implementation Steps

### Step 1 — Define the database schema (`schema/database.sql`)

Create a **single table** called `items_specification`. Project metadata is denormalized into every row — no separate projects table.

**`items_specification` table columns:**

| Column                | Type                        | Notes                                                                 |
| --------------------- | --------------------------- | --------------------------------------------------------------------- |
| `id`                  | `SERIAL PRIMARY KEY`        |                                                                       |
| `result_json_path`    | `TEXT NOT NULL`             | Path to the source `result.json`; used to detect duplicate ingestions |
| `nama_proyek`         | `TEXT`                      | nullable                                                              |
| `nama_site`           | `TEXT`                      | nullable                                                              |
| `lokasi`              | `TEXT`                      | nullable                                                              |
| `nama_bangunan`       | `TEXT`                      | nullable                                                              |
| `jenis_pekerjaan`     | `TEXT`                      | nullable                                                              |
| `header1`             | `TEXT`                      | nullable                                                              |
| `header2`             | `TEXT`                      | nullable                                                              |
| `nama_barang`         | `TEXT NOT NULL`             |                                                                       |
| `spesifikasi_general` | `TEXT`                      | nullable                                                              |
| `spesifikasi_merek`   | `TEXT`                      | nullable                                                              |
| `search_text`         | `TEXT NOT NULL`             | The concatenated string that was embedded (see Step 3)                |
| `embedding`           | `vector(1024)`              | Requires `pgvector` extension; 1024 dims for `text-embedding-v3`      |
| `created_at`          | `TIMESTAMPTZ DEFAULT now()` |                                                                       |

Add index: `CREATE INDEX ON items_specification USING ivfflat (embedding vector_cosine_ops);`

Enable extension at top of file: `CREATE EXTENSION IF NOT EXISTS vector;`

---

### Step 2 — Add `pgvector` and `psycopg2` dependencies

Add to `requirements.txt`:

```
pgvector
psycopg2-binary
```

---

### Step 3 — Build the `search_text` field

Project details are prepended to every item's embedding string so that queries like `"keramik SPPG"` or `"lantai Gedung SPPG 1"` resolve correctly.

```python
def build_search_text(project: dict, item: dict) -> str:
    nama_proyek   = project.get("nama_proyek") or ""
    nama_bangunan = project.get("nama_bangunan") or ""
    jenis         = project.get("jenis_pekerjaan") or ""
    lokasi        = project.get("lokasi") or ""
    h1    = item.get("header1") or ""
    h2    = item.get("header2") or ""
    nama  = item.get("nama_barang") or ""
    general = (item.get("spesifikasi_barang") or {}).get("spesifikasi_general") or ""
    merek   = (item.get("spesifikasi_barang") or {}).get("spesifikasi_merek") or ""
    project_ctx = ", ".join(p for p in [nama_proyek, nama_bangunan, jenis, lokasi] if p)
    item_ctx    = " | ".join(p for p in [f"{h1} > {h2}".strip(" >"), nama, general, merek] if p)
    return " || ".join(p for p in [project_ctx, item_ctx] if p)
```

**Example output:**

```
Pembangunan Gedung SPPG 1 TA 2025, Gedung SPPG 1, Gedung SPPG || Arsitektur > A. Lantai | R. Konsultasi Ahli Gizi | HT 60x60, Polished/Unpolished | Nirogranite, Granito, Romangranit
```

---

### Step 4 — Create `search/ingest.py`

This script reads a `result.json` file, embeds all items in batches, and upserts them into PostgreSQL.

**Location:** `search/ingest.py`

**CLI usage:**

```
python search/ingest.py output/1772212680/result.json
```

**Behaviour:**

1. Connect to PostgreSQL using env vars (`POSTGRES_DSN` or individual `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`).
2. **Idempotency check:** query `SELECT COUNT(*) FROM items_specification WHERE result_json_path = ?`. If count > 0, print a warning and exit early — do not re-ingest the same file.
3. Extract project-level fields from the top of `result.json` (`nama_proyek`, `nama_site`, `lokasi`, `nama_bangunan`, `jenis_pekerjaan`).
4. For each item in `result["items"]`:
   - Build `search_text` using the function from Step 3, passing both the project dict and the item dict.
5. Collect **all** `search_text` values into a list and call the embedding API **in batches of 25** (the model's max per call).
6. Insert one row per item into `items_specification`, denormalizing the project fields onto every row.
7. Print a summary: `Ingested {n} items for project: {nama_proyek}`.

---

### Step 5 — Create `search/search.py`

A reusable search module (not CLI) that exposes one function:

```python
def search_items(
    query: str,
    nama_proyek: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    ...
```

**Behaviour:**

1. Embed the query string using `text-embedding-v3` (single input).
2. Run a cosine similarity search against the `items_specification` table using `pgvector`'s `<=>` operator.
3. If `nama_proyek` is provided, filter `WHERE nama_proyek ILIKE ?` (partial match, case-insensitive).
4. Return the top `top_k` results as a list of dicts with keys:
   - `score` (float, cosine similarity, 1 = perfect match)
   - `nama_proyek`, `nama_bangunan`, `lokasi`
   - `header1`, `header2`
   - `nama_barang`
   - `spesifikasi_general`
   - `spesifikasi_merek`
   - `search_text`

SQL template:

```sql
SELECT
    1 - (embedding <=> %s::vector) AS score,
    nama_proyek, nama_bangunan, lokasi,
    header1, header2, nama_barang,
    spesifikasi_general, spesifikasi_merek, search_text
FROM items_specification
WHERE (%s IS NULL OR nama_proyek ILIKE %s)
ORDER BY embedding <=> %s::vector
LIMIT %s;
```

---

### Step 6 — Create `search/cli.py` (CLI demo)

**Location:** `search/cli.py`

**CLI usage:**

```
python search/cli.py "keramik lantai 60x60"
python search/cli.py "Toto toilet" --nama-proyek "SPPG" --top-k 5
```

**Arguments:**

- Positional: `query` (required)
- `--nama-proyek` (optional str): restrict search to projects whose name contains this substring (case-insensitive)
- `--top-k` (optional int, default 10): number of results to return

**Output format** (printed to stdout):

```
Query: "keramik lantai 60x60"
──────────────────────────────────────────────────────────────────────────────
 #  Score   Project                       Header                   Item
──────────────────────────────────────────────────────────────────────────────
 1  0.9421  Gedung SPPG 1                 Arsitektur > A. Lantai   Toilet Wanita
            Spec : HT 60x60, Unpolished
            Merek: Nirogranite, Granito, Romangranit

 2  0.9187  Gedung SPPG 1                 Arsitektur > A. Lantai   R. Konsultasi Ahli Gizi
            Spec : HT 60x60, Polished/Unpolished
            Merek: Nirogranite, Granito, Romangranit
...
```

Use Python's built-in `argparse` — no extra CLI framework needed.

---

### Step 7 — Update `requirements.txt`

Final `requirements.txt` should include:

```
openai==2.14.0
python-dotenv
PyMuPDF
Pillow
pgvector
psycopg2-binary
```

---

## File Structure After Implementation

```
hamilton-ai-construction/
├── requirements.txt          # updated
├── schema/
│   └── database.sql          # populated (Step 1)
├── items_specification/
│   └── ...                   # unchanged
├── search/
│   ├── __init__.py           # empty
│   ├── ingest.py             # Step 4
│   ├── search.py             # Step 5
│   └── cli.py                # Step 6
└── output/
    └── ...
```

---

## Environment Variables Required

| Variable                                                    | Purpose                                                                    |
| ----------------------------------------------------------- | -------------------------------------------------------------------------- |
| `DASHSCOPE_API_KEY`                                         | Qwen embedding + LLM calls                                                 |
| `POSTGRES_DSN`                                              | Full DSN e.g. `postgresql://user:pass@localhost:5432/hamilton` (preferred) |
| or `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD` | Alternative individual vars                                                |

All loaded via `python-dotenv` from a `.env` file in the project root.

---

## Notes on Denormalization

Because project fields are stored on every row:

- Searching `"Gedung SPPG keramik"` will surface items even if `header1`/`header2` don't mention the project — the embedding captures it.
- Filtering by project is done via `WHERE nama_proyek ILIKE ?` rather than a JOIN.
- Re-ingesting the same `result.json` is blocked by the idempotency check on `result_json_path`. To replace data for a file, manually `DELETE FROM items_specification WHERE result_json_path = '...'` first.

---

## Out of Scope

- No web API / HTTP server (pure CLI for now)
- No authentication
- No re-embedding on update (re-run ingest with a new `result_json_path`)
- No frontend UI
