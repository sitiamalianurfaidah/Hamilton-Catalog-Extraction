"""
search/search.py
----------------
Reusable search module. Exposes search_items() for hybrid (semantic + keyword)
similarity search against the construction.items_specification table.

Scoring
-------
For a multi-word query each individual word is checked against search_text
via case-insensitive substring match.  The keyword score is the fraction of
query words that hit (0–1).  The final score is a weighted combination:

    score = semantic_weight * semantic_score + keyword_weight * keyword_score

Default weights: 70 % semantic, 30 % keyword.
"""

import os
import re

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

embedding_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

EMBEDDING_MODEL = "text-embedding-v4"

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "hamilton"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )


# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------

def _extract_words(query: str, min_len: int = 2) -> list[str]:
    """
    Split *query* into lower-cased tokens, discarding tokens shorter than
    *min_len* characters (removes single-letter noise and punctuation).
    """
    return [w for w in re.findall(r"\b\w+\b", query.lower()) if len(w) >= min_len]


def _keyword_score_sql(words: list[str]) -> tuple[str, list]:
    """
    Return a SQL fragment that evaluates to a float in [0, 1] representing
    the fraction of *words* found (case-insensitively) in column search_text,
    together with the corresponding positional parameter values.

    If *words* is empty the expression always yields 0.0 with no params.
    """
    if not words:
        return "0.0::float", []

    cases = " + ".join(
        "(CASE WHEN search_text ILIKE %s THEN 1 ELSE 0 END)" for _ in words
    )
    expr = f"({cases})::float / {len(words)}"
    params = [f"%{w}%" for w in words]
    return expr, params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_items(
    query: str,
    nama_proyek: str | None = None,
    top_k: int = 10,
    keyword_weight: float = 0.3,
) -> list[dict]:
    """
    Hybrid (semantic + keyword) search over construction.items_specification.

    Args:
        query:           Natural-language search string (will be embedded).
        nama_proyek:     Optional project name filter (case-insensitive partial).
        top_k:           Number of results to return.
        keyword_weight:  Weight given to the keyword score (0–1).
                         The semantic score receives (1 - keyword_weight).

    Returns:
        List of dicts ordered by descending combined score, each containing:
        score, semantic_score, keyword_score,
        nama_proyek, nama_bangunan, lokasi, header1, header2,
        nama_barang, spesifikasi_general, spesifikasi_merek, search_text.
    """
    keyword_weight = max(0.0, min(1.0, keyword_weight))
    semantic_weight = 1.0 - keyword_weight

    # --- semantic embedding -------------------------------------------
    response = embedding_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    query_vector = response.data[0].embedding

    # --- keyword score fragment ---------------------------------------
    words = _extract_words(query)
    kw_expr, kw_params = _keyword_score_sql(words)

    # --- combined query via CTE so keyword params appear only once ----
    sql = f"""
        WITH scored AS (
            SELECT
                1 - (embedding <=> %s::vector)  AS semantic_score,
                ({kw_expr})                      AS keyword_score,
                nama_proyek, nama_bangunan, lokasi,
                header1, header2, nama_barang,
                spesifikasi_general, spesifikasi_merek, search_text
            FROM construction.items_specification
            WHERE (%s IS NULL OR nama_proyek ILIKE %s)
        )
        SELECT
            semantic_score * {semantic_weight} + keyword_score * {keyword_weight} AS score,
            semantic_score,
            keyword_score,
            nama_proyek, nama_bangunan, lokasi,
            header1, header2, nama_barang,
            spesifikasi_general, spesifikasi_merek, search_text
        FROM scored
        ORDER BY score DESC
        LIMIT %s;
    """

    like_pattern = f"%{nama_proyek}%" if nama_proyek else None

    params = (
        str(query_vector),   # embedding <=> %s::vector
        *kw_params,          # one %s per word in keyword score
        nama_proyek,         # IS NULL check
        like_pattern,        # ILIKE pattern
        top_k,
    )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]
    finally:
        conn.close()
