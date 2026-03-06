"""
search/ingest.py
----------------
Reads a result.json file, embeds all items in batches, and upserts them into
the construction.items_specification PostgreSQL table.

Usage:
    python search/ingest.py output/1772212680/result.json
"""

import json
import os
import sys

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
BATCH_SIZE = 10


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
# Embedding helper
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, returning a list of vectors."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        # response.data is ordered to match input
        vectors.extend([item.embedding for item in response.data])
    return vectors

def build_search_text(project: dict, item: dict) -> str:
    """
    Build the concatenated string that will be embedded for a single item.
    Project context is prepended so queries like "keramik SPPG" resolve correctly.
    """
    nama_proyek   = project.get("nama_proyek") or ""
    nama_bangunan = project.get("nama_bangunan") or ""
    _jenis        = project.get("jenis_pekerjaan") or ""
    jenis         = ", ".join(_jenis) if isinstance(_jenis, list) else _jenis
    lokasi        = project.get("lokasi") or ""

    h1      = item.get("header1") or ""
    h2      = item.get("header2") or ""
    nama    = item.get("nama_barang") or ""

    project_ctx = ", ".join(p for p in [nama_proyek, nama_bangunan, jenis, lokasi] if p)
    item_ctx    = " | ".join(p for p in [f"{h1} > {h2}".strip(" >"), nama] if p)

    return " || ".join(p for p in [project_ctx, item_ctx] if p)

# ---------------------------------------------------------------------------
# Main ingest function
# ---------------------------------------------------------------------------

def ingest(result_json_path: str) -> None:
    # Normalise path separators so the idempotency key is consistent
    result_json_path = result_json_path.replace("\\", "/")

    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # ------------------------------------------------------------------
            # Idempotency check
            # ------------------------------------------------------------------
            cur.execute(
                "SELECT COUNT(*) FROM construction.items_specification WHERE result_json_path = %s",
                (result_json_path,),
            )
            (count,) = cur.fetchone()
            if count > 0:
                print(
                    f"WARNING: {result_json_path} has already been ingested "
                    f"({count} rows). Skipping. "
                    "Delete existing rows first if you want to re-ingest."
                )
                return

            # ------------------------------------------------------------------
            # Extract project-level fields
            # ------------------------------------------------------------------
            project = {
                "nama_proyek":     data.get("nama_proyek"),
                "nama_site":       data.get("nama_site"),
                "lokasi":          data.get("lokasi"),
                "nama_bangunan":   data.get("nama_bangunan"),
                "jenis_pekerjaan": data.get("jenis_pekerjaan"),
            }

            items = data.get("items", [])
            if not items:
                print("No items found in result.json — nothing to ingest.")
                return

            # ------------------------------------------------------------------
            # Build search_text for every item
            # ------------------------------------------------------------------
            search_texts = [build_search_text(project, item) for item in items]

            # ------------------------------------------------------------------
            # Embed all texts in batches of BATCH_SIZE
            # ------------------------------------------------------------------
            print(f"Embedding {len(search_texts)} items in batches of {BATCH_SIZE}…")
            vectors = embed_texts(search_texts)

            # ------------------------------------------------------------------
            # Insert rows
            # ------------------------------------------------------------------
            insert_sql = """
                INSERT INTO construction.items_specification (
                    result_json_path,
                    nama_proyek, nama_site, lokasi, nama_bangunan, jenis_pekerjaan,
                    header1, header2,
                    nama_barang,
                    spesifikasi_general, spesifikasi_merek,
                    search_text, embedding
                ) VALUES (
                    %s,
                    %s, %s, %s, %s, %s,
                    %s, %s,
                    %s,
                    %s, %s,
                    %s, %s::vector
                )
            """

            rows = []
            for item, search_text, vector in zip(items, search_texts, vectors):
                spec = item.get("spesifikasi_barang") or {}
                rows.append((
                    result_json_path,
                    project["nama_proyek"],
                    project["nama_site"],
                    project["lokasi"],
                    project["nama_bangunan"],
                    project["jenis_pekerjaan"],
                    item.get("header1"),
                    item.get("header2"),
                    item.get("nama_barang"),
                    spec.get("spesifikasi_general"),
                    spec.get("spesifikasi_merek"),
                    search_text,
                    str(vector),
                ))

            cur.executemany(insert_sql, rows)
            conn.commit()

        print(f"Ingested {len(rows)} items for project: {project['nama_proyek']}")

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search/ingest.py <path/to/result.json>")
        sys.exit(1)

    ingest(sys.argv[1])
