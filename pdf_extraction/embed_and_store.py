"""
pdf_extraction/embed_and_store.py
----------------------------------
Reads a catalog_for_embedding.json file, embeds all items in batches, and
upserts them into the construction.items_catalog PostgreSQL table.

Usage:
    python pdf_extraction/embed_and_store.py data_output/catalog_for_embedding.json
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
    """Embed a list of strings in batches, returning a list of vectors."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        vectors.extend([item.embedding for item in response.data])
    return vectors


def build_search_text(item: dict) -> str:
    """Build the concatenated string that will be embedded for a single catalog item."""
    nama  = item.get("nama_barang") or ""
    jenis = item.get("jenis_tipe_barang") or ""
    merk  = item.get("merk_barang") or ""
    spec  = item.get("spesifikasi_detail_barang") or {}

    parts = [p for p in [nama, jenis, merk] if p]
    if spec:
        parts.append(json.dumps(spec, ensure_ascii=False))

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main ingest function
# ---------------------------------------------------------------------------

def ingest(json_path: str) -> None:
    # Normalise path separators so the idempotency key is consistent
    json_path = json_path.replace("\\", "/")

    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    print(f"Loaded {len(items)} items from {json_path}.")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # ------------------------------------------------------------------
            # Idempotency check
            # ------------------------------------------------------------------
            cur.execute(
                "SELECT COUNT(*) FROM construction.items_catalog WHERE source_json_path = %s",
                (json_path,),
            )
            (count,) = cur.fetchone()
            if count > 0:
                print(
                    f"WARNING: {json_path} has already been ingested "
                    f"({count} rows). Skipping. "
                    "Delete existing rows first if you want to re-ingest."
                )
                return

            # ------------------------------------------------------------------
            # Build search_text for every item
            # ------------------------------------------------------------------
            search_texts = [build_search_text(item) for item in items]

            # ------------------------------------------------------------------
            # Embed all texts in batches of BATCH_SIZE
            # ------------------------------------------------------------------
            print(f"Embedding {len(search_texts)} items in batches of {BATCH_SIZE}…")
            vectors = embed_texts(search_texts)

            # ------------------------------------------------------------------
            # Insert rows
            # ------------------------------------------------------------------
            insert_sql = """
                INSERT INTO construction.items_catalog (
                    source_json_path,
                    nama_barang, jenis_tipe_barang, spesifikasi,
                    merk_barang, harga_barang,
                    search_text, embedding
                ) VALUES (
                    %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s::vector
                )
            """

            rows = []
            for item, search_text, vector in zip(items, search_texts, vectors):
                rows.append((
                    json_path,
                    item.get("nama_barang"),
                    item.get("jenis_tipe_barang"),
                    json.dumps(item.get("spesifikasi_detail_barang") or {}),
                    item.get("merk_barang"),
                    item.get("harga_barang"),
                    search_text,
                    str(vector),
                ))

            cur.executemany(insert_sql, rows)
            conn.commit()

        print(f"Ingested {len(rows)} catalog items from: {json_path}")

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_extraction/embed_and_store.py <path/to/catalog_for_embedding.json>")
        sys.exit(1)

    ingest(sys.argv[1])