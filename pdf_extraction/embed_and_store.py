"""
Embedding & Storing Script
Focus: Embed catalog JSON using Local HuggingFace Model and store to PostgreSQL
"""

import os
import json
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

# ==================== CONFIGURATION ====================
DB_URI = "postgresql://postgres:hamiltonserver3.14@10.5.0.4:5432/postgres"

TABLE_NAME = "catalog_items_hf"

print("Loading local AI model (all-MiniLM-L6-v2). This might take a minute on first run to download...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    """Function to convert text into a vector using local HuggingFace model"""
    try:
        vec = model.encode(text)
        return vec.tolist()
    except Exception as e:
        print(f"Failed to fetch embedding: {e}")
        return None

def main():
    print("=" * 60)
    print("CATALOG EMBEDDING & STORING (LOCAL AI MODE)")
    print("=" * 60)
    
    json_path = "data_output/catalog_for_embedding.json"
    
    if not os.path.exists(json_path):
        json_path = "catalog_for_embedding.json"

    if not os.path.exists(json_path):
        print(f"Error: JSON file not found!")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        items = json.load(f)

    print(f"Successfully loaded {len(items)} items from JSON.")

    print("Connecting to Hamilton database...")
    try:
        with psycopg.connect(DB_URI) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            register_vector(conn)

            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                    id SERIAL PRIMARY KEY,
                    nama_barang TEXT,
                    jenis_tipe_barang TEXT,
                    spesifikasi JSONB,
                    merk_barang TEXT,
                    harga_barang NUMERIC,
                    embedding vector(384) 
                );
            """)
            print(f"Table '{TABLE_NAME}' is ready!")

            print("Starting local embedding process. Super fast, no API limits!")
            success_count = 0
            
            for i, item in enumerate(items):
                nama = item.get('nama_barang', '')
                jenis = item.get('jenis_tipe_barang', '') or 'None'
                spesifikasi = item.get('spesifikasi_detail_barang', {})
                merk = item.get('merk_barang', '') or 'None'
                harga = item.get('harga_barang')

                text_to_embed = f"Nama Barang: {nama}. Jenis: {jenis}. Spesifikasi: {json.dumps(spesifikasi)}. Merk: {merk}."

                vec = get_embedding(text_to_embed)

                if vec:
                    conn.execute(f"""
                        INSERT INTO {TABLE_NAME} (nama_barang, jenis_tipe_barang, spesifikasi, merk_barang, harga_barang, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (nama, jenis, json.dumps(spesifikasi), merk, harga, vec))
                    success_count += 1

                if (i + 1) % 50 == 0:
                    print(f"Progress: {i + 1} / {len(items)} items successfully embedded...")

            conn.commit()
            
            print("=" * 60)
            print(f"[SUCCESS] {success_count} out of {len(items)} items successfully embedded & stored to database!")
            print(f"Target Table: {TABLE_NAME}")
            print("=" * 60)

    except Exception as e:
        print(f"Database connection error: {e}")

if __name__ == "__main__":
    main()