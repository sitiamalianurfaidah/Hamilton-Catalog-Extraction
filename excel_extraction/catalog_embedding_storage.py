"""
Catalog Embedding & Storage to PostgreSQL
Menggunakan model lokal all-MiniLM-L6-v2 untuk embedding
Menyimpan ke tabel catalog_items_hf di schema construction
"""

import os
import json
import argparse
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np

# Database configuration
DB_CONFIG = {
    "host": "10.5.0.4",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "hamiltonserver3.14"
}
DB_URI = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"

TABLE_NAME = "catalog_items_hf"
SCHEMA_NAME = "construction"


class CatalogEmbeddingStorage:
    """Kelas untuk embedding dan menyimpan katalog ke PostgreSQL"""
    
    def __init__(self):
        """Initialize model embedding"""
        print("Loading local AI model (all-MiniLM-L6-v2)...")
        print("This might take a minute on first run to download.")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Model loaded successfully")
        
        # Dimensi embedding untuk model ini
        self.embedding_dim = 384
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Convert text to vector using local HuggingFace model
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing embedding vector
        """
        try:
            vec = self.model.encode(text)
            return vec.tolist()
        except Exception as e:
            print(f"Failed to fetch embedding: {e}")
            return None
    
    def prepare_text_for_embedding(self, item: Dict[str, Any]) -> str:
        """
        Prepare text from item for embedding
        
        Args:
            item: Dictionary with product details
            
        Returns:
            Combined text for embedding
        """
        nama = item.get('nama_barang', '')
        jenis = item.get('jenis_tipe_barang', '') or 'None'
        
        # Handle spesifikasi (bisa string atau dict)
        spesifikasi = item.get('spesifikasi_detail_barang', item.get('spesifikasi', {}))
        if isinstance(spesifikasi, dict):
            spesifikasi_text = json.dumps(spesifikasi, ensure_ascii=False)
        else:
            spesifikasi_text = str(spesifikasi)
        
        merk = item.get('merk_barang', '') or 'None'
        
        # Tambahkan raw_keterangan jika ada
        keterangan = item.get('raw_keterangan', '')
        
        # Gabungkan semua text
        text_parts = [
            f"Nama Barang: {nama}",
            f"Jenis/Tipe: {jenis}",
            f"Spesifikasi: {spesifikasi_text}",
            f"Merk: {merk}"
        ]
        
        if keterangan:
            text_parts.append(f"Keterangan: {keterangan}")
        
        return ". ".join(text_parts)
    
    def create_table(self, conn):
        """
        Create table if not exists
        
        Args:
            conn: PostgreSQL connection
        """
        # Set schema
        conn.execute(f"SET search_path TO {SCHEMA_NAME};")
        
        # Create extension if not exists
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        
        # Create table
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{TABLE_NAME} (
                id SERIAL PRIMARY KEY,
                nama_barang TEXT,
                jenis_tipe_barang TEXT,
                spesifikasi JSONB,
                merk_barang TEXT,
                harga_barang NUMERIC,
                tanggal_transaksi TEXT,
                raw_keterangan TEXT,
                embedding vector({self.embedding_dim}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index for faster similarity search
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_embedding 
            ON {SCHEMA_NAME}.{TABLE_NAME} 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        print(f"✓ Table '{SCHEMA_NAME}.{TABLE_NAME}' is ready!")
    
    def store_item(self, conn, item: Dict[str, Any], embedding: List[float]) -> bool:
        """
        Store single item to database
        
        Args:
            conn: PostgreSQL connection
            item: Item dictionary
            embedding: Embedding vector
            
        Returns:
            True if successful
        """
        try:
            conn.execute(f"""
                INSERT INTO {SCHEMA_NAME}.{TABLE_NAME} 
                (nama_barang, jenis_tipe_barang, spesifikasi, merk_barang, harga_barang, 
                 tanggal_transaksi, raw_keterangan, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                item.get('nama_barang', ''),
                item.get('jenis_tipe_barang', ''),
                json.dumps(item.get('spesifikasi_detail_barang', item.get('spesifikasi', {}))),
                item.get('merk_barang', ''),
                item.get('harga_barang', 0),
                item.get('tanggal', ''),
                item.get('raw_keterangan', ''),
                embedding
            ))
            return True
        except Exception as e:
            print(f"  ✗ Error storing item: {e}")
            return False
    
    def process_and_store(self, json_path: str) -> tuple:
        """
        Main method: Load JSON, create embeddings, store to PostgreSQL
        
        Args:
            json_path: Path to JSON file with items
            
        Returns:
            Tuple of (success_count, total_count)
        """
        print("\n" + "="*60)
        print("CATALOG EMBEDDING & STORING (LOCAL AI MODE)")
        print("="*60)
        
        # Load JSON
        if not os.path.exists(json_path):
            print(f"Error: JSON file not found: {json_path}")
            return 0, 0
        
        with open(json_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
        
        print(f"✓ Successfully loaded {len(items)} items from JSON.")
        
        # Connect to database
        print("Connecting to Hamilton database...")
        success_count = 0
        
        try:
            with psycopg.connect(DB_URI) as conn:
                # Create table if not exists
                self.create_table(conn)
                
                print("Starting local embedding process. Super fast, no API limits!")
                
                for i, item in enumerate(items):
                    # Prepare text for embedding
                    text_to_embed = self.prepare_text_for_embedding(item)
                    
                    # Get embedding
                    vec = self.get_embedding(text_to_embed)
                    
                    if vec:
                        if self.store_item(conn, item, vec):
                            success_count += 1
                    
                    # Progress indicator
                    if (i + 1) % 50 == 0:
                        print(f"Progress: {i + 1} / {len(items)} items successfully embedded...")
                
                conn.commit()
                
                print("="*60)
                print(f"[SUCCESS] {success_count} out of {len(items)} items successfully embedded & stored to database!")
                print(f"Target Table: {SCHEMA_NAME}.{TABLE_NAME}")
                print("="*60)
                
                return success_count, len(items)
                
        except Exception as e:
            print(f"✗ Database error: {e}")
            return success_count, len(items)
    
    def clear_table(self):
        """Clear all data from table (for testing)"""
        try:
            with psycopg.connect(DB_URI) as conn:
                conn.execute(f"SET search_path TO {SCHEMA_NAME};")
                conn.execute(f"TRUNCATE TABLE {TABLE_NAME} RESTART IDENTITY;")
                conn.commit()
                print(f"✓ Table '{SCHEMA_NAME}.{TABLE_NAME}' cleared")
        except Exception as e:
            print(f"✗ Error clearing table: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Catalog Embedding & Storage to PostgreSQL"
    )
    parser.add_argument(
        "json_file",
        help="Path to JSON file containing items for embedding"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear table before inserting"
    )
    
    args = parser.parse_args()
    
    # Initialize
    storage = CatalogEmbeddingStorage()
    
    # Clear if requested
    if args.clear:
        storage.clear_table()
    
    # Process and store
    success, total = storage.process_and_store(args.json_file)
    
    if success == total:
        print("\n✓ Semua data berhasil disimpan!")
        return 0
    else:
        print(f"\n⚠ Hanya {success}/{total} yang berhasil disimpan")
        return 1


if __name__ == "__main__":
    exit(main())