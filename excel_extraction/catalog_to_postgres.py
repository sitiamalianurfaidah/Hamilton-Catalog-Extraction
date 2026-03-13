#!/usr/bin/env python3
"""
Catalog JSON to PostgreSQL - Upload hasil ekstraksi ke database
"""

import os
import json
import argparse
import psycopg
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

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

class CatalogJsonToPostgres:
    """Kelas untuk parsing JSON hasil ekstraksi ke PostgreSQL"""
    
    def __init__(self):
        """Initialize"""
        pass
    
    def create_schema_and_table(self, conn):
        """Create schema and table if they don't exist"""
        
        # Create schema if not exists
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME};")
        print(f"✓ Schema '{SCHEMA_NAME}' is ready")
        
        # Set search path
        conn.execute(f"SET search_path TO {SCHEMA_NAME}, public;")
        
        # Drop existing table if needed (optional - uncomment if you want to recreate)
        # conn.execute(f"DROP TABLE IF EXISTS {SCHEMA_NAME}.{TABLE_NAME} CASCADE;")
        
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
                source_file TEXT,
                page_number INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create indexes for better performance
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_nama_barang 
            ON {SCHEMA_NAME}.{TABLE_NAME} (nama_barang);
        """)
        
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_harga 
            ON {SCHEMA_NAME}.{TABLE_NAME} (harga_barang);
        """)
        
        print(f"✓ Table '{SCHEMA_NAME}.{TABLE_NAME}' is created/verified")
        
        # Show table structure
        cursor = conn.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = '{SCHEMA_NAME}' 
            AND table_name = '{TABLE_NAME}'
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        print(f"  Table columns: {', '.join([f'{col[0]} ({col[1]})' for col in columns])}")
    
    def parse_extraction_json(self, json_path: str) -> List[Dict[str, Any]]:
        """Parse extraction result JSON to list of items"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        items = []
        source_file = os.path.basename(json_path)
        
        # Handle the specific JSON format
        if "items" in data and isinstance(data["items"], list):
            print(f"  Found items array with {len(data['items'])} items")
            for item in data["items"]:
                # Handle spesifikasi properly
                spesifikasi = item.get("spesifikasi_detail_barang", {})
                if not isinstance(spesifikasi, dict):
                    spesifikasi = {}
                
                items.append({
                    "nama_barang": item.get("nama_barang", ""),
                    "jenis_tipe_barang": item.get("jenis_tipe_barang", ""),
                    "spesifikasi": spesifikasi,
                    "merk_barang": item.get("merk_barang", ""),
                    "harga_barang": float(item.get("harga_barang", 0)) if item.get("harga_barang") else 0,
                    "tanggal_transaksi": None,
                    "raw_keterangan": None,
                    "source_file": source_file,
                    "page_number": item.get("_page", None)
                })
        
        elif "blocks" in data:
            for block in data.get("blocks", []):
                tanggal = block.get("tanggal")
                for item in block.get("items", []):
                    spesifikasi = item.get("spesifikasi_detail_barang", item.get("spesifikasi", {}))
                    if not isinstance(spesifikasi, dict):
                        spesifikasi = {}
                    
                    items.append({
                        "nama_barang": item.get("nama_barang", ""),
                        "jenis_tipe_barang": item.get("jenis_tipe_barang", item.get("jenis_tipe", "")),
                        "spesifikasi": spesifikasi,
                        "merk_barang": item.get("merk_barang", ""),
                        "harga_barang": float(item.get("harga_barang", item.get("harga", 0))),
                        "tanggal_transaksi": tanggal,
                        "raw_keterangan": item.get("raw_keterangan", ""),
                        "source_file": source_file,
                        "page_number": item.get("_page", block.get("page_number"))
                    })
        
        elif isinstance(data, list):
            for item in data:
                spesifikasi = item.get("spesifikasi_detail_barang", item.get("spesifikasi", {}))
                if not isinstance(spesifikasi, dict):
                    spesifikasi = {}
                
                items.append({
                    "nama_barang": item.get("nama_barang", ""),
                    "jenis_tipe_barang": item.get("jenis_tipe_barang", item.get("jenis_tipe", "")),
                    "spesifikasi": spesifikasi,
                    "merk_barang": item.get("merk_barang", ""),
                    "harga_barang": float(item.get("harga_barang", item.get("harga", 0))),
                    "tanggal_transaksi": item.get("tanggal", item.get("tanggal_transaksi", "")),
                    "raw_keterangan": item.get("raw_keterangan", ""),
                    "source_file": source_file,
                    "page_number": item.get("_page", None)
                })
        
        return items
    
    def upload_to_postgres(self, json_path: str, clear_first: bool = False, 
                          batch_size: int = 100) -> int:
        """Upload JSON data to PostgreSQL"""
        print("\n" + "="*60)
        print("CATALOG JSON TO POSTGRESQL")
        print("="*60)
        
        # Parse JSON
        print(f"\n📂 Loading JSON: {json_path}")
        items = self.parse_extraction_json(json_path)
        print(f"✓ Found {len(items)} items")
        
        if not items:
            print("✗ No items to upload")
            return 0
        
        # Show sample
        print(f"\n📋 Sample item (first of {len(items)}):")
        sample = items[0]
        for key, value in sample.items():
            if value:  # Only show non-empty values
                print(f"  {key}: {value}")
        
        # Connect to database
        print("\n🔌 Connecting to PostgreSQL...")
        print(f"  Host: {DB_CONFIG['host']}")
        print(f"  Database: {DB_CONFIG['dbname']}")
        print(f"  Schema: {SCHEMA_NAME}")
        print(f"  Table: {TABLE_NAME}")
        
        success_count = 0
        
        try:
            with psycopg.connect(DB_URI) as conn:
                # Create schema and table
                self.create_schema_and_table(conn)
                
                # Clear if requested
                if clear_first:
                    conn.execute(f"TRUNCATE TABLE {SCHEMA_NAME}.{TABLE_NAME} RESTART IDENTITY CASCADE;")
                    print("✓ Table cleared")
                
                # Check existing data
                cursor = conn.execute(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{TABLE_NAME}")
                existing_count = cursor.fetchone()[0]
                print(f"📊 Existing records in table: {existing_count}")
                
                # Process in batches
                print(f"\n📤 Uploading {len(items)} items to database...")
                
                for i, item in enumerate(items):
                    try:
                        # Convert spesifikasi to JSON string if it's a dict
                        spesifikasi_json = json.dumps(item["spesifikasi"]) if isinstance(item["spesifikasi"], dict) else "{}"
                        
                        conn.execute(f"""
                            INSERT INTO {SCHEMA_NAME}.{TABLE_NAME} 
                            (nama_barang, jenis_tipe_barang, spesifikasi, merk_barang, 
                             harga_barang, tanggal_transaksi, raw_keterangan, source_file, page_number)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            item["nama_barang"],
                            item["jenis_tipe_barang"],
                            spesifikasi_json,
                            item["merk_barang"],
                            item["harga_barang"],
                            item["tanggal_transaksi"],
                            item["raw_keterangan"],
                            item["source_file"],

                        ))
                        success_count += 1
                    except Exception as e:
                        print(f"  ✗ Error inserting item {i+1}: {e}")
                        print(f"     Item: {item['nama_barang']}")
                    
                    # Progress
                    if (i + 1) % batch_size == 0:
                        conn.commit()
                        print(f"  Progress: {i + 1}/{len(items)} items uploaded...")
                
                # Final commit
                conn.commit()
                
                print(f"\nSuccessfully uploaded {success_count}/{len(items)} items")
                
                # Verify and show statistics
                cursor = conn.execute(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{TABLE_NAME}")
                total = cursor.fetchone()[0]
                print(f"Total items in database now: {total}")
                
                # Show some stats
                cursor = conn.execute(f"""
                    SELECT 
                        COUNT(DISTINCT source_file) as file_count,
                        COUNT(*) as total_items,
                        MIN(harga_barang) as min_price,
                        MAX(harga_barang) as max_price,
                        AVG(harga_barang)::numeric(10,2) as avg_price
                    FROM {SCHEMA_NAME}.{TABLE_NAME}
                """)
                stats = cursor.fetchone()
                if stats:
                    print(f"\nStatistics:")
                    print(f"  Files processed: {stats[0]}")
                    print(f"  Total items: {stats[1]}")
                    print(f"  Price range: {stats[2]:,.0f} - {stats[3]:,.0f}")
                    print(f"  Average price: {stats[4]:,.0f}")
                
                # Show sample of inserted data
                cursor = conn.execute(f"""
                    SELECT id, nama_barang, harga_barang, page_number 
                    FROM {SCHEMA_NAME}.{TABLE_NAME} 
                    ORDER BY id DESC 
                    LIMIT 5
                """)
                recent = cursor.fetchall()
                if recent:
                    print(f"\nRecent inserts:")
                    for row in recent:
                        print(f"  ID {row[0]}: {row[1]} - Rp {row[2]:,.0f} (page {row[3]})")
                
        except Exception as e:
            print(f"\nDatabase error: {e}")
            import traceback
            traceback.print_exc()
            return 0
        
        return success_count

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Catalog JSON to PostgreSQL - Upload hasil ekstraksi ke database"
    )
    parser.add_argument(
        "json_file",
        help="Path to JSON file (extraction_result.json)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear table before uploading"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for inserts (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Validate file
    if not os.path.exists(args.json_file):
        print(f"✗ File not found: {args.json_file}")
        return 1
    
    print("\n" + "="*60)
    print("CATALOG JSON TO POSTGRESQL UPLOADER")
    print("="*60)
    print(f"JSON File: {args.json_file}")
    print(f"Clear table: {args.clear}")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize and upload
    uploader = CatalogJsonToPostgres()
    success = uploader.upload_to_postgres(
        args.json_file, 
        clear_first=args.clear,
        batch_size=args.batch_size
    )
    
    if success > 0:
        print("\n" + "="*60)
        print("✅ UPLOAD BERHASIL!")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("❌ UPLOAD GAGAL!")
        print("="*60)
        return 1

if __name__ == "__main__":
    exit(main())