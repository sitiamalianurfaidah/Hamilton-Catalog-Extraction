"""
Catalog Search Demo - Version without embeddings
Mencari produk menggunakan pencarian teks biasa (LIKE)
"""

import os
import json
import argparse
import psycopg
from tabulate import tabulate
from typing import List, Dict, Any, Optional

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


class CatalogSearchDemo:
    """Kelas untuk demo pencarian pada katalog (tanpa embedding)"""
    
    def __init__(self):
        """Initialize tanpa embedding model"""
        print("✓ Menggunakan pencarian teks biasa (LIKE)")
        pass
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for products using text matching
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching products
        """
        results = []
        
        try:
            with psycopg.connect(DB_URI) as conn:
                # Set schema
                conn.execute(f"SET search_path TO {SCHEMA_NAME};")
                
                # Search using text matching on multiple fields
                cursor = conn.execute(f"""
                    SELECT 
                        nama_barang,
                        jenis_tipe_barang,
                        spesifikasi,
                        merk_barang,
                        harga_barang,
                        tanggal_transaksi,
                        raw_keterangan
                    FROM {TABLE_NAME}
                    WHERE 
                        LOWER(nama_barang) LIKE LOWER(%s) OR
                        LOWER(jenis_tipe_barang) LIKE LOWER(%s) OR
                        LOWER(merk_barang) LIKE LOWER(%s) OR
                        LOWER(raw_keterangan) LIKE LOWER(%s) OR
                        LOWER(spesifikasi::text) LIKE LOWER(%s)
                    ORDER BY 
                        CASE 
                            WHEN LOWER(nama_barang) LIKE LOWER(%s) THEN 1
                            WHEN LOWER(jenis_tipe_barang) LIKE LOWER(%s) THEN 2
                            WHEN LOWER(merk_barang) LIKE LOWER(%s) THEN 3
                            ELSE 4
                        END
                    LIMIT %s;
                """, (
                    f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%',  # WHERE clause
                    f'%{query}%', f'%{query}%', f'%{query}%',  # ORDER BY clause
                    limit
                ))
                
                for row in cursor:
                    results.append({
                        "nama_barang": row[0],
                        "jenis_tipe_barang": row[1],
                        "spesifikasi": row[2],
                        "merk_barang": row[3],
                        "harga_barang": row[4],
                        "tanggal_transaksi": row[5],
                        "raw_keterangan": row[6]
                    })
        
        except Exception as e:
            print(f"✗ Search error: {e}")
        
        return results
    
    def print_results(self, results: List[Dict[str, Any]], query: str):
        """
        Print search results in formatted table
        
        Args:
            results: List of search results
            query: Original query
        """
        print("\n" + "="*80)
        print(f"HASIL PENCARIAN: '{query}'")
        print("="*80)
        
        if not results:
            print("Tidak ada hasil ditemukan.")
            return
        
        # Prepare data for table
        table_data = []
        for i, r in enumerate(results, 1):
            # Format spesifikasi
            spesifikasi = r.get('spesifikasi', {})
            if isinstance(spesifikasi, dict):
                spec_str = json.dumps(spesifikasi, ensure_ascii=False)[:50]
            else:
                spec_str = str(spesifikasi)[:50]
            
            # Format raw_keterangan
            keterangan = r.get('raw_keterangan', '')[:40]
            
            table_data.append([
                i,
                r['nama_barang'][:25] if r['nama_barang'] else '-',
                r['jenis_tipe_barang'][:15] if r['jenis_tipe_barang'] else '-',
                spec_str,
                r['merk_barang'][:10] if r['merk_barang'] else '-',
                f"Rp {r['harga_barang']:,.0f}" if r['harga_barang'] else "-",
                keterangan
            ])
        
        # Print table
        headers = ["No", "Nama Barang", "Jenis", "Spesifikasi", "Merk", "Harga", "Keterangan"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        print(f"\n✓ Menampilkan {len(results)} hasil teratas")
    
    def run_demo(self, initial_query: Optional[str] = None):
        """
        Run interactive search demo
        
        Args:
            initial_query: Optional initial query
        """
        print("\n" + "="*60)
        print("CATALOG SEARCH DEMO - Text Search (No Embeddings)")
        print("="*60)
        print("Ketik 'quit' untuk keluar, 'clear' untuk hapus layar\n")
        
        query = initial_query
        while True:
            if not query:
                query = input("\n🔍 Masukkan kata kunci pencarian: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Terima kasih! Sampai jumpa.")
                break
            
            if query.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                query = None
                continue
            
            if query:
                results = self.search(query)
                self.print_results(results, query)
            
            query = None  # Reset for next iteration
    
    def count_items(self) -> int:
        """Count total items in database"""
        try:
            with psycopg.connect(DB_URI) as conn:
                conn.execute(f"SET search_path TO {SCHEMA_NAME};")
                cursor = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"✗ Error counting items: {e}")
            return 0
    
    def search_by_category(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search products by category/jenis barang
        
        Args:
            category: Category to search for
            limit: Maximum number of results
            
        Returns:
            List of products in category
        """
        results = []
        
        try:
            with psycopg.connect(DB_URI) as conn:
                conn.execute(f"SET search_path TO {SCHEMA_NAME};")
                
                cursor = conn.execute(f"""
                    SELECT 
                        nama_barang,
                        jenis_tipe_barang,
                        spesifikasi,
                        merk_barang,
                        harga_barang,
                        tanggal_transaksi
                    FROM {TABLE_NAME}
                    WHERE LOWER(jenis_tipe_barang) LIKE LOWER(%s)
                    ORDER BY tanggal_transaksi DESC
                    LIMIT %s;
                """, (f'%{category}%', limit))
                
                for row in cursor:
                    results.append({
                        "nama_barang": row[0],
                        "jenis_tipe_barang": row[1],
                        "spesifikasi": row[2],
                        "merk_barang": row[3],
                        "harga_barang": row[4],
                        "tanggal_transaksi": row[5]
                    })
        
        except Exception as e:
            print(f"✗ Search error: {e}")
        
        return results
    
    def advanced_search(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Advanced search with multiple filters
        
        Args:
            **kwargs: Search filters (nama, merk, min_harga, max_harga, dll)
            
        Returns:
            List of matching products
        """
        conditions = []
        params = []
        
        if kwargs.get('nama'):
            conditions.append("LOWER(nama_barang) LIKE LOWER(%s)")
            params.append(f"%{kwargs['nama']}%")
        
        if kwargs.get('merk'):
            conditions.append("LOWER(merk_barang) LIKE LOWER(%s)")
            params.append(f"%{kwargs['merk']}%")
        
        if kwargs.get('jenis'):
            conditions.append("LOWER(jenis_tipe_barang) LIKE LOWER(%s)")
            params.append(f"%{kwargs['jenis']}%")
        
        if kwargs.get('min_harga'):
            conditions.append("harga_barang >= %s")
            params.append(kwargs['min_harga'])
        
        if kwargs.get('max_harga'):
            conditions.append("harga_barang <= %s")
            params.append(kwargs['max_harga'])
        
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        
        results = []
        try:
            with psycopg.connect(DB_URI) as conn:
                conn.execute(f"SET search_path TO {SCHEMA_NAME};")
                
                query = f"""
                    SELECT 
                        nama_barang,
                        jenis_tipe_barang,
                        spesifikasi,
                        merk_barang,
                        harga_barang,
                        tanggal_transaksi
                    FROM {TABLE_NAME}
                    WHERE {where_clause}
                    ORDER BY tanggal_transaksi DESC
                    LIMIT %s;
                """
                params.append(kwargs.get('limit', 20))
                
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    results.append({
                        "nama_barang": row[0],
                        "jenis_tipe_barang": row[1],
                        "spesifikasi": row[2],
                        "merk_barang": row[3],
                        "harga_barang": row[4],
                        "tanggal_transaksi": row[5]
                    })
        
        except Exception as e:
            print(f"✗ Advanced search error: {e}")
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Catalog Search Demo - Pencarian Teks Biasa"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (if not provided, interactive mode)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=5,
        help="Number of results to show (default: 5)"
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Show total items in database"
    )
    parser.add_argument(
        "--category", "-c",
        help="Search by category/jenis barang"
    )
    parser.add_argument(
        "--merk", "-m",
        help="Search by merk/brand"
    )
    parser.add_argument(
        "--min-price",
        type=float,
        help="Minimum price filter"
    )
    parser.add_argument(
        "--max-price",
        type=float,
        help="Maximum price filter"
    )
    
    args = parser.parse_args()
    
    # Initialize
    searcher = CatalogSearchDemo()
    
    # Show count if requested
    if args.count:
        total = searcher.count_items()
        print(f"\n📊 Total items di database: {total}")
        return 0
    
    # Advanced search with filters
    if args.category or args.merk or args.min_price or args.max_price:
        results = searcher.advanced_search(
            jenis=args.category,
            merk=args.merk,
            min_harga=args.min_price,
            max_harga=args.max_price,
            limit=args.limit
        )
        filter_desc = []
        if args.category: filter_desc.append(f"kategori={args.category}")
        if args.merk: filter_desc.append(f"merk={args.merk}")
        if args.min_price: filter_desc.append(f"min={args.min_price}")
        if args.max_price: filter_desc.append(f"max={args.max_price}")
        searcher.print_results(results, f"Filter: {', '.join(filter_desc)}")
    
    # Category search
    elif args.category:
        results = searcher.search_by_category(args.category, args.limit)
        searcher.print_results(results, f"Kategori: {args.category}")
    
    # Regular search
    elif args.query:
        results = searcher.search(args.query, args.limit)
        searcher.print_results(results, args.query)
    else:
        searcher.run_demo()
    
    return 0


if __name__ == "__main__":
    exit(main())