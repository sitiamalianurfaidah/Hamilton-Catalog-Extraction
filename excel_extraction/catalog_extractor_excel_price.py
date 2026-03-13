"""
Catalog Extractor Terintegrasi - Versi dengan Koreksi Kolom OPERASIONAL
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Any
import re
from openai import OpenAI
import os
from datetime import datetime
import numpy as np

# Configuration
try:
    from config import Config
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", Config.DASHSCOPE_API_KEY)
    QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", Config.QWEN_BASE_URL)
    QWEN_MAX_MODEL = os.getenv("QWEN_MAX_MODEL", Config.QWEN_MAX_MODEL)
except ImportError:
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-0fc24c229d174d1b99624a49544b1c7a")
    QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    QWEN_MAX_MODEL = "qwen-max"


class ExcelDataExtractor:
    """Excel Raw Data Extractor for Cash Fund Usage Reports"""
    
    def __init__(self):
        """Initialize extractor"""
        pass
    
    def debug_print_dataframe(self, df: pd.DataFrame, num_rows: int = 15):
        """Print dataframe for debugging"""
        print("\n" + "="*50)
        print("DEBUG - Dataframe preview:")
        print("="*50)
        
        for idx in range(min(num_rows, len(df))):
            row = df.iloc[idx]
            print(f"\nRow {idx}:")
            for col_idx in range(min(9, len(row))):
                val = row[col_idx]
                if pd.notna(val):
                    print(f"  Col {col_idx}: {val} ({type(val)})")
    
    def detect_header_row(self, df: pd.DataFrame) -> int:
        """
        Detect which row contains the headers
        """
        header_keywords = ['tanggal', 'dana masuk', 'operasional', 'keterangan', 'saldo']
        
        print("\n[DEBUG] Searching for header row...")
        for idx, row in df.iterrows():
            row_values = []
            for val in row.values:
                if pd.notna(val):
                    str_val = str(val).lower().strip()
                    if str_val and str_val != 'nan':
                        row_values.append(str_val)
            
            row_str = ' '.join(row_values)
            matches = sum(1 for keyword in header_keywords if keyword in row_str)
            
            if row_values:
                print(f"  Row {idx}: matches={matches}, values={row_values[:5]}")
            
            if matches >= 2:
                print(f"  ✓ Header found at row {idx} with {matches} matches")
                return idx
        
        print("  No header row detected, using row 0")
        return 0
    
    def extract_column_groups(self, headers: List[str]) -> Dict[str, List[int]]:
        """
        Identify groups of columns - MODIFIED for merged cells structure
        """
        print("\n[DEBUG] Analyzing headers:")
        groups = {
            'tanggal': [],
            'dana_masuk': [],
            'operasional_label': [],  # Column with "Rp" label
            'operasional_value': [],   # Column with actual value
            'keterangan': [],
            'saldo_label': [],         # Column with "Rp" label for saldo
            'saldo_value': []           # Column with actual saldo value
        }
        
        for idx, header in enumerate(headers):
            if pd.isna(header):
                continue
                
            header_str = str(header).strip()
            header_lower = header_str.lower()
            print(f"  Col {idx}: '{header_str}'")
            
            if 'tanggal' in header_lower:
                groups['tanggal'].append(idx)
                print(f"    → Identified as TANGGAL")
            elif 'dana masuk' in header_lower:
                groups['dana_masuk'].append(idx)
                print(f"    → Identified as DANA MASUK")
            elif 'operasional' in header_lower:
                groups['operasional_label'].append(idx)
                # The actual value is likely in the next column
                groups['operasional_value'].append(idx + 1)
                print(f"    → Identified as OPERASIONAL (label at {idx}, value likely at {idx+1})")
            elif 'keterangan' in header_lower:
                groups['keterangan'].append(idx)
                print(f"    → Identified as KETERANGAN")
            elif 'saldo' in header_lower:
                groups['saldo_label'].append(idx)
                # The actual value is likely in the next column
                groups['saldo_value'].append(idx + 1)
                print(f"    → Identified as SALDO (label at {idx}, value likely at {idx+1})")
        
        return groups
    
    def clean_amount(self, value: Any) -> Optional[float]:
        """
        Clean and convert amount values to float
        """
        if pd.isna(value):
            return None
        
        # If it's already a number (this is what we want!)
        if isinstance(value, (int, float, np.float64, np.int64)):
            result = float(value)
            return result
        
        # If it's a string, try to clean it
        if isinstance(value, str):
            # Remove 'Rp', spaces, and other non-numeric characters
            cleaned = re.sub(r'[^\d,-]', '', value)
            cleaned = cleaned.replace(',', '.')
            try:
                result = float(cleaned) if cleaned else None
                return result
            except ValueError:
                return None
        
        return None
    
    def is_saldo_row(self, row_values: List[Any]) -> bool:
        """Check if this row is just a saldo row (should be skipped)"""
        for val in row_values:
            if pd.notna(val) and 'sisa saldo' in str(val).lower():
                return True
        return False
    
    def split_items(self, keterangan: str) -> List[str]:
        """Split multiple items in keterangan"""
        if not keterangan or pd.isna(keterangan):
            return []
        
        # Split by comma and clean up
        items = [item.strip() for item in str(keterangan).split(',')]
        # Remove empty items
        items = [item for item in items if item and item.lower() != 'sisa saldo']
        
        return items
    
    def extract_transactions(self, excel_path: str, sheet_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract transactions from Excel file - FIXED for merged cells structure
        """
        print("\n" + "="*70)
        print("EXCEL DATA EXTRACTOR - Extracting Transactions")
        print("="*70 + "\n")
        
        # Load Excel file
        print(f"[Step 1] Loading Excel file: {excel_path}")
        excel_file = pd.ExcelFile(excel_path)
        
        # Get sheet name
        if sheet_name is None:
            sheet_name = excel_file.sheet_names[0]
            print(f"  Using first sheet: {sheet_name}")
        else:
            print(f"  Using sheet: {sheet_name}")
        
        # Read the sheet with no headers
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        print(f"  Raw data shape: {df.shape}")
        
        # Debug: Print first few rows
        self.debug_print_dataframe(df)
        
        # Detect header row
        print("\n[Step 2] Detecting header row...")
        header_row = self.detect_header_row(df)
        print(f"  Header found at row: {header_row + 1}")
        
        # Set headers
        headers = df.iloc[header_row].tolist()
        
        # Extract column groups
        print("\n[Step 3] Identifying column groups...")
        column_groups = self.extract_column_groups(headers)
        
        for group_name, indices in column_groups.items():
            if indices:
                print(f"  {group_name}: columns {indices}")
        
        # Check if we found required columns
        if not column_groups['operasional_value']:
            print("\n[ERROR] OPERASIONAL value column not found!")
            return []
        
        if not column_groups['keterangan']:
            print("\n[ERROR] KETERANGAN column not found!")
            return []
        
        # Extract data rows (after header)
        data_rows = df.iloc[header_row + 1:].reset_index(drop=True)
        print(f"\n[Step 4] Processing {len(data_rows)} data rows...")
        
        # Process each row into transactions
        transactions = []
        current_page = 1
        last_valid_date = None
        
        for row_idx, row in data_rows.iterrows():
            actual_row_number = header_row + 1 + row_idx + 1
            print(f"\n[DEBUG Row {actual_row_number}] Processing row:")
            
            # Skip rows that are just "Sisa Saldo"
            if self.is_saldo_row(row):
                print(f"  → Skipping: Sisa Saldo row")
                continue
            
            # Get tanggal (date)
            tanggal = None
            if column_groups['tanggal']:
                col_idx = column_groups['tanggal'][0]
                val = row[col_idx]
                if pd.notna(val) and str(val).strip():
                    if isinstance(val, datetime):
                        tanggal = val.strftime('%Y-%m-%d')
                    else:
                        tanggal = str(val).strip()
                    if tanggal:
                        last_valid_date = tanggal
                        current_page += 1
                        print(f"  → Date found: {tanggal} (new page: {current_page})")
                else:
                    tanggal = last_valid_date
                    print(f"  → Using last date: {tanggal}")
            
            # EXTRACT OPERASIONAL VALUES - FIXED: Read from operasional_value column
            operasional_values = []
            print(f"  → Checking OPERASIONAL value columns: {column_groups['operasional_value']}")
            
            for col_idx in column_groups['operasional_value']:
                val = row[col_idx]
                print(f"    Col {col_idx} value: '{val}' (type: {type(val)})")
                
                if pd.notna(val):
                    # This should be the actual number (float/int)
                    cleaned = self.clean_amount(val)
                    if cleaned is not None and cleaned != 0:
                        operasional_values.append(cleaned)
                        print(f"    ✓ Added operasional: {cleaned}")
            
            # Extract KETERANGAN values
            keterangan_values = []
            print(f"  → Checking KETERANGAN columns: {column_groups['keterangan']}")
            
            for col_idx in column_groups['keterangan']:
                val = row[col_idx]
                print(f"    Col {col_idx} value: '{val}'")
                
                if pd.notna(val) and str(val).strip():
                    str_val = str(val).strip()
                    # Skip if it's just "Sisa Saldo"
                    if 'sisa saldo' not in str_val.lower():
                        keterangan_values.append(str_val)
                        print(f"    ✓ Added keterangan: {str_val[:30]}...")
            
            # If we have operasional values, process them
            if operasional_values:
                print(f"  → Found {len(operasional_values)} operasional values: {operasional_values}")
                
                # If we have keterangan, split into items
                if keterangan_values:
                    for ket_val in keterangan_values:
                        # Split multiple items in keterangan
                        items = self.split_items(ket_val)
                        print(f"    Split '{ket_val[:30]}...' into {len(items)} items: {items}")
                        
                        if not items:
                            continue
                        
                        # For this data structure, we usually have 1 operasional value per row
                        # And multiple items in keterangan
                        if len(operasional_values) == 1:
                            # One amount for multiple items - assign same amount to each
                            op_val = operasional_values[0]
                            for item in items:
                                transaction = {
                                    "tanggal": tanggal or last_valid_date,
                                    "operasional": op_val,
                                    "keterangan": item,
                                    "_page": current_page
                                }
                                transactions.append(transaction)
                                print(f"    ✓ Added transaction: {item[:20]}... Rp{op_val}")
                        
                        elif len(operasional_values) == len(items):
                            # Match each item with its amount
                            for i, item in enumerate(items):
                                transaction = {
                                    "tanggal": tanggal or last_valid_date,
                                    "operasional": operasional_values[i],
                                    "keterangan": item,
                                    "_page": current_page
                                }
                                transactions.append(transaction)
                                print(f"    ✓ Added transaction: {item[:20]}... Rp{operasional_values[i]}")
                        
                        else:
                            # Default: use first amount for all items
                            op_val = operasional_values[0] if operasional_values else None
                            for item in items:
                                transaction = {
                                    "tanggal": tanggal or last_valid_date,
                                    "operasional": op_val,
                                    "keterangan": item,
                                    "_page": current_page
                                }
                                transactions.append(transaction)
                                print(f"    ✓ Added transaction: {item[:20]}... Rp{op_val}")
                else:
                    # No keterangan, but have operasional - create generic transactions
                    for op_val in operasional_values:
                        transaction = {
                            "tanggal": tanggal or last_valid_date,
                            "operasional": op_val,
                            "keterangan": "Transaksi Operasional",
                            "_page": current_page
                        }
                        transactions.append(transaction)
                        print(f"    ✓ Added generic transaction: Rp{op_val}")
            
            elif keterangan_values:
                # Have keterangan but no operasional - create transactions with null price
                print(f"  → Found keterangan but no operasional values")
                for ket_val in keterangan_values:
                    items = self.split_items(ket_val)
                    for item in items:
                        transaction = {
                            "tanggal": tanggal or last_valid_date,
                            "operasional": None,
                            "keterangan": item,
                            "_page": current_page
                        }
                        transactions.append(transaction)
                        print(f"    ✓ Added transaction (no price): {item[:30]}...")
        
        print(f"\n[SUMMARY] Found {len(transactions)} total transactions")
        
        # Debug: Print first few transactions with their prices
        if transactions:
            print("\n[DEBUG] First 15 transactions with prices:")
            transactions_with_price = [t for t in transactions if t['operasional'] is not None]
            transactions_without_price = [t for t in transactions if t['operasional'] is None]
            
            print(f"\n  Transactions WITH price: {len(transactions_with_price)}")
            for i, t in enumerate(transactions_with_price[:10]):
                print(f"    {i+1}. {t['keterangan'][:30]}... - Rp {t['operasional']:,.0f} (Page {t['_page']})")
            
            print(f"\n  Transactions WITHOUT price: {len(transactions_without_price)}")
            for i, t in enumerate(transactions_without_price[:5]):
                print(f"    {i+1}. {t['keterangan'][:30]}... - No price (Page {t['_page']})")
        
        return transactions


class LLMSemanticProcessor:
    """LLM Semantic Reasoning for Product Details Extraction"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize with OpenAI client for Qwen"""
        self.api_key = api_key or DASHSCOPE_API_KEY
        self.base_url = base_url or QWEN_BASE_URL
        self.qwen_model = QWEN_MAX_MODEL
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def extract_product_details_llm(self, keterangan_text: str, operasional_amount: float, page_num: int) -> Dict[str, Any]:
        """
        Use LLM to parse keterangan text and extract product details
        """
        if not keterangan_text or keterangan_text.strip() == "":
            return {
                "nama_barang": "Tidak ada keterangan",
                "jenis_tipe_barang": None,
                "spesifikasi_detail_barang": {},
                "merk_barang": None,
                "harga_barang": operasional_amount,
                "_page": page_num
            }
        
        harga_barang = operasional_amount if operasional_amount is not None else None
        
        # Special handling for BBM KLX
        if "BBM KLX" in keterangan_text:
            # Check if it mentions number of times
            spesifikasi = {}
            if "2 Kali Isi" in keterangan_text:
                spesifikasi = {"isi_ulang": 2}
            elif "1 Kali Isi" in keterangan_text:
                spesifikasi = {"isi_ulang": 1}
            
            return {
                "nama_barang": "BBM KLX",
                "jenis_tipe_barang": "Bahan Bakar Minyak",
                "spesifikasi_detail_barang": spesifikasi,
                "merk_barang": None,
                "harga_barang": harga_barang,
                "_page": page_num
            }
        
        # Special handling for Tali Rafia
        if "Tali Rafia" in keterangan_text:
            return {
                "nama_barang": "Tali Rafia",
                "jenis_tipe_barang": "Alat dan Bahan",
                "spesifikasi_detail_barang": {},
                "merk_barang": None,
                "harga_barang": harga_barang,
                "_page": page_num
            }
        
        prompt = f"""Analyze this transaction description and extract product information.

DESCRIPTION: "{keterangan_text}"
AMOUNT: {f'Rp {operasional_amount:,.0f}' if operasional_amount else 'Tidak ada nominal'}

TASK:
Extract the product information into this EXACT JSON structure:
{{
    "nama_barang": "Product name",
    "jenis_tipe_barang": "Product type/category or null if unclear",
    "spesifikasi_detail_barang": {{}},
    "merk_barang": "Brand name or null if unclear",
    "harga_barang": {harga_barang if harga_barang is not None else 'null'}
}}

RULES:
1. nama_barang: Extract the main product/item name (be specific)
2. jenis_tipe_barang: If not clear, use null
3. spesifikasi_detail_barang: Create an object with relevant specs
4. merk_barang: If mentioned, extract brand; otherwise use null
5. harga_barang: MUST BE SET TO {harga_barang if harga_barang is not None else 'null'} - this is the actual amount from the transaction

IMPORTANT:
- Return ONLY valid JSON, no other text
- Keep all field names in Indonesian as shown
- The harga_barang field MUST be set to the amount provided above

Now process this transaction:
Description: "{keterangan_text}"
Amount: {f'Rp {operasional_amount:,.0f}' if operasional_amount else 'Tidak ada nominal'}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.qwen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from the response
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = result_text[start_idx:end_idx]
                product = json.loads(json_str)
            else:
                product = json.loads(result_text)
            
            # FORCE harga_barang to be the actual amount from transaction
            product["harga_barang"] = harga_barang
            
            # Add page number
            product["_page"] = page_num
            
            return product
            
        except Exception as e:
            print(f"  ✗ LLM error: {str(e)}")
            
            # Fallback: create generic product entry with the actual price
            return {
                "nama_barang": keterangan_text[:100],
                "jenis_tipe_barang": None,
                "spesifikasi_detail_barang": {
                    "deskripsi": keterangan_text
                },
                "merk_barang": None,
                "harga_barang": harga_barang,  # Use the actual price
                "_page": page_num
            }


class IntegratedCatalogExtractor:
    """Integrated extractor combining Excel extraction and LLM processing"""
    
    def __init__(self, api_key: str = None):
        """Initialize both extractors"""
        self.excel_extractor = ExcelDataExtractor()
        self.llm_processor = LLMSemanticProcessor(api_key=api_key)
    
    def process_excel_file(self, excel_path: str, sheet_name: Optional[str] = None, 
                          output_path: Optional[str] = None, max_items: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete workflow: Extract from Excel and process with LLM
        """
        print(f"\n{'='*70}")
        print(f"INTEGRATED CATALOG EXTRACTOR")
        print(f"{'='*70}\n")
        
        try:
            # Step 1: Extract transactions from Excel
            print("[PHASE 1] Extracting transactions from Excel...")
            transactions = self.excel_extractor.extract_transactions(excel_path, sheet_name)
            print(f"\n  Total transactions extracted: {len(transactions)}")
            
            if len(transactions) == 0:
                print("\n[WARNING] No transactions found!")
                return {
                    "total_items": 0,
                    "stats": {
                        "total_items": 0,
                        "items_with_price": 0,
                        "coverage_percent": 0
                    },
                    "failed_pages": [],
                    "items": []
                }
            
            # Count transactions with price before LLM
            transactions_with_price = sum(1 for t in transactions if t['operasional'] is not None)
            print(f"\n  Transactions with price before LLM: {transactions_with_price}/{len(transactions)}")
            
            # Apply limit if specified
            if max_items and max_items < len(transactions):
                transactions = transactions[:max_items]
                print(f"  Limited to {max_items} items for processing")
            
            # Step 2: Process each transaction with LLM
            print("\n[PHASE 2] Processing transactions with LLM...")
            items = []
            failed_pages = set()
            
            for idx, transaction in enumerate(transactions):
                print(f"\n  Processing {idx + 1}/{len(transactions)}:")
                print(f"    Keterangan: {transaction['keterangan'][:100]}")
                print(f"    Amount from Excel: {transaction['operasional']}")
                print(f"    Page: {transaction['_page']}")
                
                try:
                    product = self.llm_processor.extract_product_details_llm(
                        transaction['keterangan'],
                        transaction['operasional'],
                        transaction['_page']
                    )
                    items.append(product)
                    print(f"    ✓ Success: {product.get('nama_barang', 'Unknown')} - Price: {product.get('harga_barang')}")
                    
                except Exception as e:
                    print(f"    ✗ Failed: {str(e)}")
                    failed_pages.add(transaction['_page'])
                    # Add fallback item with the actual price
                    items.append({
                        "nama_barang": transaction['keterangan'][:100],
                        "jenis_tipe_barang": None,
                        "spesifikasi_detail_barang": {
                            "deskripsi": transaction['keterangan']
                        },
                        "merk_barang": None,
                        "harga_barang": transaction['operasional'],  # Use the actual price
                        "_page": transaction['_page']
                    })
            
            # Step 3: Compile statistics
            print("\n[PHASE 3] Compiling statistics...")
            items_with_price = sum(1 for item in items if item.get('harga_barang') is not None)
            total_items = len(items)
            
            result = {
                "total_items": total_items,
                "stats": {
                    "total_items": total_items,
                    "items_with_price": items_with_price,
                    "coverage_percent": (items_with_price / total_items * 100) if total_items > 0 else 0
                },
                "failed_pages": sorted(list(failed_pages)),
                "items": items
            }
            
            # Step 4: Save result
            print("\n[PHASE 4] Saving results...")
            output_file = self.save_result(result, excel_path, output_path)
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"✓ PROCESSING COMPLETE")
            print(f"{'='*70}")
            print(f"  Total items: {result['total_items']}")
            print(f"  Items with price: {result['stats']['items_with_price']}")
            print(f"  Coverage: {result['stats']['coverage_percent']:.2f}%")
            if result['failed_pages']:
                print(f"  Failed pages: {result['failed_pages']}")
            print(f"  Output file: {output_file}")
            
            return result
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_result(self, result: Dict[str, Any], excel_path: str, output_path: Optional[str] = None) -> str:
        """Save result to JSON file"""
        if output_path is None:
            excel_file = Path(excel_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = excel_file.parent / f"{excel_file.stem}_catalog_result_{timestamp}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return str(output_path)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Integrated Catalog Extractor - Excel to JSON with LLM processing"
    )
    parser.add_argument(
        "excel_file",
        help="Path to Excel file containing cash fund usage data"
    )
    parser.add_argument(
        "--sheet", "-s",
        help="Sheet name to process (default: first sheet)",
        default=None
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file path",
        default=None
    )
    parser.add_argument(
        "--api-key",
        help="DashScope API key",
        default=None
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Maximum number of items to process (for testing)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Initialize integrated extractor
    extractor = IntegratedCatalogExtractor(api_key=args.api_key)
    
    # Process file
    try:
        result = extractor.process_excel_file(
            args.excel_file, 
            args.sheet, 
            args.output,
            args.max_items
        )
        
        # Print sample items with focus on prices
        if result["items"]:
            print(f"\n📋Sample items with prices:")
            
            # Show items that have prices
            items_with_price = [item for item in result["items"] if item.get('harga_barang') is not None]
            items_without_price = [item for item in result["items"] if item.get('harga_barang') is None]
            
            print(f"\nItems WITH price ({len(items_with_price)}):")
            for i, item in enumerate(items_with_price[:10]):
                print(f"\n  {i+1}. {item.get('nama_barang')}")
                print(f"     Harga: Rp {item.get('harga_barang'):,.0f}")
                print(f"     Page: {item.get('_page')}")
            
            print(f"\nItems WITHOUT price ({len(items_without_price)}):")
            for i, item in enumerate(items_without_price[:5]):
                print(f"\n  {i+1}. {item.get('nama_barang')}")
                print(f"     Page: {item.get('_page')}")
        else:
            print("\nNo items were extracted.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())