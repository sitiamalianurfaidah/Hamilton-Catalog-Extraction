"""
Catalog Extractor Terintegrasi - Versi dengan Ekstraksi Harga per Item dari Nota
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
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


class NoteMatcher:
    """Class untuk mencocokkan transaksi dengan nota dan mengekstrak harga per item"""
    
    def __init__(self):
        """Initialize note matcher"""
        self.notes_by_date = {}
        self.total_by_date = {}
    
    def normalize_date(self, date_str: str) -> str:
        """Normalize date string for comparison"""
        if not date_str or pd.isna(date_str):
            return ""
        
        date_str = str(date_str).strip()
        
        # Try different date formats
        # Format: YYYY-MM-DD HH:MM:SS
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_str)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        
        # Format: DD-MM-YYYY or DD/MM/YYYY
        match = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})', date_str)
        if match:
            day, month, year = match.groups()
            # Pad day and month to 2 digits
            day = day.zfill(2)
            month = month.zfill(2)
            # Handle 2-digit year
            if len(year) == 2:
                year = '20' + year if int(year) < 50 else '19' + year
            return f"{year}-{month}-{day}"
        
        return date_str
    
    def extract_items_with_prices_from_note(self, note_text: str) -> List[Dict[str, Any]]:
        """
        Extract individual items with their prices from a note text
        Contoh: "BBM KLX 50.000, Canebo 25.000, Sapu Mobil 20.000, Tisu Mobil 15.000"
        """
        items_with_prices = []
        
        # Pattern 1: Item dengan harga di sebelahnya (format: "Nama Item Harga")
        # Contoh: "BBM KLX 50000", "Canebo 25000", dll.
        pattern1 = r'([A-Za-z\s]+?)\s+(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)(?:\s|,|$)'
        
        # Pattern 2: Item dengan format "Nama Item: Rp Harga"
        pattern2 = r'([A-Za-z\s]+?)\s*:\s*(?:Rp|RP|rp)?\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)'
        
        # Pattern 3: Format dengan "Rp" sebelum harga
        pattern3 = r'([A-Za-z\s]+?)\s+(?:Rp|RP|rp)?\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)'
        
        # Coba semua pattern
        for pattern in [pattern1, pattern2, pattern3]:
            matches = re.finditer(pattern, note_text)
            for match in matches:
                item_name = match.group(1).strip()
                price_str = match.group(2).replace('.', '').replace(',', '')
                
                # Skip if item name is too short or looks like metadata
                if len(item_name) < 3 or item_name.lower() in ['rp', 'total', 'jumlah']:
                    continue
                
                try:
                    price = float(price_str)
                    items_with_prices.append({
                        'item': item_name,
                        'price': price,
                        'matched_pattern': 'pattern_with_price'
                    })
                    print(f"      ✓ Extracted item with price: {item_name} = Rp {price:,.0f}")
                except ValueError:
                    pass
            
            if items_with_prices:
                break
        
        # Jika tidak ditemukan item dengan harga, coba cari format daftar
        if not items_with_prices:
            # Split by comma and try to find prices
            parts = [p.strip() for p in note_text.split(',')]
            for part in parts:
                # Cari harga di setiap bagian
                price_match = re.search(r'(?:Rp|RP|rp)?\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)', part)
                if price_match:
                    item_name = re.sub(r'(?:Rp|RP|rp)?\.?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?', '', part).strip()
                    price_str = price_match.group(1).replace('.', '').replace(',', '')
                    
                    if item_name and len(item_name) > 2:
                        try:
                            price = float(price_str)
                            items_with_prices.append({
                                'item': item_name,
                                'price': price,
                                'matched_pattern': 'comma_separated'
                            })
                            print(f"      ✓ Extracted item with price (comma): {item_name} = Rp {price:,.0f}")
                        except ValueError:
                            items_with_prices.append({
                                'item': part,
                                'price': None,
                                'matched_pattern': 'no_price'
                            })
                    else:
                        items_with_prices.append({
                            'item': part,
                            'price': None,
                            'matched_pattern': 'no_price'
                        })
                else:
                    items_with_prices.append({
                        'item': part,
                        'price': None,
                        'matched_pattern': 'no_price'
                    })
        
        return items_with_prices
    
    def extract_notes_from_sheet(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Extract notes from lampiran sheet dengan ekstraksi harga per item
        """
        notes = {}
        current_date = None
        current_items = []
        
        print("\n[DEBUG] Extracting notes from lampiran sheet...")
        
        for idx, row in df.iterrows():
            # Check first column for date
            val = row[0] if len(row) > 0 else None
            
            if pd.notna(val):
                val_str = str(val).strip()
                
                # Skip header rows
                if any(keyword in val_str.lower() for keyword in ['nota belanja', 'pt.', 'jalan tol', 'monitoring']):
                    continue
                
                # Check if this is a date
                date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', val_str) or \
                             re.search(r'(\d{4}-\d{2}-\d{2})', val_str)
                
                if date_match:
                    # Save previous date's notes
                    if current_date and current_items:
                        # Gabungkan semua item untuk tanggal ini
                        combined_text = ' '.join([item['full_text'] for item in current_items if item.get('full_text')])
                        
                        # Ekstrak item dengan harga dari teks gabungan
                        items_with_prices = self.extract_items_with_prices_from_note(combined_text)
                        
                        if items_with_prices:
                            notes[current_date] = items_with_prices
                            print(f"  Found {len(items_with_prices)} items with prices for date {current_date}")
                        else:
                            # Jika tidak bisa ekstrak harga, simpan sebagai items biasa
                            notes[current_date] = current_items
                            print(f"  Found {len(current_items)} items (no prices) for date {current_date}")
                    
                    # Start new date
                    current_date = val_str
                    current_items = []
                    print(f"  New date found: {current_date}")
                    
                    # Check if there are items in same row
                    if len(row) > 1 and pd.notna(row[1]):
                        item_text = str(row[1]).strip()
                        if item_text and len(item_text) > 3:
                            current_items.append({
                                'item': item_text,
                                'row_idx': idx,
                                'full_text': item_text
                            })
                
                elif current_date:
                    # This might be an item
                    if len(row) > 1 and pd.notna(row[1]):
                        item_text = str(row[1]).strip()
                        if item_text and len(item_text) > 3 and not any(keyword in item_text.lower() for keyword in ['nota belanja', 'pt.', 'jalan tol']):
                            current_items.append({
                                'item': item_text,
                                'row_idx': idx,
                                'full_text': item_text
                            })
        
        # Add last date's notes
        if current_date and current_items:
            combined_text = ' '.join([item['full_text'] for item in current_items if item.get('full_text')])
            items_with_prices = self.extract_items_with_prices_from_note(combined_text)
            
            if items_with_prices:
                notes[current_date] = items_with_prices
                print(f"  Found {len(items_with_prices)} items with prices for date {current_date}")
            else:
                notes[current_date] = current_items
                print(f"  Found {len(current_items)} items (no prices) for date {current_date}")
        
        # Calculate totals
        for date, items in notes.items():
            total = 0
            price_count = 0
            for item in items:
                if item.get('price'):
                    total += item['price']
                    price_count += 1
            
            self.total_by_date[date] = total
            print(f"  Total for {date}: Rp {total:,.0f} (from {price_count} items with prices)")
        
        return notes
    
    def match_transaction_with_notes(self, 
                                     transaction_date: str, 
                                     transaction_amount: float, 
                                     transaction_items: List[str],
                                     notes: Dict[str, List[Dict]]) -> Tuple[List[Dict], bool, bool]:
        """
        Match transaction with notes dan tentukan apakah perlu splitting harga
        Returns (matched_items, exact_match_found, has_per_item_prices)
        """
        # Try to find exact date match
        exact_match = False
        matched_items = []
        has_per_item_prices = False
        
        # Normalize date formats for comparison
        trans_date_norm = self.normalize_date(transaction_date)
        print(f"    Normalized transaction date: {trans_date_norm}")
        
        # First try: exact date match
        for note_date, items in notes.items():
            note_date_norm = self.normalize_date(note_date)
            print(f"    Comparing with note date: {note_date} -> {note_date_norm}")
            
            if trans_date_norm == note_date_norm:
                print(f"  ✓ Found exact date match: {note_date}")
                exact_match = True
                
                # Check if items have per-item prices
                items_with_prices = [item for item in items if item.get('price') is not None]
                
                if items_with_prices:
                    has_per_item_prices = True
                    matched_items = items_with_prices
                    print(f"  ✓ Found {len(items_with_prices)} items with per-item prices")
                    
                    # Verify total matches
                    note_total = sum(item['price'] for item in items_with_prices)
                    if transaction_amount and abs(note_total - transaction_amount) < 100:
                        print(f"  ✓ Note total matches transaction: Rp {note_total:,.0f}")
                    else:
                        print(f"  ⚠ Note total (Rp {note_total:,.0f}) != transaction (Rp {transaction_amount:,.0f})")
                else:
                    # No per-item prices, use items as is
                    has_per_item_prices = False
                    matched_items = items
                    print(f"  ⚠ No per-item prices found, using items as is")
                
                break
        
        # Second try: if no exact date match, try amount match
        if not exact_match and transaction_amount:
            print(f"  No exact date match, trying amount match...")
            for note_date, items in notes.items():
                items_with_prices = [item for item in items if item.get('price') is not None]
                if items_with_prices:
                    note_total = sum(item['price'] for item in items_with_prices)
                    if note_total > 0 and abs(note_total - transaction_amount) < 100:
                        print(f"  ⚠ Found amount match with different date: {note_date}")
                        has_per_item_prices = True
                        matched_items = items_with_prices
                        break
        
        return matched_items, exact_match, has_per_item_prices


class ExcelDataExtractor:
    """Excel Raw Data Extractor for Cash Fund Usage Reports"""
    
    def __init__(self):
        """Initialize extractor"""
        self.note_matcher = NoteMatcher()
        self.all_notes = {}
    
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
        Identify groups of columns
        """
        print("\n[DEBUG] Analyzing headers:")
        groups = {
            'tanggal': [],
            'dana_masuk': [],
            'operasional_label': [],
            'operasional_value': [],
            'keterangan': [],
            'saldo_label': [],
            'saldo_value': []
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
                groups['operasional_value'].append(idx + 1)
                print(f"    → Identified as OPERASIONAL (label at {idx}, value at {idx+1})")
            elif 'keterangan' in header_lower:
                groups['keterangan'].append(idx)
                print(f"    → Identified as KETERANGAN")
            elif 'saldo' in header_lower:
                groups['saldo_label'].append(idx)
                groups['saldo_value'].append(idx + 1)
                print(f"    → Identified as SALDO (label at {idx}, value at {idx+1})")
        
        return groups
    
    def clean_amount(self, value: Any) -> Optional[float]:
        """Clean and convert amount values to float"""
        if pd.isna(value):
            return None
        
        if isinstance(value, (int, float, np.float64, np.int64)):
            return float(value)
        
        if isinstance(value, str):
            cleaned = re.sub(r'[^\d,-]', '', value)
            cleaned = cleaned.replace(',', '.')
            try:
                return float(cleaned) if cleaned else None
            except ValueError:
                return None
        
        return None
    
    def is_saldo_row(self, row_values: List[Any]) -> bool:
        """Check if this row is just a saldo row"""
        for val in row_values:
            if pd.notna(val) and 'sisa saldo' in str(val).lower():
                return True
        return False
    
    def split_items(self, keterangan: str) -> List[str]:
        """Split multiple items in keterangan"""
        if not keterangan or pd.isna(keterangan):
            return []
        
        items = [item.strip() for item in str(keterangan).split(',')]
        items = [item for item in items if item and item.lower() != 'sisa saldo']
        
        return items
    
    def extract_transactions_from_monitoring(self, df: pd.DataFrame, sheet_name: str) -> List[Dict[str, Any]]:
        """
        Extract transactions from monitoring sheet
        """
        print(f"\n[Step] Extracting transactions from monitoring sheet: {sheet_name}")
        
        header_row = self.detect_header_row(df)
        headers = df.iloc[header_row].tolist()
        column_groups = self.extract_column_groups(headers)
        
        if not column_groups['operasional_value'] or not column_groups['keterangan']:
            print("\n[ERROR] Required columns not found!")
            return []
        
        data_rows = df.iloc[header_row + 1:].reset_index(drop=True)
        print(f"\n[Step] Processing {len(data_rows)} data rows...")
        
        transactions = []
        current_page = 1
        last_valid_date = None
        
        for row_idx, row in data_rows.iterrows():
            actual_row_number = header_row + 1 + row_idx + 1
            
            if self.is_saldo_row(row):
                continue
            
            # Get tanggal
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
                else:
                    tanggal = last_valid_date
            
            # Get operasional values
            operasional_values = []
            for col_idx in column_groups['operasional_value']:
                val = row[col_idx]
                if pd.notna(val):
                    cleaned = self.clean_amount(val)
                    if cleaned is not None and cleaned != 0:
                        operasional_values.append(cleaned)
            
            # Get keterangan values
            keterangan_values = []
            for col_idx in column_groups['keterangan']:
                val = row[col_idx]
                if pd.notna(val) and str(val).strip():
                    str_val = str(val).strip()
                    if 'sisa saldo' not in str_val.lower():
                        keterangan_values.append(str_val)
            
            if operasional_values and keterangan_values:
                for ket_val in keterangan_values:
                    items = self.split_items(ket_val)
                    
                    if not items:
                        continue
                    
                    if len(operasional_values) == 1:
                        op_val = operasional_values[0]
                        transaction = {
                            "tanggal": tanggal or last_valid_date,
                            "total_amount": op_val,
                            "items": items,
                            "raw_keterangan": ket_val,
                            "_page": current_page,
                            "_sheet": sheet_name,
                            "_row": actual_row_number
                        }
                        transactions.append(transaction)
        
        print(f"\n[SUMMARY] Found {len(transactions)} transactions in {sheet_name}")
        return transactions
    
    def extract_all_sheets(self, excel_path: str) -> Dict[str, Any]:
        """
        Extract data from all sheets in Excel file
        """
        print("\n" + "="*70)
        print("EXCEL DATA EXTRACTOR - Processing all sheets")
        print("="*70 + "\n")
        
        excel_file = pd.ExcelFile(excel_path)
        all_sheets = excel_file.sheet_names
        print(f"  Found {len(all_sheets)} sheets: {all_sheets}")
        
        # Separate monitoring and lampiran sheets
        monitoring_sheets = []
        lampiran_sheets = []
        
        for sheet in all_sheets:
            if 'lampiran' in sheet.lower() or 'nota' in sheet.lower():
                lampiran_sheets.append(sheet)
            else:
                monitoring_sheets.append(sheet)
        
        print(f"\n  Monitoring sheets: {monitoring_sheets}")
        print(f"  Lampiran sheets: {lampiran_sheets}")
        
        # Extract notes from lampiran sheets
        print("\n[Step 2] Extracting notes from lampiran sheets...")
        all_notes = {}
        
        for sheet in lampiran_sheets:
            print(f"\n  Processing lampiran sheet: {sheet}")
            df = pd.read_excel(excel_file, sheet_name=sheet, header=None)
            notes = self.note_matcher.extract_notes_from_sheet(df)
            all_notes.update(notes)
        
        self.all_notes = all_notes
        print(f"\n  Total notes found: {len(all_notes)} dates")
        
        # Extract transactions from monitoring sheets
        print("\n[Step 3] Extracting transactions from monitoring sheets...")
        all_transactions = []
        
        for sheet in monitoring_sheets:
            print(f"\n  Processing monitoring sheet: {sheet}")
            df = pd.read_excel(excel_file, sheet_name=sheet, header=None)
            transactions = self.extract_transactions_from_monitoring(df, sheet)
            all_transactions.extend(transactions)
        
        print(f"\n[SUMMARY] Total transactions: {len(all_transactions)}")
        
        # Match transactions with notes
        print("\n[Step 4] Matching transactions with notes...")
        matched_transactions = []
        
        for transaction in all_transactions:
            tanggal = transaction['tanggal']
            total_amount = transaction.get('total_amount')
            items = transaction.get('items', [])
            raw_keterangan = transaction.get('raw_keterangan', '')
            
            print(f"\n  Processing transaction:")
            print(f"    Date: {tanggal}")
            print(f"    Total: Rp {total_amount:,.0f}" if total_amount else "    Total: None")
            print(f"    Items from keterangan: {items}")
            print(f"    Raw: {raw_keterangan[:100]}")
            
            # Try to match with notes
            matched_items, exact_match, has_per_item_prices = self.note_matcher.match_transaction_with_notes(
                tanggal, total_amount, items, all_notes
            )
            
            if matched_items and has_per_item_prices:
                # Use matched items with per-item prices from notes
                transaction['matched_items'] = matched_items
                transaction['match_type'] = 'exact_with_prices' if exact_match else 'amount_with_prices'
                transaction['has_per_item_prices'] = True
                
                print(f"    ✓ Matched with {len(matched_items)} items from notes with per-item prices")
                for mitem in matched_items[:3]:
                    print(f"      - {mitem.get('item', '')[:30]}: Rp {mitem.get('price', 0):,.0f}")
            
            elif matched_items and not has_per_item_prices:
                # Use matched items but without per-item prices
                transaction['matched_items'] = matched_items
                transaction['match_type'] = 'exact_no_prices' if exact_match else 'amount_no_prices'
                transaction['has_per_item_prices'] = False
                print(f"    ⚠ Matched with {len(matched_items)} items from notes but no per-item prices")
            
            else:
                # No match found, use original transaction
                transaction['matched_items'] = []
                transaction['match_type'] = 'none'
                transaction['has_per_item_prices'] = False
                print(f"    No match found, using original transaction")
            
            matched_transactions.append(transaction)
        
        return {
            'transactions': matched_transactions,
            'notes': all_notes,
            'stats': {
                'total_transactions': len(matched_transactions),
                'total_notes': len(all_notes),
                'sheets_processed': {
                    'monitoring': monitoring_sheets,
                    'lampiran': lampiran_sheets
                }
            }
        }


class LLMSemanticProcessor:
    """LLM Semantic Reasoning for Product Details Extraction"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or DASHSCOPE_API_KEY
        self.base_url = base_url or QWEN_BASE_URL
        self.qwen_model = QWEN_MAX_MODEL
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def extract_product_details_llm(self, 
                                   item_text: str, 
                                   price: Optional[float], 
                                   page_num: int,
                                   original_keterangan: str = None) -> Dict[str, Any]:
        """
        Use LLM to parse item text and extract product details
        """
        if not item_text or item_text.strip() == "":
            return {
                "nama_barang": "Tidak ada keterangan",
                "jenis_tipe_barang": None,
                "spesifikasi_detail_barang": {},
                "merk_barang": None,
                "harga_barang": price,
                "_page": page_num,
                "_original_text": original_keterangan
            }
        
        harga_barang = price if price is not None else None
        
        prompt = f"""Analyze this item description and extract product information.

DESCRIPTION: "{item_text}"
PRICE: {f'Rp {harga_barang:,.0f}' if harga_barang else 'Tidak ada nominal'}

TASK:
Extract the product information into this EXACT JSON structure:
{{
    "nama_barang": "Product name (be specific)",
    "jenis_tipe_barang": "Product type/category or null if unclear",
    "spesifikasi_detail_barang": {{"deskripsi": "additional details if any"}},
    "merk_barang": "Brand name or null if unclear",
    "harga_barang": {harga_barang if harga_barang is not None else 'null'}
}}

RULES:
1. nama_barang: Extract the main product/item name
2. jenis_tipe_barang: If not clear, use null
3. spesifikasi_detail_barang: Create an object with relevant specs
4. merk_barang: If mentioned, extract brand; otherwise use null
5. harga_barang: MUST BE SET TO {harga_barang if harga_barang is not None else 'null'}

IMPORTANT:
- Return ONLY valid JSON, no other text
- Keep all field names in Indonesian as shown
- The harga_barang field MUST be set to the price provided above

Now process this item:
Description: "{item_text}"
Price: {f'Rp {harga_barang:,.0f}' if harga_barang else 'Tidak ada nominal'}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.qwen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            result_text = response.choices[0].message.content.strip()
            
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = result_text[start_idx:end_idx]
                product = json.loads(json_str)
            else:
                product = json.loads(result_text)
            
            product["harga_barang"] = harga_barang
            product["_page"] = page_num
            product["_original_text"] = item_text
            
            return product
            
        except Exception as e:
            print(f"  ✗ LLM error: {str(e)}")
            
            return {
                "nama_barang": item_text[:100],
                "jenis_tipe_barang": None,
                "spesifikasi_detail_barang": {
                    "deskripsi": item_text
                },
                "merk_barang": None,
                "harga_barang": harga_barang,
                "_page": page_num,
                "_original_text": item_text
            }


class IntegratedCatalogExtractor:
    """Integrated extractor combining Excel extraction and LLM processing"""
    
    def __init__(self, api_key: str = None):
        self.excel_extractor = ExcelDataExtractor()
        self.llm_processor = LLMSemanticProcessor(api_key=api_key)
    
    def process_excel_file(self, excel_path: str, 
                          output_path: Optional[str] = None, 
                          max_items: Optional[int] = None) -> Dict[str, Any]:
        """
        Complete workflow: Extract from Excel and process with LLM
        """
        print(f"\n{'='*70}")
        print(f"INTEGRATED CATALOG EXTRACTOR - MULTI-SHEET VERSION")
        print(f"{'='*70}\n")
        
        try:
            # Step 1: Extract all data from Excel
            print("[PHASE 1] Extracting all data from Excel...")
            extracted_data = self.excel_extractor.extract_all_sheets(excel_path)
            
            transactions = extracted_data['transactions']
            notes = extracted_data['notes']
            
            print(f"\n  Total transactions extracted: {len(transactions)}")
            print(f"  Total notes: {len(notes)}")
            
            if len(transactions) == 0:
                print("\n[WARNING] No transactions found!")
                return {
                    "total_items": 0,
                    "stats": {
                        "total_items": 0,
                        "items_with_price": 0,
                        "matched_with_notes": 0,
                        "items_with_per_item_prices": 0,
                        "coverage_percent": 0
                    },
                    "failed_pages": [],
                    "items": []
                }
            
            if max_items and max_items < len(transactions):
                transactions = transactions[:max_items]
                print(f"  Limited to {max_items} transactions for processing")
            
            # Step 2: Process each transaction item with LLM
            print("\n[PHASE 2] Processing items with LLM...")
            items = []
            failed_pages = set()
            
            matched_count = 0
            per_item_price_count = 0
            
            for idx, transaction in enumerate(transactions):
                print(f"\n  Processing transaction {idx + 1}/{len(transactions)}:")
                print(f"    Date: {transaction['tanggal']}")
                print(f"    Total amount: {transaction.get('total_amount')}")
                print(f"    Match type: {transaction.get('match_type', 'none')}")
                print(f"    Has per-item prices: {transaction.get('has_per_item_prices', False)}")
                
                if transaction.get('has_per_item_prices') and transaction.get('matched_items'):
                    # Use matched items with per-item prices from notes
                    matched_count += 1
                    for matched_item in transaction['matched_items']:
                        item_text = matched_item.get('item', '')
                        item_price = matched_item.get('price')
                        
                        if item_price is not None:
                            per_item_price_count += 1
                        
                        print(f"    Item from notes: {item_text[:50]}...")
                        print(f"    Price from notes: Rp {item_price:,.0f}" if item_price else "    Price from notes: None")
                        
                        try:
                            product = self.llm_processor.extract_product_details_llm(
                                item_text,
                                item_price,
                                transaction['_page'],
                                transaction.get('raw_keterangan', '')
                            )
                            items.append(product)
                            print(f"      ✓ Success: {product.get('nama_barang', 'Unknown')}")
                            
                        except Exception as e:
                            print(f"      ✗ Failed: {str(e)}")
                            failed_pages.add(transaction['_page'])
                            items.append({
                                "nama_barang": item_text[:100],
                                "jenis_tipe_barang": None,
                                "spesifikasi_detail_barang": {
                                    "deskripsi": item_text
                                },
                                "merk_barang": None,
                                "harga_barang": item_price,
                                "_page": transaction['_page'],
                                "_original_text": item_text
                            })
                else:
                    # No matched notes with per-item prices, use original transaction
                    # Split the total amount equally among items as fallback
                    items_list = transaction.get('items', [])
                    total_amount = transaction.get('total_amount')
                    
                    if items_list and total_amount and len(items_list) > 1:
                        # Multiple items, split amount equally
                        per_item_amount = total_amount / len(items_list)
                        print(f"    Splitting Rp {total_amount:,.0f} equally among {len(items_list)} items")
                        
                        for item_text in items_list:
                            print(f"    Original item: {item_text[:50]}... (Rp {per_item_amount:,.0f})")
                            
                            try:
                                product = self.llm_processor.extract_product_details_llm(
                                    item_text,
                                    per_item_amount,
                                    transaction['_page']
                                )
                                items.append(product)
                                print(f"      ✓ Success: {product.get('nama_barang', 'Unknown')}")
                                
                            except Exception as e:
                                print(f"      ✗ Failed: {str(e)}")
                                failed_pages.add(transaction['_page'])
                                items.append({
                                    "nama_barang": item_text[:100],
                                    "jenis_tipe_barang": None,
                                    "spesifikasi_detail_barang": {
                                        "deskripsi": item_text
                                    },
                                    "merk_barang": None,
                                    "harga_barang": per_item_amount,
                                    "_page": transaction['_page'],
                                    "_original_text": item_text
                                })
                    else:
                        # Single item or no amount
                        for item_text in items_list:
                            print(f"    Original item: {item_text[:50]}...")
                            
                            try:
                                product = self.llm_processor.extract_product_details_llm(
                                    item_text,
                                    total_amount,
                                    transaction['_page']
                                )
                                items.append(product)
                                print(f"      ✓ Success: {product.get('nama_barang', 'Unknown')}")
                                
                            except Exception as e:
                                print(f"      ✗ Failed: {str(e)}")
                                failed_pages.add(transaction['_page'])
                                items.append({
                                    "nama_barang": item_text[:100],
                                    "jenis_tipe_barang": None,
                                    "spesifikasi_detail_barang": {
                                        "deskripsi": item_text
                                    },
                                    "merk_barang": None,
                                    "harga_barang": total_amount,
                                    "_page": transaction['_page'],
                                    "_original_text": item_text
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
                    "matched_with_notes": matched_count,
                    "items_with_per_item_prices": per_item_price_count,
                    "coverage_percent": (items_with_price / total_items * 100) if total_items > 0 else 0
                },
                "failed_pages": sorted(list(failed_pages)),
                "items": items,
                "notes_summary": {
                    "total_notes": len(notes),
                    "notes_by_date": list(notes.keys())
                }
            }
            
            # Step 4: Save result
            print("\n[PHASE 4] Saving results...")
            output_file = self.save_result(result, excel_path, output_path)
            
            print(f"\n{'='*70}")
            print(f"✓ PROCESSING COMPLETE")
            print(f"{'='*70}")
            print(f"  Total items: {result['total_items']}")
            print(f"  Items with price: {result['stats']['items_with_price']}")
            print(f"  Matched with notes: {result['stats']['matched_with_notes']}")
            print(f"  Items with per-item prices from notes: {result['stats']['items_with_per_item_prices']}")
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
    parser = argparse.ArgumentParser(
        description="Integrated Catalog Extractor - Multi-sheet Excel to JSON with LLM processing"
    )
    parser.add_argument(
        "excel_file",
        help="Path to Excel file containing cash fund usage data"
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
        help="Maximum number of transactions to process (for testing)",
        default=None
    )
    
    args = parser.parse_args()
    
    extractor = IntegratedCatalogExtractor(api_key=args.api_key)
    
    try:
        result = extractor.process_excel_file(
            args.excel_file, 
            args.output,
            args.max_items
        )
        
        if result["items"]:
            print(f"\n📋 Sample items with prices:")
            
            items_with_price = [item for item in result["items"] if item.get('harga_barang') is not None]
            
            print(f"\nItems WITH price ({len(items_with_price)}):")
            for i, item in enumerate(items_with_price[:10]):
                print(f"\n  {i+1}. {item.get('nama_barang')}")
                print(f"     Harga: Rp {item.get('harga_barang'):,.0f}")
                print(f"     Page: {item.get('_page')}")
                if item.get('_original_text'):
                    print(f"     Original: {item.get('_original_text')[:50]}...")
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