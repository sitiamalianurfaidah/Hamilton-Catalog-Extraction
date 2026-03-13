"""
Tahap 1: Excel Raw Data Extractor
Extracts all raw data from Excel file based on headers:
- Tanggal
- Dana Masuk
- Operasional (as multiple columns)
- Keterangan (as multiple columns)
- Sisa Saldo Sebelum
- Saldo Saat Ini

Output: catalog_excel_extraction_result.json with raw structured data
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Any
import re


class CatalogExtractorStep1:
    """Excel Raw Data Extractor for Cash Fund Usage Reports"""
    
    def __init__(self):
        """Initialize extractor"""
        pass
    
    def detect_header_row(self, df: pd.DataFrame) -> int:
        """
        Detect which row contains the headers by looking for key column names
        
        Args:
            df: Raw DataFrame from Excel
            
        Returns:
            Index of header row
        """
        # Keywords to look for in headers
        header_keywords = ['tanggal', 'dana masuk', 'operasional', 'keterangan', 'sisa saldo', 'saldo']
        
        for idx, row in df.iterrows():
            row_str = ' '.join([str(val).lower() for val in row.values if pd.notna(val)])
            # Check if multiple keywords found in this row
            matches = sum(1 for keyword in header_keywords if keyword in row_str)
            if matches >= 3:  # Found at least 3 header keywords
                return idx
        
        # If not found, assume first row is header
        return 0
    
    def extract_column_groups(self, headers: List[str]) -> Dict[str, List[int]]:
        """
        Identify groups of Operasional and Keterangan columns
        
        Args:
            headers: List of column headers
            
        Returns:
            Dictionary with column indices for each group
        """
        groups = {
            'tanggal': [],
            'dana_masuk': [],
            'operasional': [],
            'keterangan': [],
            'sisa_saldo_sebelum': [],
            'saldo_saat_ini': []
        }
        
        for idx, header in enumerate(headers):
            header_lower = str(header).lower().strip()
            
            if 'tanggal' in header_lower:
                groups['tanggal'].append(idx)
            elif 'dana masuk' in header_lower or 'dana_masuk' in header_lower:
                groups['dana_masuk'].append(idx)
            elif 'operasional' in header_lower:
                groups['operasional'].append(idx)
            elif 'keterangan' in header_lower:
                groups['keterangan'].append(idx)
            elif 'sisa saldo sebelum' in header_lower or 'sisa_saldo_sebelum' in header_lower:
                groups['sisa_saldo_sebelum'].append(idx)
            elif 'saldo saat ini' in header_lower or 'saldo_saat_ini' in header_lower:
                groups['saldo_saat_ini'].append(idx)
        
        return groups
    
    def clean_amount(self, value: Any) -> Optional[float]:
        """
        Clean and convert amount values to float
        
        Args:
            value: Raw value from Excel
            
        Returns:
            Cleaned float value or None
        """
        if pd.isna(value):
            return None
        
        # If it's already a number
        if isinstance(value, (int, float)):
            return float(value)
        
        # If it's a string, clean it
        if isinstance(value, str):
            # Remove currency symbols, dots (thousand separators), and spaces
            cleaned = re.sub(r'[Rp\s.]', '', value)
            # Replace comma with dot for decimal
            cleaned = cleaned.replace(',', '.')
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
    def clean_text(self, value: Any) -> str:
        """
        Clean text values
        
        Args:
            value: Raw value from Excel
            
        Returns:
            Cleaned string
        """
        if pd.isna(value):
            return ""
        
        if isinstance(value, (int, float)):
            return str(int(value)) if value.is_integer() else str(value)
        
        return str(value).strip()
    
    def parse_date(self, value: Any) -> Optional[str]:
        """
        Parse date to consistent string format
        
        Args:
            value: Raw date value from Excel
            
        Returns:
            Formatted date string or None
        """
        if pd.isna(value):
            return None
        
        # If it's already a datetime
        if isinstance(value, pd.Timestamp):
            return value.strftime('%Y-%m-%d %H:%M:%S')
        
        # If it's a string, try to parse
        if isinstance(value, str):
            # Try common formats
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%Y/%m/%d']:
                try:
                    dt = pd.to_datetime(value, format=fmt)
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    continue
            
            # Let pandas try to infer
            try:
                dt = pd.to_datetime(value)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                return value  # Return as-is if can't parse
        
        return str(value)
    
    def process_excel(self, excel_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Main method to process Excel file and extract raw data
        
        Args:
            excel_path: Path to Excel file
            sheet_name: Optional sheet name (if None, uses first sheet)
            
        Returns:
            Dictionary with extracted raw data
        """
        print(f"\n{'='*70}")
        print(f"CATALOG EXTRACTOR STEP 1 - Excel Raw Data Extraction")
        print(f"{'='*70}\n")
        
        # Load Excel file
        print(f"[Step 1] Loading Excel file: {excel_path}")
        excel_file = pd.ExcelFile(excel_path)
        
        # Get sheet name
        if sheet_name is None:
            sheet_name = excel_file.sheet_names[0]
            print(f"  Using first sheet: {sheet_name}")
        else:
            print(f"  Using sheet: {sheet_name}")
        
        # Read the sheet
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        print(f"  Raw data shape: {df.shape}")
        
        # Detect header row
        print(f"\n[Step 2] Detecting header row...")
        header_row = self.detect_header_row(df)
        print(f"  Header found at row: {header_row + 1}")
        
        # Set headers
        headers = df.iloc[header_row].tolist()
        print(f"  Headers: {headers}")
        
        # Extract column groups
        print(f"\n[Step 3] Identifying column groups...")
        column_groups = self.extract_column_groups(headers)
        
        for group_name, indices in column_groups.items():
            if indices:
                print(f"  {group_name}: columns {[i+1 for i in indices]}")
        
        # Extract data rows (after header)
        data_rows = df.iloc[header_row + 1:].reset_index(drop=True)
        
        # Process each row into blocks
        print(f"\n[Step 4] Processing data rows...")
        blocks = []
        current_block = None
        
        for idx, row in data_rows.iterrows():
            # Get tanggal (date) - primary key for new block
            tanggal = None
            if column_groups['tanggal']:
                col_idx = column_groups['tanggal'][0]  # Use first tanggal column
                tanggal = self.parse_date(row[col_idx])
            
            # Get dana_masuk
            dana_masuk = None
            if column_groups['dana_masuk']:
                col_idx = column_groups['dana_masuk'][0]
                dana_masuk = self.clean_amount(row[col_idx])
            
            # Get sisa_saldo_sebelum
            sisa_saldo_sebelum = None
            if column_groups['sisa_saldo_sebelum']:
                col_idx = column_groups['sisa_saldo_sebelum'][0]
                sisa_saldo_sebelum = self.clean_amount(row[col_idx])
            
            # Get saldo_saat_ini
            saldo_saat_ini = None
            if column_groups['saldo_saat_ini']:
                col_idx = column_groups['saldo_saat_ini'][0]
                saldo_saat_ini = self.clean_amount(row[col_idx])
            
            # If we have a new date or this is first row, start new block
            if tanggal or current_block is None:
                # Save previous block if exists
                if current_block:
                    blocks.append(current_block)
                
                # Start new block
                current_block = {
                    "tanggal": tanggal,
                    "dana_masuk": dana_masuk,
                    "operasional": [],
                    "keterangan": [],
                    "sisa_saldo_sebelum": sisa_saldo_sebelum,
                    "saldo_saat_ini": saldo_saat_ini,
                    "raw_entries": []
                }
            
            # Extract operasional values from all operasional columns
            operasional_values = []
            for col_idx in column_groups['operasional']:
                val = self.clean_amount(row[col_idx])
                if val is not None and val != 0:
                    operasional_values.append(val)
            
            # Extract keterangan values from all keterangan columns
            keterangan_values = []
            for col_idx in column_groups['keterangan']:
                val = self.clean_text(row[col_idx])
                if val and val.strip():
                    keterangan_values.append(val)
            
            # Add to current block (only if there's data)
            if operasional_values or keterangan_values:
                # Add to arrays
                current_block["operasional"].extend(operasional_values)
                current_block["keterangan"].extend(keterangan_values)
                
                # Add raw entry for traceability
                current_block["raw_entries"].append({
                    "operasional": operasional_values if len(operasional_values) > 1 else (operasional_values[0] if operasional_values else None),
                    "keterangan": keterangan_values if len(keterangan_values) > 1 else (keterangan_values[0] if keterangan_values else "")
                })
        
        # Add last block
        if current_block:
            blocks.append(current_block)
        
        print(f"  Found {len(blocks)} date blocks")
        
        # Compile result
        print(f"\n[Step 5] Compiling results...")
        result = {
            "status": "success",
            "file_path": str(excel_path),
            "sheet_name": sheet_name,
            "blocks": blocks,
            "summary": {
                "total_blocks": len(blocks),
                "total_transactions": sum(len(block.get("operasional", [])) for block in blocks),
                "date_range": {
                    "first_date": blocks[0].get("tanggal") if blocks else None,
                    "last_date": blocks[-1].get("tanggal") if blocks else None
                }
            },
            "errors": []
        }
        
        print(f"\n{'='*70}")
        print(f"✓ Extraction Complete")
        print(f"{'='*70}")
        print(f"  Total blocks: {result['summary']['total_blocks']}")
        print(f"  Total transactions: {result['summary']['total_transactions']}")
        print(f"  Date range: {result['summary']['date_range']['first_date']} to {result['summary']['date_range']['last_date']}")
        
        return result
    
    def save_result(self, result: Dict[str, Any], excel_path: str, output_path: Optional[str] = None) -> str:
        """
        Save extracted result to JSON file
        
        Args:
            result: Extracted result dictionary
            excel_path: Path to input Excel file (for reference)
            output_path: Optional custom output path
            
        Returns:
            Path where file was saved
        """
        if output_path is None:
            excel_file = Path(excel_path)
            # Create output filename: original name + _extraction_result.json
            output_path = excel_file.parent / f"{excel_file.stem}_extraction_result.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Result saved to: {output_path}")
        return str(output_path)
    
    def process_file(self, excel_path: str, sheet_name: Optional[str] = None, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main workflow: Process Excel file and save results
        
        Args:
            excel_path: Path to Excel file
            sheet_name: Optional sheet name
            output_path: Optional custom output path
            
        Returns:
            Extracted result dictionary
        """
        try:
            # Step 1-4: Process Excel
            result = self.process_excel(excel_path, sheet_name)
            
            # Step 5: Save result
            self.save_result(result, excel_path, output_path)
            
            return result
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Catalog Extractor Step 1 - Excel Raw Data Extraction"
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
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = CatalogExtractorStep1()
    
    # Process file
    try:
        result = extractor.process_file(args.excel_file, args.sheet, args.output)
        
        # Print sample
        if result["blocks"]:
            print(f"\n📋 Sample block (first date):")
            sample = result["blocks"][0]
            print(f"  Tanggal: {sample.get('tanggal')}")
            print(f"  Dana Masuk: {sample.get('dana_masuk')}")
            print(f"  Operasional entries: {len(sample.get('operasional', []))}")
            print(f"  Keterangan entries: {len(sample.get('keterangan', []))}")
            if sample.get("operasional"):
                print(f"  First amount: Rp {sample['operasional'][0]:,.0f}")
            if sample.get("keterangan"):
                print(f"  First description: {sample['keterangan'][0][:50]}...")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())