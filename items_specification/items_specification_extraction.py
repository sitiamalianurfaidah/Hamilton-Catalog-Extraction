"""
items_specification_extraction.py — main orchestrator

Usage:
    python items_specification_extraction.py <path/to/document.pdf>
    python items_specification_extraction.py <path/to/spreadsheet.xlsx> 
"""

import sys
import json
import argparse
from pathlib import Path
from time import time

# Make the project root importable so search.* modules are accessible
# regardless of which directory the script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pdf_to_images import pdf_to_page_images
from stage1_metadata import extract_project_metadata
from stage2_page_extraction import extract_page_items
from merge_results import merge_all
from items_specification.ingest import ingest

try:
    from excel_parser import excel_to_result_json
    EXCEL_SUPPORT = True
except ImportError:
    EXCEL_SUPPORT = False
    print("Warning: Excel support not available. Install pandas and openpyxl for Excel support.")

OUTPUT_DIR = f"output/{int(time())}"
RESULT_PATH = f"{OUTPUT_DIR}/result.json"
STAGE1_PAGES = 3  # number of pages sent to Stage 1

def process_pdf(pdf_path: str, output_dir: str, result_path: str) -> dict:
    # ------------------------------------------------------------------ #
    # 1. Render PDF pages to base64 images
    # ------------------------------------------------------------------ #
    print(f"[1/4] Rendering pages from: {pdf_path}")
    pages = pdf_to_page_images(pdf_path)
    print(f"      {len(pages)} page(s) rendered.")

    # ------------------------------------------------------------------ #
    # 2. Stage 1 — extract project metadata + context_prompt
    # ------------------------------------------------------------------ #
    stage1_pages = pages[:STAGE1_PAGES]
    print(f"\n[2/4] Stage 1: extracting metadata from first {len(stage1_pages)} page(s)…")
    metadata = extract_project_metadata(stage1_pages)
    context_prompt = metadata.get("context_prompt", "")

    print("      Metadata:")
    for key in ("nama_proyek", "nama_site", "lokasi", "nama_bangunan", "jenis_pekerjaan"):
        print(f"        {key}: {metadata.get(key)}")

    # ------------------------------------------------------------------ #
    # 3. Stage 2 — extract items from every page
    # ------------------------------------------------------------------ #
    print(f"\n[3/4] Stage 2: extracting items from all {len(pages)} page(s)…")
    last_header1: str | None = None
    last_header2: str | None = None
    for page in pages:
        n = page["page_num"]
        print(f"      Processing page {n}/{len(pages)}…", end=" ", flush=True)
        result, last_header1, last_header2 = extract_page_items(
            page,
            context_prompt,
            output_dir=OUTPUT_DIR,
            last_header1=last_header1,
            last_header2=last_header2,
        )
        item_count = len(result.get("items", []))
        print(f"{item_count} item(s) found.  [header1={last_header1!r}, header2={last_header2!r}]")

    # ------------------------------------------------------------------ #
    # 4. Merge all per-page JSONs into a single result file
    # ------------------------------------------------------------------ #
    print(f"\n[4/4] Merging results…")
    final = merge_all(metadata, pages_dir=OUTPUT_DIR, out_path=RESULT_PATH)

    print(f"\nDone. Final output: {RESULT_PATH}")
    print(f"Total items extracted: {len(final.get('items', []))}")

def process_excel(excel_path: str, output_dir: str, result_path: str) -> dict:
        """Process Excel file directly to result.json."""
        print(f"[1/1] Processing Excel file: {excel_path}")
        
        if not EXCEL_SUPPORT:
            print("Error: Excel support not installed. Run: pip install pandas openpyxl")
            sys.exit(1)
        
        # Parse Excel to result.json format
        result_path = excel_to_result_json(excel_path, output_dir)
        
        # Load the result for return
        with open(result_path, 'r', encoding='utf-8') as f:
            final = json.load(f)
        
        print(f"\nDone. Excel parsed: {len(final.get('items', []))} items → {result_path}")
        return final


def main():
    parser = argparse.ArgumentParser(description='Extract construction items from PDF or Excel')
    parser.add_argument('input_file', help='Path to input file (PDF or Excel)')
    parser.add_argument('--excel', action='store_true', help='Force Excel mode (auto-detected by extension)')
    parser.add_argument('--no-ingest', action='store_true', help='Skip database ingestion')
    parser.add_argument('--output-dir', help='Custom output directory')
    
    args = parser.parse_args()
    
    # Determine output directory
    output_dir = args.output_dir or f"output/{int(time())}"
    result_path = f"{output_dir}/result.json"
    
    # Auto-detect file type
    file_ext = Path(args.input_file).suffix.lower()
    is_excel = args.excel or file_ext in ['.xlsx', '.xls', '.xlsm', '.csv']
    
    # Process based on file type
    if is_excel:
        final = process_excel(args.input_file, output_dir, result_path)
    else:
        final = process_pdf(args.input_file, output_dir, result_path)
    
    # 5: Ingest into vector store
    if not args.no_ingest:
        print(f"\n[5/5] Ingesting into vector store…")
        abs_result_path = str(Path(result_path).resolve())
        try:
            ingest(abs_result_path)
        except Exception as exc:
            print(f"WARNING: Ingestion failed: {exc}")
    
    print(f"\nComplete! Output: {result_path}")
    print(f"Total items: {len(final.get('items', []))}")

if __name__ == "__main__":
    main()
