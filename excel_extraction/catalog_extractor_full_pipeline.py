"""
Complete Pipeline: Excel Catalog Extractor + Embedding & Storage
Run this script to extract catalog from Excel, then embed and store in database _-
"""

import sys
import os
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import tempfile

# Import the catalog extractor
from catalog_extractor_excel import IntegratedCatalogExtractor


class CatalogPipeline:
    """Complete pipeline: Extract from Excel -> Prepare for embedding -> Embed & Store"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.extractor = IntegratedCatalogExtractor(api_key=api_key)
        
    def run(self, excel_path: str, output_dir: str = None, skip_embedding: bool = False):
        """
        Run complete pipeline
        """
        print("\n" + "="*80)
        print("COMPLETE CATALOG PIPELINE")
        print("="*80)
        
        # Step 1: Extract catalog from Excel
        print("\n[STEP 1] Extracting catalog from Excel...")
        
        if output_dir is None:
            output_dir = Path(excel_path).parent / "catalog_output"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_name = Path(excel_path).stem
        catalog_json = output_dir / f"{excel_name}_catalog_{timestamp}.json"
        embed_json = output_dir / f"{excel_name}_for_embedding_{timestamp}.json"
        
        # Run extraction
        result = self.extractor.process_all_sheets(
            excel_path,
            str(catalog_json)
        )
        
        print(f"\n[STEP 1 COMPLETE] Catalog saved to: {catalog_json}")
        print(f"  Total items: {result['total_items']}")
        print(f"  Items with price: {result['stats']['items_with_price']}")
        print(f"  Items from receipt: {result['stats']['items_from_receipt']}")
        
        # Step 2: Prepare for embedding
        print("\n[STEP 2] Preparing catalog for embedding...")
        embed_items = self.prepare_for_embedding(result['items'])
        
        # Save embedding-ready JSON
        with open(embed_json, 'w', encoding='utf-8') as f:
            json.dump(embed_items, f, indent=2, ensure_ascii=False)
        
        print(f"\n[STEP 2 COMPLETE] Embedding-ready catalog saved to: {embed_json}")
        print(f"  Total items for embedding: {len(embed_items)}")
        
        # Step 3: Embed and store (if not skipped)
        if not skip_embedding:
            print("\n[STEP 3] Embedding and storing in database...")
            self.catalog_embed_and_store(str(embed_json))
        else:
            print("\n[STEP 3] Skipped embedding (--skip-embedding flag used)")
            print(f"\nTo embed manually later, run:")
            print(f"  catalog_embed_and_store.py {embed_json}")
        
        # Print summary
        self.print_pipeline_summary(result, catalog_json, embed_json, skip_embedding)
        
        return {
            "extraction_result": result,
            "catalog_json": str(catalog_json),
            "embed_json": str(embed_json)
        }
    
    def prepare_for_embedding(self, items: list) -> list:
        """
        Prepare catalog items for embedding by reformatting to match expected structure
        """
        embed_items = []
        
        for item in items:
            # Create a clean copy without internal fields
            embed_item = {
                "nama_barang": item.get("nama_barang", ""),
                "jenis_tipe_barang": item.get("jenis_tipe_barang"),
                "spesifikasi_detail_barang": item.get("spesifikasi_detail_barang", {}),
                "merk_barang": item.get("merk_barang"),
                "harga_barang": item.get("harga_barang"),
                "_source": {
                    "page": item.get("_page"),
                    "date": item.get("_date"),
                    "matched_with_receipt": item.get("_matched_with_receipt", False)
                }
            }
            
            # Remove None values
            embed_item = {k: v for k, v in embed_item.items() if v is not None}
            
            embed_items.append(embed_item)
        
        return embed_items
    
    def embed_and_store(self, embed_json_path: str):
        """
        Run the catalog_embed_and_store.py script
        """
        try:
            # Get the directory of this script
            script_dir = Path(__file__).parent
            
            # Path to catalog_embed_and_store.py
            embed_script = script_dir / "catalog_embed_and_store.py"
            
            if not embed_script.exists():
                print(f"  ERROR: catalog_embed_and_store.py not found at {embed_script}")
                print("  Please make sure the script exists in the current directory")
                return False
            
            # Run the embedding script
            print(f"  Running: python {embed_script} {embed_json_path}")
            
            result = subprocess.run(
                [sys.executable, str(embed_script), embed_json_path],
                capture_output=True,
                text=True,
                cwd=script_dir
            )
            
            if result.returncode == 0:
                print("  ✓ Embedding and storage completed successfully")
                print(f"  Output: {result.stdout}")
                return True
            else:
                print("  ✗ Embedding failed with error:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"  ✗ Error running embed_and_store.py: {e}")
            return False
    
    def print_pipeline_summary(self, result: dict, catalog_json: Path, embed_json: Path, skip_embedding: bool):
        """Print pipeline summary"""
        print("\n" + "="*80)
        print("PIPELINE SUMMARY")
        print("="*80)
        
        stats = result['stats']
        
        print(f"\nExtraction Statistics:")
        print(f"  Total items extracted: {stats['total_items']}")
        print(f"  Items with price: {stats['items_with_price']} ({stats['coverage_percent']:.1f}%)")
        print(f"  Items from receipt OCR: {stats['items_from_receipt']} ({stats['receipt_match_rate']:.1f}%)")
        
        print(f"\nOutput Files:")
        print(f"  Raw catalog: {catalog_json}")
        print(f"  For embedding: {embed_json}")
        
        if not skip_embedding:
            print(f"\nDatabase Status:")
            print(f"  Items stored in: construction.items_catalog")
            print(f"  Source identifier: {embed_json.name}")
        else:
            print(f"\nDatabase Status:")
            print(f"  Not stored (embedding skipped)")
            print(f"  To store later: python pdf_extraction/embed_and_store.py {embed_json}")


def create_batch_script(output_dir: Path, excel_files: list):
    """Create a batch script to process multiple Excel files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_script = output_dir / f"batch_process_{timestamp}.bat"
    
    with open(batch_script, 'w') as f:
        f.write("@echo off\n")
        f.write("echo Starting batch processing...\n")
        f.write("echo.\n")
        
        for excel_file in excel_files:
            f.write(f'echo Processing: {excel_file}\n')
            f.write(f'python complete_pipeline.py "{excel_file}" --output-dir "{output_dir}"\n')
            f.write(f'if %errorlevel% neq 0 echo Error processing {excel_file}\n')
            f.write("echo.\n")
        
        f.write("echo Batch processing complete!\n")
    
    return batch_script


def main():
    parser = argparse.ArgumentParser(
        description="Complete Catalog Pipeline: Extract from Excel -> Embed -> Store in DB"
    )
    parser.add_argument(
        "excel_file",
        help="Path to Excel file containing cash fund usage data"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for JSON files (default: ./catalog_output)",
        default=None
    )
    parser.add_argument(
        "--api-key",
        help="DashScope API key for OCR and LLM",
        default=None
    )
    parser.add_argument(
        "--skip-embedding",
        help="Skip embedding and database storage (just extract to JSON)",
        action="store_true"
    )
    parser.add_argument(
        "--batch-mode",
        help="Create batch script instead of processing (provide list of files)",
        nargs="+",
        metavar="FILE"
    )
    
    args = parser.parse_args()
    
    # Handle batch mode
    if args.batch_mode:
        output_dir = Path(args.output_dir) if args.output_dir else Path("catalog_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        batch_script = create_batch_script(output_dir, args.batch_mode)
        print(f"\n✓ Batch script created: {batch_script}")
        print(f"  Run this script to process all {len(args.batch_mode)} files")
        return 0
    
    # Check if input file exists
    if not os.path.exists(args.excel_file):
        print(f"✗ Error: File {args.excel_file} not found")
        return 1
    
    # Run pipeline
    pipeline = CatalogPipeline(api_key=args.api_key)
    
    try:
        result = pipeline.run(
            args.excel_file,
            args.output_dir,
            args.skip_embedding
        )
        
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())