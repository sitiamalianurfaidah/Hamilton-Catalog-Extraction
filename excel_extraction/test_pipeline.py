"""
Script untuk testing pipeline lengkap:
1. Ekstrak dari Excel (Step 1)
2. Proses dengan LLM (Step 2)
3. Upload ke PostgreSQL dengan embedding
4. Test search
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from colorama import init, Fore, Style

init(autoreset=True)

def run_step1(excel_file):
    """Run catalog_extractor_step1.py"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("STEP 1: Excel Raw Data Extraction")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    result = subprocess.run([
        sys.executable, "catalog_extractor_excel.py", excel_file
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"{Fore.RED}{result.stderr}{Style.RESET_ALL}")
    
    # Find output file
    excel_path = Path(excel_file)
    catalog_json = excel_path.parent / f"{excel_path.stem}_extraction_result.json"
    
    return catalog_json if catalog_json.exists() else None

def run_step2(catalog_json):
    """Run catalog_extractor_step2.py"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("STEP 2: LLM Semantic Reasoning")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    result = subprocess.run([
        sys.executable, "catalog_extractor_step2.py", str(catalog_json)
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"{Fore.RED}{result.stderr}{Style.RESET_ALL}")
    
    # Find output file
    extraction_json = catalog_json.parent / f"{catalog_json.stem.replace('_extraction_result', '')}_extraction_result.json"
    
    return extraction_json if extraction_json.exists() else None

def run_postgres_upload(extraction_json):
    """Run catalog_to_postgres.py"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("STEP 3: Upload to PostgreSQL with Embedding")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    result = subprocess.run([
        sys.executable, "catalog_to_postgres.py", str(extraction_json)
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(f"{Fore.RED}{result.stderr}{Style.RESET_ALL}")
    
    return result.returncode == 0

def test_search():
    """Test search functionality"""
    print(f"\n{Fore.CYAN}{'='*60}")
    print("STEP 4: Testing Search")
    print(f"{'='*60}{Style.RESET_ALL}")
    
    # Test queries
    test_queries = [
        "bensin",
        "air mineral",
        "aqua",
        "kertas",
        "alat tulis"
    ]
    
    for query in test_queries:
        print(f"\n{Fore.YELLOW}Testing query: '{query}'{Style.RESET_ALL}")
        result = subprocess.run([
            sys.executable, "catalog_search_demo.py", 
            "--query", query,
            "--type", "semantic",
            "--limit", "3"
        ], capture_output=True, text=True)
        
        print(result.stdout)

def main():
    """Main test pipeline"""
    print(f"\n{Fore.GREEN}{'='*70}")
    print("TESTING COMPLETE CATALOG EXTRACTION PIPELINE")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    # Get Excel file from command line or use default
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    else:
        excel_file = input(f"{Fore.GREEN}Enter path to Excel file:{Style.RESET_ALL} ").strip()
    
    if not Path(excel_file).exists():
        print(f"{Fore.RED}Error: Excel file not found: {excel_file}{Style.RESET_ALL}")
        return 1
    
    # Run pipeline
    try:
        # Step 1
        catalog_json = run_step1(excel_file)
        if not catalog_json:
            print(f"{Fore.RED}Step 1 failed{Style.RESET_ALL}")
            return 1
        
        # Step 2
        extraction_json = run_step2(catalog_json)
        if not extraction_json:
            print(f"{Fore.RED}Step 2 failed{Style.RESET_ALL}")
            return 1
        
        # Step 3
        if not run_postgres_upload(extraction_json):
            print(f"{Fore.RED}Step 3 failed{Style.RESET_ALL}")
            return 1
        
        # Step 4
        test_search()
        
        print(f"\n{Fore.GREEN}{'='*70}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}{Style.RESET_ALL}")
        
        # Show next steps
        print(f"\n{Fore.CYAN}Next steps:{Style.RESET_ALL}")
        print("1. Run interactive search: python catalog_search_demo.py --interactive")
        print("2. Try different queries: python catalog_search_demo.py --query 'search term'")
        print("3. Check database: psql postgresql://postgres:hamiltonserver3.14@10.5.0.4:5432/postgres")
        
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())