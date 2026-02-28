"""
Tahap 2: LLM Semantic Reasoning for Product Extraction
Takes raw catalog JSON and applies LLM to extract product details:
- Nama barang (product name)
- Jenis/tipe barang (product type/category)
- Spesifikasi detail (specifications object)
- Merk barang (brand)
- Harga barang (price)

Output: catalog_excel_extraction_result.json with structured product objects
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Any
from openai import OpenAI

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


class CatalogExtractorStep2:
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
    
    def load_catalog_json(self, json_path: str) -> Dict[str, Any]:
        """
        Load raw catalog JSON from Stage 1
        
        Args:
            json_path: Path to catalog JSON file
            
        Returns:
            Dictionary with catalog data
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Loaded catalog JSON: {json_path}")
        print(f"  Total blocks: {len(data.get('blocks', []))}")
        
        return data
    
    def extract_product_details_llm(self, keterangan_text: str, operasional_amount: float) -> List[Dict[str, Any]]:
        """
        Use LLM to parse keterangan text and extract product details
        
        Args:
            keterangan_text: Description text from Excel
            operasional_amount: Amount spent in Rupiah
            
        Returns:
            List of product objects with extracted details
        """
        prompt = f"""Analyze this transaction description text and extract product information.

TRANSACTION DESCRIPTION:
"{keterangan_text}"

AMOUNT: Rp {operasional_amount:,.0f}

TASK:
Extract ALL products mentioned in the description. For each product, identify:
1. nama_barang: Product name/item name
2. jenis_tipe: Product type/category (e.g., "Bahan Bakar", "Material", "Service", "Peralatan")
3. spesifikasi: Object with detailed specs (create relevant spec fields like ukuran, warna, jumlah, dll)
4. merk: Brand name (if mentioned, otherwise "Tidak Ada")
5. harga: Price per unit in Rupiah (estimate if total given, or null if unclear)

RESPONSE FORMAT:
If ONE product: Return single object
If MULTIPLE products: Return array of objects

Example for single product:
{{
    "nama_barang": "BBM Pertamax",
    "jenis_tipe_barang": "Bahan Bakar Motor",
    "spesifikasi_detail_barang": {{
        "jenis": "Pertamax",
        "peruntukan": "Kendaraan"
    }},
    "merk_barang": "Shell",
    "harga_barang": 50000
}}

Example for multiple products:
[
    {{
        "nama_barang": "Galon Air Minum",
        "jenis_tipe_barang": "Minuman",
        "spesifikasi_detail_barang": {{
            "ukuran": "19L",
            "temperatur": "Dingin"
        }},
        "merk_barang": "Aqua/Local",
        "harga_barang": 25000
    }},
    {{
        "nama_barang": "Biaya Jasa Tarik Tunai",
        "jenis_tipe": "Service",
        "spesifikasi": {{
            "layanan": "ATM Withdrawal Fee",
            "bank": "Umum"
        }},
        "merk_barang": "Bank",
        "harga_barang": 5000
    }}
]

Return ONLY valid JSON (object or array), NO additional text."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.qwen_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3  # Slightly higher for semantic understanding
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if result_text.startswith('['):
                products = json.loads(result_text)
            else:
                products = [json.loads(result_text)]
            
            # Ensure it's always a list
            if not isinstance(products, list):
                products = [products]
            
            return products
            
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parse error: {str(e)}")
            # Fallback: create generic product entry
            return [{
                "nama_barang": keterangan_text[:50],
                "jenis_tipe": "Pengeluaran Operasional",
                "spesifikasi": {
                    "deskripsi": keterangan_text
                },
                "merk_barang": "Tidak Ada",
                "harga_barang": operasional_amount
            }]
        
        except Exception as e:
            print(f"  ✗ LLM error: {str(e)}")
            return [{
                "nama_barang": keterangan_text[:50],
                "jenis_tipe_barang": "Pengeluaran Operasional",
                "spesifikasi_detail_barang": {
                    "deskripsi": keterangan_text,
                    "error": str(e)
                },
                "merk_barang": "Tidak Ada",
                "harga_barang": operasional_amount
            }]
    
    def process_blocks(self, catalog_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process all blocks from catalog JSON through LLM
        Extract product details for each keterangan entry
        
        Args:
            catalog_data: Raw catalog data from stage 1
            
        Returns:
            Processed data with extracted product details
        """
        processed_result = {
            "metadata": {
                "file_path": catalog_data.get("file_path"),
                "sheet_name": catalog_data.get("sheet_name"),
                "stage": "extraction_result",
                "total_blocks": len(catalog_data.get("blocks", [])),
                "total_transactions": 0,
                "total_products": 0
            },
            "blocks": []
        }
        
        print(f"\n[Step 1] Processing blocks with LLM...")
        
        # Process each block
        for block_idx, block in enumerate(catalog_data.get("blocks", [])):
            processed_block = {
                "tanggal": block.get("tanggal"),
                "dana_masuk": block.get("dana_masuk"),
                "sisa_saldo_sebelum": block.get("sisa_saldo_sebelum"),
                "saldo_saat_ini": block.get("saldo_saat_ini"),
                "total_operasional": sum(block.get("operasional", [])),
                "items": []
            }
            
            # Process each transaction in block
            operasional_list = block.get("operasional", [])
            keterangan_list = block.get("keterangan", [])
            
            # Align lists (in case they have different lengths)
            num_items = max(len(operasional_list), len(keterangan_list))
            
            for item_idx in range(num_items):
                keterangan = keterangan_list[item_idx] if item_idx < len(keterangan_list) else "Tidak ada keterangan"
                operasional = operasional_list[item_idx] if item_idx < len(operasional_list) else 0
                
                print(f"  Block {block_idx + 1}/{len(catalog_data.get('blocks', []))}, Item {item_idx + 1}: {keterangan[:40]}...")
                
                # Extract product details using LLM
                products = self.extract_product_details_llm(keterangan, operasional)
                
                # Add metadata to each product
                for product in products:
                    product["raw_keterangan"] = keterangan
                    product["raw_operasional"] = operasional
                    processed_block["items"].append(product)
                    processed_result["metadata"]["total_products"] += 1
            
            processed_block["num_items"] = len(processed_block["items"])
            processed_result["blocks"].append(processed_block)
            processed_result["metadata"]["total_transactions"] += num_items
        
        return processed_result
    
    def save_result(self, result: Dict[str, Any], input_path: str, output_path: Optional[str] = None) -> str:
        """
        Save processed result to JSON file
        
        Args:
            result: Processed result dictionary
            input_path: Path to input catalog JSON (for reference)
            output_path: Optional custom output path
            
        Returns:
            Path where file was saved
        """
        if output_path is None:
            input_file = Path(input_path)
            # Replace _catalog.json with _extraction_result.json
            output_path = input_file.parent / f"{input_file.stem.replace('_catalog', '')}_extraction_result.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Result saved to: {output_path}")
        return str(output_path)
    
    def process_catalog_file(self, catalog_json_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main workflow: Load catalog JSON and process through LLM
        
        Args:
            catalog_json_path: Path to catalog JSON from stage 1
            output_path: Optional custom output path
            
        Returns:
            Processed result dictionary
        """
        print(f"\n{'='*70}")
        print(f"CATALOG EXTRACTOR STEP 2 - LLM Semantic Reasoning")
        print(f"{'='*70}\n")
        
        try:
            # Step 1: Load catalog JSON
            print("[Step 1] Loading catalog JSON from Stage 1...")
            catalog_data = self.load_catalog_json(catalog_json_path)
            
            # Step 2: Process with LLM
            print("\n[Step 2] Processing with LLM semantic reasoning...")
            result = self.process_blocks(catalog_data)
            
            # Step 3: Save result
            print("\n[Step 3] Saving results...")
            output_file = self.save_result(result, catalog_json_path, output_path)
            
            print(f"\n{'='*70}")
            print(f"✓ Processing Complete")
            print(f"{'='*70}")
            print(f"  Total blocks: {result['metadata']['total_blocks']}")
            print(f"  Total transactions: {result['metadata']['total_transactions']}")
            print(f"  Total products extracted: {result['metadata']['total_products']}")
            
            return result
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Catalog Extractor Step 2 - LLM Semantic Reasoning"
    )
    parser.add_argument(
        "catalog_json",
        help="Path to catalog JSON file from Step 1 (e.g., *_catalog.json)"
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
    
    args = parser.parse_args()
    
    # Initialize extractor stage 2
    extractor = CatalogExtractorStep2(api_key=args.api_key)
    
    # Process catalog
    try:
        result = extractor.process_catalog_file(args.catalog_json, args.output)
        
        # Print sample
        if result["blocks"]:
            print(f"\n📋 Sample block (first date):")
            sample_block = result["blocks"][0]
            print(f"  Tanggal: {sample_block.get('tanggal')}")
            print(f"  Total items: {sample_block.get('num_items')}")
            if sample_block.get("items"):
                sample_item = sample_block["items"][0]
                print(f"  First product: {sample_item.get('nama_barang')}")
                print(f"  Type: {sample_item.get('jenis_tipe')}")
                print(f"  Price: {sample_item.get('harga')}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
