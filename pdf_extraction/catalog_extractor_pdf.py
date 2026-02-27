"""
Catalog Extractor
Focus: Extract catalog items from multi-page PDF to structured JSON
"""

import os
import json
import re
import time
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import fitz
from langgraph.graph import StateGraph, END
from openai import OpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# CONFIG
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-7cb444645d0f49798ccba297720b4d8c")
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
MAX_WORKERS = 5  # parallel requests
OCR_DPI = 150    # 150 pixel per inch (lower -> faster)

# URL DATABASE HAMILTON 
DB_URI = "postgresql://postgres:hamiltonserver3.14@10.5.0.4:5432/postgres"

# STATE
class CatalogState(dict):
    """State for catalog extraction pipeline"""
    pdf_path: str
    pages_data: List[Dict[str, Any]]  # each page: {page_num, content, type}
    llm_outputs: List[Dict[str, Any]]
    normalized_items: List[Dict[str, Any]]
    failed_pages: List[int]
    extraction_stats: Dict[str, Any]


# HELPER FUNCTIONS
def sanitize_json_response(content: str) -> str:
    """Clean LLM response to get valid JSON"""
    # Remove everything before the first occurrence of '[' or '{'
    content = re.sub(r'^[^\[{]*', '', content, flags=re.DOTALL)
    # emove everything after the last occurrence of ']' or '}'
    content = re.sub(r'[^\]}]*$', '', content, flags=re.DOTALL)
    return content.strip()

def extract_json_array(text: str) -> Optional[List]:
    """Extract JSON array from response text"""
    try:
        # Try direct parsing
        return json.loads(text)
    except:
        # Find and extract anything inside square brackets [ ], including newlines
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                return json.loads(match.group())
            except:
                return None
    return None

def clean_price(value: Any) -> Optional[float]:
    """Clean price format (in Rupiah) to float"""
    if not value:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    
    # Remove everything except digits (dot, comma, Rp, space, etc)
    s = re.sub(r'[^\d]', '', str(value))
    
    try:
        return int(s)
    except:
        return None

def normalize_specification(spec: Any) -> Dict:
    """Normalize specification field to consistent format"""
    if not spec:
        return {}
    
    if isinstance(spec, str):
        # Try to parse as JSON
        try:
            spec = json.loads(spec)
        except:
            # Fallback to raw string
            return {"description": spec}
    
    if isinstance(spec, dict):
        # Standardize common keys
        standardized = {}
        key_mapping = {
            'ukuran': ['ukuran', 'size', 'dimension', 'dimensi', 'uk'],
            'bahan': ['bahan', 'material'],
            'warna': ['warna', 'color', 'colour'],
            'berat': ['berat', 'weight', 'bobot'],
            'volume': ['volume', 'vol', 'liter'],
            'panjang': ['panjang', 'length'],
            'lebar': ['lebar', 'width'],
            'tinggi': ['tinggi', 'height']
        }
        
        for key, value in spec.items():
            key_lower = key.lower()
            matched = False
            for std_key, variants in key_mapping.items():
                if key_lower in variants or any(v in key_lower for v in variants):
                    standardized[std_key] = value
                    matched = True
                    break
            if not matched:
                standardized[key] = value
        
        return standardized
    
    return {}


# PROCESSING FUNCTIONS
def process_single_page(page_data: Dict, client: OpenAI) -> List[Dict]:
    """Process a single page with Qwen"""
    
    page_num = page_data["page_num"]
    content = page_data["content"]
    content_type = page_data["type"]
    
    # Base system prompt
    system_prompt = """You are a construction catalog extraction system.

Extract ALL product items from the given text/image into a JSON array.

OUTPUT FORMAT - MUST BE EXACTLY LIKE THIS EXAMPLE:
[
    {
        "nama_barang": "Keramik 60x60",
        "jenis_tipe_barang": "Keramik Lantai",
        "spesifikasi_detail_barang": {
        "ukuran": "60x60",
        "polished": true,
        "bahan": "porselen"
        },
        "merk_barang": "Roman",
        "harga_barang": 125000
    },
    {
        "nama_barang": "Semen 50kg",
        "jenis_tipe_barang": "Semen",
        "spesifikasi_detail_barang": {
        "berat": "50kg",
        "tipe": "portland"
        },
        "merk_barang": "Tiga Roda",
        "harga_barang": 68000
    }
]

RULES:
1. nama_barang: required, product name
2. jenis_tipe_barang: optional, product type/category in Indonesian language
3. spesifikasi_detail_barang: object with relevant specs (size, weight, material, etc), all keys and values in Indonesian language (except 'True' and 'False')
4. merk_barang: optional, brand name
5. harga_barang: number only (no Rp, no dots, no commas), null if not available

Output ONLY valid JSON array without any additional text."""
    
    if content_type == "text":
        # Truncate if too long (Qwen context limit ~8k tokens)
        truncated_content = content[:3500] if len(content) > 3500 else content
        
        prompt = f"""Page {page_num} catalog text:

{truncated_content}

Extract all products into JSON array following the exact format from system prompt.

Output ONLY the JSON array."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = client.chat.completions.create(
                model="qwen-max",
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            clean_text = sanitize_json_response(response_text)
            items = extract_json_array(clean_text)
            
            if items and isinstance(items, list):
                # Add page metadata
                for item in items:
                    item['_page'] = page_num
                return items
            else:
                print(f"Page {page_num}: No valid JSON extracted")
                return []
                
        except Exception as e:
            print(f"Page {page_num} error: {e}")
            return []
            
    else:  # image mode
        prompt = f"""Extract all products from this catalog image (page {page_num}) into JSON array.

Extract all products into JSON array following the exact format from system prompt.

Output ONLY the JSON array."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{content}"}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        try:
            response = client.chat.completions.create(
                model="qwen-vl-max",
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            clean_text = sanitize_json_response(response_text)
            items = extract_json_array(clean_text)
            
            if items and isinstance(items, list):
                for item in items:
                    item['_page'] = page_num
                return items
            else:
                print(f"Page {page_num} (image): No valid JSON extracted")
                return []
                
        except Exception as e:
            print(f"Page {page_num} (image) error: {e}")
            return []


# LANGGRAPH NODES
def node_load_pdf(state: CatalogState) -> CatalogState:
    """Extract per page from PDF"""
    pdf_path = state.get("pdf_path")
    pages_data = []
    
    print(f"Loading PDF: {pdf_path}")
    
    start_time = time.time()
    try:
        doc = fitz.open(pdf_path)  # open pdf file
        print(f"Opened PDF with {len(doc)} pages in {time.time() - start_time:.2f}s")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            
            if text and len(text.strip()) > 50:
                # Text mode
                pages_data.append({
                    "page_num": page_num + 1,
                    "content": text.strip(),
                    "type": "text"
                })
                print(f"Page {page_num + 1}: Text mode ({len(text.strip())} chars)")
            else:
                # Convert page to image
                pix = page.get_pixmap(dpi=OCR_DPI)  # render the page into an image 
                img_bytes = pix.tobytes("png")      # convert pixmap to PNG format 
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                pages_data.append({
                    "page_num": page_num + 1,
                    "content": img_base64,
                    "type": "image"
                })
                print(f"Page {page_num + 1}: Image mode (OCR fallback)")
        
        doc.close()
        
    except Exception as e:
        print(f"Error reading PDF: {e}")
        state["pages_data"] = []
        return state
    
    state["pages_data"] = pages_data
    print(f"Loaded {len(pages_data)} pages total")
    return state

def node_extract_parallel(state: CatalogState) -> CatalogState:
    """Parallel extraction using ThreadPoolExecutor"""
    
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=QWEN_BASE_URL
    )
    
    all_items = []
    failed_pages = []
    pages_data = state.get("pages_data", [])
    total_pages = len(pages_data)
    
    print(f"Starting parallel extraction of {total_pages} pages...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_page = {
            executor.submit(process_single_page, page_data, client): page_data["page_num"]
            for page_data in pages_data
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_page):
            page_num = future_to_page[future]
            try:
                items = future.result(timeout=60)  # 60 second timeout per page
                if items:
                    all_items.extend(items)
                    print(f"Page {page_num}: Extracted {len(items)} items")
                else:
                    failed_pages.append(page_num)
                    print(f"Page {page_num}: No items extracted")
            except Exception as e:
                failed_pages.append(page_num)
                print(f"Page {page_num}: Failed - {e}")
    
    elapsed = time.time() - start_time
    print(f"Extraction completed in {elapsed:.2f}s")
    print(f"Total: {len(all_items)} items from {total_pages - len(failed_pages)}/{total_pages} pages")
    
    if failed_pages:
        print(f"Failed pages: {failed_pages}")
    
    state["llm_outputs"] = all_items
    state["failed_pages"] = failed_pages
    return state

def node_normalize(state: CatalogState) -> CatalogState:
    """Normalize extracted items to standard format"""
    
    normalized = []
    seen_items = set()  # For simple deduplication
    
    for item in state.get("llm_outputs", []):
        # Generate key for deduplication
        item_key = f"{item.get('nama_barang', '')}|{item.get('merk_barang', '')}|{json.dumps(item.get('spesifikasi_detail_barang', {}), sort_keys=True)}"
        
        if item_key in seen_items:
            continue
        
        seen_items.add(item_key)
        
        # Normalize specification
        spec = normalize_specification(item.get("spesifikasi_detail_barang"))
        
        normalized_item = {
            "nama_barang": str(item.get("nama_barang", "")).strip() or None,
            "jenis_tipe_barang": str(item.get("jenis_tipe_barang", "")).strip() or None,
            "spesifikasi_detail_barang": spec,
            "merk_barang": str(item.get("merk_barang", "")).strip() or None,
            "harga_barang": clean_price(item.get("harga_barang")),
            "_page": item.get("_page")
        }
        
        # Only include items with at least a name
        if normalized_item["nama_barang"]:
            normalized.append(normalized_item)
    
    # Calculate stats
    items_with_price = sum(1 for i in normalized if i['harga_barang'] is not None)
    
    state["normalized_items"] = normalized
    state["extraction_stats"] = {
        "total_items": len(normalized),
        "items_with_price": items_with_price,
        "coverage_percent": (items_with_price / len(normalized) * 100) if normalized else 0
    }
    
    print(f"Normalized {len(normalized)} unique items")
    print(f"Items with price: {items_with_price}/{len(normalized)} ({state['extraction_stats']['coverage_percent']:.1f}%)")
    
    return state

def node_retry_failed(state: CatalogState) -> CatalogState:
    """Retry failed pages with different parameters"""
    failed_pages = state.get("failed_pages", [])
    if not failed_pages:
        return state
    
    print(f"Retrying {len(failed_pages)} failed pages...")
    
    # Filter pages_data for failed pages
    failed_data = [p for p in state.get("pages_data", []) if p["page_num"] in failed_pages]
    
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=QWEN_BASE_URL
    )
    
    retry_items = []
    still_failed = []
    
    for page_data in failed_data:
        print(f"Retrying page {page_data['page_num']}...")
        
        # Simpler prompt for retry
        if page_data["type"] == "text":
            simple_prompt = f"""List all products on page {page_data['page_num']} as JSON array.
Each item: name (nama_barang), type (jenis_tipe_barang), specs (spesifikasi_detail_barang as object), brand (merk_barang), price (harga_barang as number).

Content:
{page_data['content'][:2000]}"""
            
            messages = [
                {"role": "system", "content": "Extract products to JSON array. Only output valid JSON."},
                {"role": "user", "content": simple_prompt}
            ]
        else:
            # Image mode
            simple_prompt = f"Extract all products from this catalog image (page {page_data['page_num']}) to JSON array."
            messages = [
                {"role": "system", "content": "Extract products to JSON array."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_data['content']}"}},
                        {"type": "text", "text": simple_prompt}
                    ]
                }
            ]
        
        try:
            response = client.chat.completions.create(
                model="qwen-vl-max" if page_data["type"] == "image" else "qwen-vl-plus",
                messages=messages,
                temperature=0.3,  # Slightly more creative for retry
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            clean_text = sanitize_json_response(response_text)
            items = extract_json_array(clean_text)
            
            if items and isinstance(items, list):
                for item in items:
                    item['_page'] = page_data['page_num']
                retry_items.extend(items)
                print(f"Page {page_data['page_num']}: Recovered {len(items)} items on retry")
            else:
                still_failed.append(page_data['page_num'])
                print(f"Page {page_data['page_num']}: Still failed on retry")
                
        except Exception as e:
            still_failed.append(page_data['page_num'])
            print(f"Page {page_data['page_num']} retry error: {e}")
    
    state["llm_outputs"].extend(retry_items)  # Add successful retry items to existing outputs
    state["failed_pages"] = still_failed      # Overwrite updated data
    
    print(f"Retry complete: Recovered {len(retry_items)} items")
    if still_failed:
        print(f"Still failed: {still_failed}")
    
    return state

def node_save_results(state: CatalogState) -> CatalogState:
    """Save results to JSON file"""
    items = state.get("normalized_items", [])
    
    # Save to file
    with open("catalog_extraction_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_items": len(items),
            "stats": state.get("extraction_stats", {}),
            "failed_pages": state.get("failed_pages", []),
            "items": items
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to catalog_extraction_results.json")
    
    # Also save separate file for embedding team
    with open("catalog_for_embedding.json", "w", encoding="utf-8") as f:
        # Remove metadata fields for embedding
        clean_items = []
        for item in items:
            clean_item = {k: v for k, v in item.items() if not k.startswith('_')}
            clean_items.append(clean_item)
        json.dump(clean_items, f, ensure_ascii=False, indent=2)
    
    print(f"Embedding-ready data saved to catalog_for_embedding.json")
    
    return state


# WORKFLOW
def build_catalog_extractor():
    """Build the LangGraph workflow"""
    
    graph = StateGraph(CatalogState)
    
    # Add nodes
    graph.add_node("load_pdf", node_load_pdf)
    graph.add_node("extract_parallel", node_extract_parallel)
    graph.add_node("normalize", node_normalize)
    graph.add_node("retry_failed", node_retry_failed)
    graph.add_node("save_results", node_save_results)
    
    # Set entry point
    graph.set_entry_point("load_pdf")
    
    # Define edges
    graph.add_edge("load_pdf", "extract_parallel")
    graph.add_edge("extract_parallel", "normalize")

    graph.add_conditional_edges(    # Retry if there are failed pages
        "normalize",
        lambda state: "retry" if state.get("failed_pages") else "save",
        {
            "retry": "retry_failed",
            "save": "save_results"
        }
    )
    
    graph.add_edge("retry_failed", "save_results")
    graph.add_edge("save_results", END)
    
    # Create PostgreSQL checkpointer instance
    pool = ConnectionPool(conninfo=DB_URI)
    checkpointer = PostgresSaver(pool)
    
    checkpointer.setup() 
    
    return graph.compile(checkpointer=checkpointer)


# MAIN
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract catalog items from PDF")
    parser.add_argument("pdf_path", help="Path to PDF catalog file")
    parser.add_argument("--thread-id", default="extraction_1", help="Thread ID for checkpointing")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: File {args.pdf_path} not found")
        exit(1)
    
    print("=" * 60)
    print("CATALOG EXTRACTOR (PDF)")
    print("=" * 60)
    
    # Build and run workflow
    workflow = build_catalog_extractor()
    
    try:
        result = workflow.invoke(
            {"pdf_path": args.pdf_path},
            config={"configurable": {"thread_id": args.thread_id}}
        )
        
        print("=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"RAW items from LLM: {len(result.get('llm_outputs', []))}")
        print(f"Normalized items: {len(result.get('normalized_items', []))}")
        print(f"Failed pages: {result.get('failed_pages', [])}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Workflow failed: {e}")
        exit(1)