"""
items_specification_extraction.py — main orchestrator

Usage:
    python items_specification_extraction.py <path/to/document.pdf>
"""

import sys
import json
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

OUTPUT_DIR = f"output/{int(time())}"
RESULT_PATH = f"{OUTPUT_DIR}/result.json"
STAGE1_PAGES = 3  # number of pages sent to Stage 1


def main(pdf_path: str) -> None:
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

    # ------------------------------------------------------------------ #
    # 5. Auto-ingest into PostgreSQL vector store
    # ------------------------------------------------------------------ #
    print(f"\n[5/5] Ingesting into vector store…")
    abs_result_path = str(Path(RESULT_PATH).resolve())
    try:
        ingest(abs_result_path)
    except Exception as exc:
        print(f"WARNING: Ingestion failed (database may not be configured): {exc}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python items_specification_extraction.py <path/to/document.pdf>")
        sys.exit(1)

    main(sys.argv[1])
