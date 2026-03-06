## Plan: PDF Construction Items Extraction Pipeline

**Overview:** Convert a scanned PDF into images page-by-page using PyMuPDF, then run a two-stage qwen-vl-max (vision) pipeline: Stage 1 scans the first few pages to extract project metadata AND generates a tailored "context prompt" for Stage 2. Stage 2 processes every remaining page individually using that context prompt, saving a JSON per page. A final merge step assembles everything into one output JSON.

---

**Steps**

1. **Update requirements.txt** — add `PyMuPDF` (renders PDF pages to images) and `Pillow` (optional for image resizing before sending).

2. **Create `pdf_to_images.py`** — a helper module with a single function `pdf_to_page_images(pdf_path) -> list[dict]` that uses `fitz.open()` to render each page to a PNG at a reasonable DPI (~150), encodes it as base64, and returns a list of `{page_num, base64_image}` dicts. This is the basis for all LLM calls.

3. **Create `stage1_metadata.py`** — function `extract_project_metadata(page_images: list[dict]) -> dict`. Sends the first 3 page images (or fewer if short PDF) to `qwen-vl-max` via a single multimodal chat call. The system prompt asks the model to:
   - Extract: `nama_proyek`, `nama_site`, `lokasi`, `nama_bangunan`, `jenis_pekerjaan[]`
   - **Also output** a `context_prompt` string — a concise description of what construction item tables look like in _this specific document_ (column names, language, typical columns such as No., Uraian/Nama Barang, Spesifikasi, Merek, etc.) so subsequent pages get targeted guidance.
   - Return JSON. Response is parsed and the `context_prompt` key is extracted for Stage 2.

4. **Create `stage2_page_extraction.py`** — function `extract_page_items(page: dict, context_prompt: str) -> dict`. Sends a single page image to `qwen-vl-max`. The user message prepends the `context_prompt` from Stage 1 plus a strict instruction to extract items into a JSON array:

   ```
   [
     {
       "nama_barang": "...",
       "spesifikasi_barang": {
         "spesifikasi_general": "...",
         "spesifikasi_merek": "..."
       }
     }
   ]
   ```

   If no items are found, return `{"items": []}`. Saves result to `output/page_{n}.json`.

5. **Create `merge_results.py`** — function `merge_all(metadata: dict, pages_dir: str) -> dict`. Reads all `page_*.json` files, flattens the `items[]` arrays across all pages, attaches the project metadata fields as top-level keys, and writes `output/result.json`.

6. **Rewrite items_specification_extraction.py** as the main orchestrator:
   - Accept a PDF path (CLI arg or hardcoded for now)
   - Call `pdf_to_page_images()` → get `pages`
   - Call `extract_project_metadata(pages[:3])` → get `metadata` + `context_prompt`
   - Loop over `pages` (all pages, the model will skip if no table found), calling `extract_page_items(page, context_prompt)` for each — sequentially to stay within rate limits
   - Call `merge_all()` → write final `output/result.json`
   - Print progress per page

---

**File layout after implementation**

```
items_specification_extraction.py   ← orchestrator (main entry point)
pdf_to_images.py
stage1_metadata.py
stage2_page_extraction.py
merge_results.py
output/
  page_1.json
  page_2.json
  ...
  result.json
```

---

**Verification**

- Run with a sample scanned PDF: `python items_specification_extraction.py sample.pdf`
- Check `output/result.json` for correct structure and that `nama_proyek`, `items[]` are populated
- Inspect a few `page_*.json` to confirm per-page extraction quality
- Test with a page that has no table (e.g. a cover page) — should produce `{"items": []}`

---

**Decisions**

- Using `qwen-vl-max` (vision) over `qwen-max` (text) because the PDF is scanned — no selectable text
- First 3 pages sent to Stage 1 (not just 1) to hedge against cover-only pages having no metadata
- Per-page JSON files are kept before merging to allow resuming without re-calling the API on already-processed pages (add a skip-if-exists check in the loop)
- Sequential page processing to avoid DashScope rate limits; can be made concurrent later with `asyncio`
