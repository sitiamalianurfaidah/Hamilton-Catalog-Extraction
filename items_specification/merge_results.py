import glob
import json
import os
import re


def merge_all(metadata: dict, pages_dir: str = "output", out_path: str = "output/result.json") -> dict:
    """
    Merge per-page JSON files into a single result JSON.

    Reads all page_*.json files in `pages_dir`, flattens every "items" array
    across pages (in page-number order), attaches project metadata as top-level
    keys, and writes the combined result to `out_path`.

    Args:
        metadata:   Dict returned by Stage 1 (contains project fields +
                    "context_prompt"; context_prompt is excluded from output).
        pages_dir:  Directory containing page_*.json files.
        out_path:   Destination file for the final merged JSON.

    Returns:
        The merged result dict.
    """
    # Collect and sort page files by their page number
    pattern = os.path.join(pages_dir, "page_*.json")
    page_files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"page_(\d+)\.json", p).group(1)),
    )

    all_items = []
    for fpath in page_files:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        items = data.get("items", [])
        all_items.extend(items)

    # Build output — include metadata fields but exclude internal keys
    excluded_keys = {"context_prompt"}
    result = {k: v for k, v in metadata.items() if k not in excluded_keys}
    result["items"] = all_items

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(page_files)} page(s), {len(all_items)} item(s) → {out_path}")
    return result
