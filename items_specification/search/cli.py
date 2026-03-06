"""
search/cli.py
-------------
CLI demo for semantic search over construction item specifications.

Usage:
    python search/cli.py "keramik lantai 60x60"
    python search/cli.py "Toto toilet" --nama-proyek "SPPG" --top-k 5
"""

import argparse
import sys

from search import search_items

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

SEP = "─" * 78


def _col(text: str, width: int) -> str:
    """Left-align text, truncating with ellipsis if too long."""
    text = text or ""
    if len(text) > width:
        return text[: width - 1] + "…"
    return text.ljust(width)


def print_results(query: str, results: list[dict]) -> None:
    print(f'\nQuery: "{query}"')
    print(SEP)
    print(f" {'#':<3} {'Score':<7} {'Project':<30} {'Header':<24} {'Item'}")
    print(SEP)

    if not results:
        print("  No results found.")
        print(SEP)
        return

    for i, r in enumerate(results, start=1):
        score        = r.get("score") or 0.0
        project      = r.get("nama_bangunan") or r.get("nama_proyek") or ""
        h1           = r.get("header1") or ""
        h2           = r.get("header2") or ""
        header       = f"{h1} > {h2}".strip(" >") if (h1 or h2) else ""
        nama         = r.get("nama_barang") or ""
        spec_general = r.get("spesifikasi_general") or ""
        spec_merek   = r.get("spesifikasi_merek") or ""

        print(
            f" {i:<3} {score:<7.4f} {_col(project, 30)} {_col(header, 24)} {nama}"
        )
        if spec_general:
            print(f"{'':>12} Spec : {spec_general}")
        if spec_merek:
            print(f"{'':>12} Merek: {spec_merek}")
        print()

    print(SEP)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic search over construction item specifications."
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument(
        "--nama-proyek",
        default=None,
        metavar="NAME",
        help="Restrict results to projects whose name contains this substring (case-insensitive)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        metavar="N",
        help="Number of results to return (default: 10)",
    )

    args = parser.parse_args()

    try:
        results = search_items(
            query=args.query,
            nama_proyek=args.nama_proyek,
            top_k=args.top_k,
        )
    except Exception as exc:
        print(f"ERROR: Search failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print_results(args.query, results)


if __name__ == "__main__":
    main()
