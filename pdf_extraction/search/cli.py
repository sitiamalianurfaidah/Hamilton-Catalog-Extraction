"""
search/cli.py
-------------
CLI demo for semantic search over the construction catalog.

Usage:
    python search/cli.py "keramik lantai 60x60"
    python search/cli.py "semen portland" --merk "Tiga Roda" --top-k 5
    python search/cli.py "pipa PVC" --jenis "Pipa" --top-k 10
"""

import argparse
import sys

from search import search_items

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

SEP = "─" * 88


def _col(text: str, width: int) -> str:
    """Left-align text, truncating with ellipsis if too long."""
    text = str(text) if text is not None else ""
    if len(text) > width:
        return text[: width - 1] + "…"
    return text.ljust(width)


def _fmt_price(harga) -> str:
    if harga is None:
        return "-"
    try:
        return f"Rp {int(harga):,}".replace(",", ".")
    except (TypeError, ValueError):
        return str(harga)


def print_results(query: str, results: list[dict]) -> None:
    print(f'\nQuery: "{query}"')
    print(SEP)
    print(f" {'#':<3} {'Score':<7} {'Nama Barang':<34} {'Merk':<18} {'Harga'}")
    print(SEP)

    if not results:
        print("  No results found.")
        print(SEP)
        return

    for i, r in enumerate(results, start=1):
        score  = r.get("score") or 0.0
        nama   = r.get("nama_barang") or ""
        jenis  = r.get("jenis_tipe_barang") or ""
        merk   = r.get("merk_barang") or ""
        harga  = _fmt_price(r.get("harga_barang"))
        spec   = r.get("spesifikasi")

        print(
            f" {i:<3} {score:<7.4f} {_col(nama, 34)} {_col(merk, 18)} {harga}"
        )
        if jenis:
            print(f"{'':>12} Jenis: {jenis}")
        if spec:
            print(f"{'':>12} Spec : {spec}")
        print()

    print(SEP)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic search over construction catalog items."
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument(
        "--merk",
        default=None,
        metavar="BRAND",
        help="Filter results by brand name (case-insensitive partial match)",
    )
    parser.add_argument(
        "--jenis",
        default=None,
        metavar="TYPE",
        help="Filter results by item type/category (case-insensitive partial match)",
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
            merk_barang=args.merk,
            jenis_tipe_barang=args.jenis,
            top_k=args.top_k,
        )
    except Exception as exc:
        print(f"ERROR: Search failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print_results(args.query, results)


if __name__ == "__main__":
    main()
