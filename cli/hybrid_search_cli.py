import argparse

from lib.hybrid_search import (
    normalize_command,
    weighted_search_command
)
from lib.search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of integers")
    normalize_parser.add_argument("numbers", type=float, nargs="*", help="List of integers to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Search movies using weighted search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Weighting factor for hybrid search")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results")

    args = parser.parse_args()

    try:
        match args.command:
            case "weighted-search":
                results = weighted_search_command(args.query, args.alpha, args.limit)
                for i, result in enumerate(results, start=1):
                    print(f"{i}. {result["title"]}")
                    print(f"   Hybrid Score: {result["hybrid_score"]:.4f}")
                    print(f"   BM25 Score: {result["bm25_score"]:.4f}, Semantic Score: {result["semantic_score"]:.4f}")
                    print(f"   Document: {result["document"][:100]}")
            case "normalize":
                for score in normalize_command(args.numbers):
                    print(f"{score:.4f}")
            case _:
                parser.print_help()
    except Exception as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())