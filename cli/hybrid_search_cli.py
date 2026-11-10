import argparse

from lib.formatting import (
    format_hybrid_result,
    format_rrf_result,
)
from lib.hybrid_search import (
    normalize_command,
    weighted_search_command,
    rrf_search_command,
)
from lib.search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_RRF_K,
)
from lib.query_enhancement import (
    ENHANCEMENTS,
    enhance_query,
)
from lib.reranking import (
    rerank,
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
    weighted_search_parser.add_argument("--enhance", type=str, choices=list(ENHANCEMENTS), help="Query enhancement method")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Search movies using RRF search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("-k", type=int, default=DEFAULT_RRF_K, help="Higher values gives lower ranked movies more influence")
    rrf_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results")
    rrf_search_parser.add_argument("--enhance", type=str, choices=list(ENHANCEMENTS), help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual"], help="Re-ranks results with LLMs")

    args = parser.parse_args()

    try:
        match args.command:
            case "weighted-search":
                query = enhance_query(args.query, args.enhance)
                results = weighted_search_command(query, args.alpha, args.limit)
                for i, result in enumerate(results, start=1):
                    print(format_hybrid_result(result, i))
            case "rrf-search":
                query = enhance_query(args.query, args.enhance)
                search_command = rerank(args.rerank_method)(rrf_search_command)
                results = search_command(query, args.k, limit=args.limit)
                for i, result in enumerate(results, start=1):
                    print(format_rrf_result(result, i))
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