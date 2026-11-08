import argparse

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
from llm import (
    spell_correct_query,
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

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Search movies using RRF search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("-k", type=int, default=DEFAULT_RRF_K, help="Higher values gives lower ranked movies more influence")
    rrf_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell"], help="Query enhancement method")

    args = parser.parse_args()

    try:
        match args.command:
            case "weighted-search":
                results = weighted_search_command(args.query, args.alpha, args.limit)
                for i, result in enumerate(results, start=1):
                    print(f"{i}. {result["title"]}")
                    print(f"   Hybrid Score: {result["hybrid_score"]:.4f}")
                    print(f"   BM25 Score: {result["bm25_score"]:.4f}, Semantic Score: {result["semantic_score"]:.4f}")
                    print(f"   {result["document"][:100]}")
            case "rrf-search":
                query = args.query
                if args.enhance == "spell":
                    query = spell_correct_query(query)
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'")
                results = rrf_search_command(query, args.k, args.limit)
                for i, result in enumerate(results, start=1):
                    bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "N/A"
                    semantic_rank = result["semantic_rank"] if result["semantic_rank"] is not None else "N/A"
                    print(f"{i}. {result["title"]}")
                    print(f"   RRF Score: {result["rrf_score"]:.4f}")
                    print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                    print(f"   {result["document"][:100]}")
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