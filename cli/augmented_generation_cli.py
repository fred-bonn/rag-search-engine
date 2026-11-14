import argparse

from lib.augmented_generation import (
    generate_rag_response,
    generate_summarize_response,
)
from lib.formatting import (
    format_rag_response,
)
from lib.search_utils import (
    DEFAULT_RRF_K,
    DEFAULT_SEARCH_LIMIT,
)
from lib.search_runner import (
    run_search_pipeline,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize the results of a search"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for summarize")
    summarize_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results")

    args = parser.parse_args()

    match args.command:
        case "rag":
            results = run_search_pipeline(
                "rrf",
                query=args.query,
                limit=DEFAULT_SEARCH_LIMIT,
                command_kwargs={"k": DEFAULT_RRF_K},
            )
            answer = generate_rag_response(args.query, results)
            print(format_rag_response(results, answer))
        case "summarize":
            results = run_search_pipeline(
                "rrf",
                query=args.query,
                limit=args.limit,
                command_kwargs={"k": DEFAULT_RRF_K},
            )
            answer = generate_summarize_response(args.query, results)
            print(format_rag_response(results, answer))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()