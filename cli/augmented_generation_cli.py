import argparse

from lib.augmented_generation import (
    generate_rag_response,
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()