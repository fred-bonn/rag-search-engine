import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    embed_query_text,
    verify_embeddings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Query to generate embedding for")

    subparsers.add_parser("verify_embeddings", help="Verify the movie embeddings")

    args = parser.parse_args()

    try:
        match args.command:
            case "verify":
                verify_model()
            case "embed_text":
                embed_text(args.text)
            case "embed_query":
                embed_query_text(args.query)
            case "verify_embeddings":
                verify_embeddings()
            case _:
                parser.print_help()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()