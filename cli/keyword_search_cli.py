import argparse

from lib.keyword_search import (
    search_command,
    build_command,
    tf_command,
    idf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document ID and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    try:
        match args.command:
            case "build":
                print(f"Building the inverted index...")
                build_command()
                print("Inverted index built successfully.")
            case "search":
                print(f"Searching for: {args.query}")
                results = search_command(args.query)
                for i, movie in enumerate(results):
                    print(f"{i+1}: {movie['title']}")
            case "tf":
                tf = tf_command(args.doc_id, args.term)
                print(f"Term frequency for '{args.term}' in document {args.doc_id}: {tf}")
            case "idf":
                idf = idf_command(args.term)
                print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            case _:
                parser.print_help()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()