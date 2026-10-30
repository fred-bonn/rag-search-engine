import argparse, json

from keyword_search import search_command

DATA_FILE_PATH = "data/movies.json"

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, movie in enumerate(results):
                print(f"{i}: {movie['title']}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()