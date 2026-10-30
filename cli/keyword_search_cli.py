import argparse, json

from search_utils import load_movies, title_match

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
            movies = load_movies()
            count = 0
            for movie in movies.values():
                if title_match(movie, args.query):
                    print(f"{movie['id']}: {movie['title']}")
                    count += 1
                if count == 5:
                    break
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()