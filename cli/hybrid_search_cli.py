import argparse

from lib.hybrid_search import (
    normalize_command
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of integers")
    normalize_parser.add_argument("numbers", type=float, nargs="*", help="List of integers to normalize")

    args = parser.parse_args()

    try:
        match args.command:
            case "normalize":
                for score in normalize_command(args.numbers):
                    print(f"{score:.4f}")
            case _:
                parser.print_help()
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()