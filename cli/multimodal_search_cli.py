import argparse

from lib.formatting import format_image_result
from lib.multimodal_search import (
    verify_image_embedding,
)
from lib.search_runner import run_search_pipeline
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding generation")
    verify_image_parser.add_argument("image", type=str, help="Path to the image file")

    image_search_parser = subparsers.add_parser("image_search", help="Search movies using an image")
    image_search_parser.add_argument("image", type=str, help="Path to the image file")
    image_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results")

    args = parser.parse_args()

    try:
        match args.command:
            case "verify_image_embedding":
                embedding = verify_image_embedding(args.image)
                print(f"Embedding shape: {embedding.shape[0]} dimensions")
            case "image_search":
                results = run_search_pipeline("image", query=args.image, limit=args.limit)
                for i, result in enumerate(results, start=1):
                    print(format_image_result(result, i))
                    print()
            case _:
                parser.print_help()
    except Exception as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
