import argparse

from lib.multimodal_search import (
    verify_image_embedding,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_parser = subparsers.add_parser("verify_image_embedding", help="Verify image embedding generation")
    verify_image_parser.add_argument("image", type=str, help="Path to the image file")

    args = parser.parse_args()

    try:
        match args.command:
            case "verify_image_embedding":
                embedding = verify_image_embedding(args.image)
                print(f"Embedding shape: {embedding.shape[0]} dimensions")
            case _:
                parser.print_help()
    except Exception as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
