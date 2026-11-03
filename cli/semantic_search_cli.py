import argparse

from lib.semantic_search import (
    verify_model,
    embed_text,
    embed_query_text,
    verify_embeddings,
    semantic_search,
    fixed_size_chunking,
    semantic_chunking,
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_SIZE_SEMANTIC,
    DEFAULT_OVERLAP,
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

    search_parser = subparsers.add_parser("search", help="Search movies by semantic similarity")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Number of overlapping words between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a text into semantically meaningful pieces")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_CHUNK_SIZE_SEMANTIC, help="Maximum size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="Number of overlapping words between chunks")

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
            case "search":
                semantic_search(args.query, args.limit)
            case "chunk":
                fixed_size_chunking(args.text, args.chunk_size, args.overlap)
            case "semantic_chunk":
                semantic_chunking(args.text, args.max_chunk_size, args.overlap)
            case _:
                parser.print_help()
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()