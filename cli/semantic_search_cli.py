import argparse

from lib.semantic_search import (
    verify_model_command,
    embed_text_command,
    embed_query_text_command,
    verify_embeddings_command,
    embed_chunks_command,
    semantic_search_command,
    search_chunked_command,
    fixed_size_chunking_command,
    semantic_chunking_command,
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_SIZE_SEMANTIC,
    DEFAULT_OVERLAP_CLI,
)
from lib.query_enhancement import (
    ENHANCEMENTS,
    enhance_query,
)


def print_chunks(header: str, chunks: list[str]) -> None:
    print(header)
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. {chunk}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Query to generate embedding for")

    subparsers.add_parser("verify_embeddings", help="Verify the movie embeddings")

    subparsers.add_parser("embed_chunks", help="Generate and cache chunk embeddings")

    search_parser = subparsers.add_parser("search", help="Search movies by semantic similarity")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results")
    search_parser.add_argument("--enhance", type=str, choices=list(ENHANCEMENTS), help="Query enhancement method")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies by chunk semantic similarity")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results")
    search_chunked_parser.add_argument("--enhance", type=str, choices=list(ENHANCEMENTS), help="Query enhancement method")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_CLI, help="Number of overlapping words between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk a text into semantically meaningful pieces")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_CHUNK_SIZE_SEMANTIC, help="Maximum size of each chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_CLI, help="Number of overlapping words between chunks")

    args = parser.parse_args()

    try:
        match args.command:
            case "verify":
                model, max_seq_length = verify_model_command()
                print(f"Model loaded:        {model}")
                print(f"Max sequence length: {max_seq_length}")
            case "embed_text":
                embedding = embed_text_command(args.text)
                print(f"Text:                            {args.text}")
                print(f"First 3 dimensions of embedding: {embedding[:3]}")
                print(f"Dimensions:                      {embedding.shape[0]}")
            case "embed_query":
                embedding = embed_query_text_command(args.query)
                print(f"Text:                            {args.query}")
                print(f"First 3 dimensions of embedding: {embedding[:3]}")
                print(f"Dimensions:                      {embedding.shape[0]}")
            case "verify_embeddings":
                embeddings = verify_embeddings_command()
                print(f"Number of docs:   {embeddings.shape[0]}")
                print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
            case "search":
                query = enhance_query(args.query, args.enhance)
                results = semantic_search_command(query, args.limit)
                for rank, result in enumerate(results, start=1):
                    print(f"{rank}: {result['title']}: {result['document']}")
            case "search_chunked":
                query = enhance_query(args.query, args.enhance)
                results = search_chunked_command(query, args.limit)
                for i, result in enumerate(results, start=1):
                    print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                    print(f"   {result['document'][:100]}...")
            case "chunk":
                chunks = fixed_size_chunking_command(args.text, args.chunk_size, args.overlap)
                print_chunks(f"Chunking {len(args.text)} characters", chunks)
            case "semantic_chunk":
                chunks = semantic_chunking_command(args.text, args.max_chunk_size, args.overlap)
                print_chunks(f"Semantically chunking {len(args.text)} characters", chunks)
            case "embed_chunks":
                embeddings = embed_chunks_command()
                print(f"Generated {len(embeddings)} chunked embeddings")
            case _:
                parser.print_help()
    except Exception as e:
        print(e)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())