import argparse

from lib.formatting import (
    format_keyword_result,
    format_relevance_report,
    format_scored_result,
)
from lib.keyword_search import (
    BM25_K1,
    BM25_B,
    build_command,
    tf_command,
    idf_command,
    tf_idf_command,
    bm25_idf_command,
    bm25_tf_command,
)
from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT
)
from lib.query_enhancement import (
    ENHANCEMENTS,
)
from lib.reranking import (
    RERANKERS,
)
from lib.search_evaluation import evaluate_relevance
from lib.search_runner import run_search_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results"
    )
    search_parser.add_argument("--enhance", type=str, choices=list(ENHANCEMENTS), help="Query enhancement method")
    search_parser.add_argument("--rerank-method", type=str, choices=list(RERANKERS), help="Re-ranks results with LLMs")
    search_parser.add_argument("--evaluate", action="store_true", help="Rate each result's relevance with an LLM (0-3)")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document ID and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="Term to get inverse document frequency for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF for a term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF for a term in a document")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 B parameter")

    bm25_search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument(
        "--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Maximum number of results"
    )
    bm25_search_parser.add_argument("--enhance", type=str, choices=list(ENHANCEMENTS), help="Query enhancement method")
    bm25_search_parser.add_argument("--rerank-method", type=str, choices=list(RERANKERS), help="Re-ranks results with LLMs")
    bm25_search_parser.add_argument("--evaluate", action="store_true", help="Rate each result's relevance with an LLM (0-3)")

    args = parser.parse_args()

    try:
        match args.command:
            case "build":
                print(f"Building the inverted index...")
                build_command()
                print("Inverted index built successfully.")
            case "search":
                results = run_search_pipeline(
                    "keyword",
                    query=args.query,
                    limit=args.limit,
                    enhancement=args.enhance,
                    rerank_method=args.rerank_method,
                )
                for i, result in enumerate(results, start=1):
                    print(format_keyword_result(result, i))
                if args.evaluate:
                    scores = evaluate_relevance(args.query, results)
                    print()
                    print(format_relevance_report(results, scores))
            case "bm25search":
                results = run_search_pipeline(
                    "bm25",
                    query=args.query,
                    limit=args.limit,
                    enhancement=args.enhance,
                    rerank_method=args.rerank_method,
                )
                for i, result in enumerate(results, start=1):
                    print(format_scored_result(result, i))
                if args.evaluate:
                    scores = evaluate_relevance(args.query, results)
                    print()
                    print(format_relevance_report(results, scores))
            case "tf":
                tf = tf_command(args.doc_id, args.term)
                print(f"Term frequency for '{args.term}' in document {args.doc_id}: {tf}")
            case "idf":
                idf = idf_command(args.term)
                print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            case "tfidf":
                tf_idf = tf_idf_command(args.doc_id, args.term)
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
            case "bm25idf":
                bm25_idf = bm25_idf_command(args.term)
                print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
            case "bm25tf":
                bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
                print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
            case _:
                parser.print_help()
    except Exception as e:
        print(e)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())