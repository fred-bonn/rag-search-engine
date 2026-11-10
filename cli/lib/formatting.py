from .search_utils import (
    HybridSearchResult,
    KeywordSearchResult,
    RRFSearchResult,
    ScoredSearchResult,
)


def _append_rerank_score(lines: list[str], rerank_score: int | None) -> None:
    if rerank_score is not None:
        lines.append(f"   Re-rank Score: {rerank_score:.1f}/10")


def _truncate_document(document: str, max_length: int = 100) -> str:
    if len(document) <= max_length:
        return document
    return f"{document[:max_length]}..."


def format_keyword_result(result: KeywordSearchResult, rank: int) -> str:
    lines = [f"{rank}. {result['title']}"]
    _append_rerank_score(lines, result.get("rerank_score"))
    return "\n".join(lines)


def format_scored_result(result: ScoredSearchResult, rank: int) -> str:
    lines = [f"{rank}. {result['title']}"]
    _append_rerank_score(lines, result.get("rerank_score"))
    lines.extend(
        [
            f"   Score: {result['score']:.4f}",
            f"   {_truncate_document(result['document'])}",
        ]
    )
    return "\n".join(lines)


def format_hybrid_result(result: HybridSearchResult, rank: int) -> str:
    lines = [f"{rank}. {result['title']}"]
    _append_rerank_score(lines, result.get("rerank_score"))
    lines.extend(
        [
            f"   Hybrid Score: {result['hybrid_score']:.4f}",
            f"   BM25 Score: {result['bm25_score']:.4f}, Semantic Score: {result['semantic_score']:.4f}",
            f"   {_truncate_document(result['document'])}",
        ]
    )
    return "\n".join(lines)


def format_rrf_result(result: RRFSearchResult, rank: int) -> str:
    bm25_rank = result["bm25_rank"] if result["bm25_rank"] is not None else "N/A"
    semantic_rank = result["semantic_rank"] if result["semantic_rank"] is not None else "N/A"

    lines = [f"{rank}. {result['title']}"]
    _append_rerank_score(lines, result.get("rerank_score"))
    lines.extend(
        [
            f"   RRF Score: {result['rrf_score']:.3f}",
            f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}",
            f"   {_truncate_document(result['document'])}",
        ]
    )
    return "\n".join(lines)
