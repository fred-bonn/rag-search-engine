from collections.abc import Callable
from typing import Any, TypeVar

from .query_enhancement import enhance_query
from .reranking import RERANKERS
from .search_logging import (
    log_error,
    log_rerank,
    log_search,
    new_search_id,
)
from .search_utils import RERANK_LIMIT_MULTIPLIER


SearchResult = TypeVar("SearchResult")


def run_search_pipeline(
    search_command: Callable[..., list[SearchResult]],
    *,
    query: str,
    engine: str,
    search_type: str,
    limit: int,
    enhancement: str | None = None,
    rerank_method: str | None = None,
    command_kwargs: dict[str, Any] | None = None,
) -> list[SearchResult]:
    """Run a search with optional query enhancement and re-ranking, logging each stage."""
    search_id = new_search_id()
    kwargs = command_kwargs or {}

    try:
        enhanced_query = enhance_query(query, enhancement)
        fetch_limit = limit * RERANK_LIMIT_MULTIPLIER if rerank_method is not None else limit
        results = search_command(enhanced_query, limit=fetch_limit, **kwargs)

        log_search(
            search_id=search_id,
            query=query,
            engine=engine,
            search_type=search_type,
            limit=limit,
            results=results,
            enhancement=enhancement,
            enhanced_query=enhanced_query if enhancement is not None else None,
            params=kwargs,
        )

        if rerank_method is None:
            return results

        reranked_results = RERANKERS[rerank_method](enhanced_query, results)[:limit]
        log_rerank(
            search_id=search_id,
            rerank_method=rerank_method,
            results=reranked_results,
        )
        return reranked_results
    except Exception as e:
        log_error(
            search_id=search_id,
            query=query,
            engine=engine,
            search_type=search_type,
            error=e,
        )
        raise
