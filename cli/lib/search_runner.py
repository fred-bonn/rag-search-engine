from collections.abc import Callable
from typing import Any, Literal, NamedTuple

from .hybrid_search import (
    rrf_search_command,
    weighted_search_command,
)
from .keyword_search import (
    bm25_search_command,
    search_command,
)
from .query_enhancement import enhance_query
from .reranking import RERANKERS
from .search_logging import (
    log_error,
    log_rerank,
    log_search,
    new_search_id,
)
from .search_utils import RERANK_LIMIT_MULTIPLIER
from .semantic_search import (
    search_chunked_command,
    semantic_search_command,
)


SearchCommand = Literal[
    "keyword",
    "bm25",
    "semantic",
    "semantic_chunked",
    "weighted",
    "rrf",
]


class SearchCommandSpec(NamedTuple):
    engine: str
    search_type: str
    command: Callable[..., list[Any]]


SEARCH_COMMANDS: dict[SearchCommand, SearchCommandSpec] = {
    "keyword": SearchCommandSpec("keyword", "search", search_command),
    "bm25": SearchCommandSpec("keyword", "bm25search", bm25_search_command),
    "semantic": SearchCommandSpec("semantic", "search", semantic_search_command),
    "semantic_chunked": SearchCommandSpec("semantic", "chunked_search", search_chunked_command),
    "weighted": SearchCommandSpec("hybrid", "weighted_search", weighted_search_command),
    "rrf": SearchCommandSpec("hybrid", "rrf_search", rrf_search_command),
}


def run_search_pipeline(
    command: SearchCommand,
    *,
    query: str,
    limit: int,
    enhancement: str | None = None,
    rerank_method: str | None = None,
    command_kwargs: dict[str, Any] | None = None,
) -> list[Any]:
    """Run a search with optional query enhancement and re-ranking, logging each stage."""
    spec = SEARCH_COMMANDS[command]
    search_id = new_search_id()
    kwargs = command_kwargs or {}

    try:
        enhanced_query = enhance_query(query, enhancement)
        fetch_limit = limit * RERANK_LIMIT_MULTIPLIER if rerank_method is not None else limit
        results = spec.command(enhanced_query, limit=fetch_limit, **kwargs)

        log_search(
            search_id=search_id,
            query=query,
            engine=spec.engine,
            search_type=spec.search_type,
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
            engine=spec.engine,
            search_type=spec.search_type,
            error=e,
        )
        raise
