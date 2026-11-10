from collections.abc import Callable
from functools import wraps
from time import sleep
from typing import TypeVar

from .query_enhancement import (
    MODEL,
    get_client,
)
from .search_utils import (
    RERANK_DELAY_SECONDS,
    RERANK_LIMIT_MULTIPLIER,
    HybridSearchResult,
    KeywordSearchResult,
    RRFSearchResult,
    ScoredSearchResult,
)

SearchResult = TypeVar(
    "SearchResult",
    KeywordSearchResult,
    ScoredSearchResult,
    HybridSearchResult,
    RRFSearchResult,
)

Reranker = Callable[[str, list[SearchResult]], list[SearchResult]]


def rerank(
    method: str | None,
    multiplier: int = RERANK_LIMIT_MULTIPLIER,
) -> Callable[..., Callable[..., list[SearchResult]]]:
    """Wrap a search command so it over-fetches results, reranks them, and truncates to the requested limit."""

    def decorator(
        search_command: Callable[..., list[SearchResult]],
    ) -> Callable[..., list[SearchResult]]:
        if method is None:
            return search_command

        @wraps(search_command)
        def wrapper(query: str, *args, limit: int, **kwargs) -> list[SearchResult]:
            results = search_command(query, *args, limit=limit * multiplier, **kwargs)
            assert method is not None
            return rerank_results(query, results, method)[:limit]

        return wrapper

    return decorator


def rerank_results(query: str, results: list[SearchResult], method: str) -> list[SearchResult]:
    return RERANKERS[method](query, results)


def individual_prompt(query: str, doc: SearchResult) -> str:
    return f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""


def individual_rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    scored_results: list[tuple[int, SearchResult]] = []

    for index, result in enumerate(results):
        if index > 0:
            sleep(RERANK_DELAY_SECONDS)

        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": individual_prompt(query, result),
                }
            ],
        )
        score_text = response.choices[0].message.content
        if score_text is None:
            raise RuntimeError("LLM returned an empty response")
        score = int(score_text.strip())
        result["rerank_score"] = score
        scored_results.append((score, result))

    scored_results.sort(key=lambda scored: scored[0], reverse=True)
    return [result for _, result in scored_results]


RERANKERS: dict[str, Reranker] = {
    "individual": individual_rerank,
}
