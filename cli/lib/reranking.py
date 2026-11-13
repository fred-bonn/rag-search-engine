import json
from collections.abc import Callable
from functools import cache
from time import sleep
from typing import TypeVar

from sentence_transformers import CrossEncoder

from .query_enhancement import (
    MODEL,
    get_client,
)
from .search_utils import (
    CROSS_ENCODER_MODEL,
    RERANK_DELAY_SECONDS,
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


@cache
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(CROSS_ENCODER_MODEL)


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


def batch_prompt(query: str, results: list[SearchResult]) -> str:
    doc_list_str = "".join(
        f"  {doc['title']}: '{doc['document']}'\n"
        for doc in results
    )
    return f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""


def batch_rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": batch_prompt(query, results),
            }
        ],
    )
    ranking_text = response.choices[0].message.content
    if ranking_text is None:
        raise RuntimeError("LLM returned an empty response")

    ranked_ids = json.loads(ranking_text)
    rank_by_id = {movie_id: rank for rank, movie_id in enumerate(ranked_ids)}
    results.sort(key=lambda result: rank_by_id.get(result["id"], len(ranked_ids)))
    return results


def cross_encoder_rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    pairs = [
        [query, f"{doc.get('title', '')} - {doc.get('document', '')}"]
        for doc in results
    ]
    scores = get_cross_encoder().predict(pairs)

    scored_results: list[tuple[float, SearchResult]] = []
    for result, score in zip(results, scores):
        cross_encoder_score = float(score)
        result["cross_encoder_score"] = cross_encoder_score
        scored_results.append((cross_encoder_score, result))

    scored_results.sort(key=lambda scored: scored[0], reverse=True)
    return [result for _, result in scored_results]


RERANKERS: dict[str, Reranker] = {
    "individual": individual_rerank,
    "batch": batch_rerank,
    "cross_encoder": cross_encoder_rerank,
}
