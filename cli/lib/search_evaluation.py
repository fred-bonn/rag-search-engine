import json
from collections.abc import Mapping, Sequence
from typing import Any

from .llm_client import (
    MODEL,
    get_client,
)


def relevance_prompt(query: str, results: Sequence[Mapping[str, Any]]) -> str:
    formatted_results = [
        f"{rank}. {result['title']}: {result['document']}"
        for rank, result in enumerate(results, start=1)
    ]
    return f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""


def evaluate_relevance(query: str, results: Sequence[Mapping[str, Any]]) -> list[int]:
    if not results:
        return []

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": relevance_prompt(query, results),
            }
        ],
    )
    score_text = response.choices[0].message.content
    if score_text is None:
        raise RuntimeError("LLM returned an empty response")

    scores = json.loads(score_text)
    if not isinstance(scores, list) or len(scores) != len(results):
        raise ValueError(f"Expected {len(results)} scores from the LLM, got: {score_text}")
    if any(not isinstance(score, int) or score < 0 or score > 3 for score in scores):
        raise ValueError(f"LLM scores must be integers from 0 to 3, got: {score_text}")

    return scores
