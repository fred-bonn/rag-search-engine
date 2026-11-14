from collections.abc import Mapping, Sequence
from typing import Any

from .query_enhancement import (
    MODEL,
    get_client,
)


def rag_prompt(query: str, results: Sequence[Mapping[str, Any]]) -> str:
    docs = "\n".join(
        f"{rank}. {result['title']}: {result['document']}"
        for rank, result in enumerate(results, start=1)
    )
    return f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""


def generate_rag_response(query: str, results: Sequence[Mapping[str, Any]]) -> str:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": rag_prompt(query, results),
            }
        ],
    )
    answer = response.choices[0].message.content
    if answer is None:
        raise RuntimeError("LLM returned an empty response")
    return answer.strip()
