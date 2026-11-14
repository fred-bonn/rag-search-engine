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


def summarize_prompt(query: str, results: Sequence[Mapping[str, Any]]) -> str:
    docs = "\n".join(
        f"{rank}. {result['title']}: {result['document']}"
        for rank, result in enumerate(results, start=1)
    )
    return f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{docs}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""


def generate_summarize_response(query: str, results: Sequence[Mapping[str, Any]]) -> str:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": summarize_prompt(query, results),
            }
        ],
    )
    answer = response.choices[0].message.content
    if answer is None:
        raise RuntimeError("LLM returned an empty response")
    return answer.strip()


def citations_prompt(query: str, results: Sequence[Mapping[str, Any]]) -> str:
    docs = "\n".join(
        f"[{rank}] {result['title']}: {result['document']}"
        for rank, result in enumerate(results, start=1)
    )
    return f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""


def generate_citations_response(query: str, results: Sequence[Mapping[str, Any]]) -> str:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": citations_prompt(query, results),
            }
        ],
    )
    answer = response.choices[0].message.content
    if answer is None:
        raise RuntimeError("LLM returned an empty response")
    return answer.strip()
