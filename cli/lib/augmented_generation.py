from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .llm_client import (
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


def question_prompt(question: str, results: Sequence[Mapping[str, Any]]) -> str:
    docs = "\n".join(
        f"{rank}. {result['title']}: {result['document']}"
        for rank, result in enumerate(results, start=1)
    )
    return f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{docs}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""


ResponsePrompt = Callable[[str, Sequence[Mapping[str, Any]]], str]

PROMPTS: dict[str, ResponsePrompt] = {
    "rag": rag_prompt,
    "summarize": summarize_prompt,
    "citations": citations_prompt,
    "question": question_prompt,
}


def generate_response(
    query: str,
    results: Sequence[Mapping[str, Any]],
    prompt_type: str,
) -> str:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": PROMPTS[prompt_type](query, results),
            }
        ],
    )
    answer = response.choices[0].message.content
    if answer is None:
        raise RuntimeError("LLM returned an empty response")
    return answer.strip()
