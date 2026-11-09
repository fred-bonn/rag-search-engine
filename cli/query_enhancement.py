import os
from collections.abc import Callable
from functools import cache

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

API_KEY_ENV = "OPENROUTER_API_KEY"
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openrouter/free"

Enhancement = Callable[[str], str]

@cache
def get_client() -> OpenAI:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} environment variable not set")
    return OpenAI(base_url=BASE_URL, api_key=api_key)

def spell_prompt(query: str) -> str:
    return f"""
Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

def rewrite_prompt(query: str) -> str:
    return f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

def expand_prompt(query: str) -> str:
    return f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""

ENHANCEMENTS: dict[str, Enhancement] = {
    "spell": spell_prompt,
    "rewrite": rewrite_prompt,
    "expand": expand_prompt,
}

def enhance_query(query: str, enhancement: str | None) -> str:
    if enhancement is None:
        return query
    
    response = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": ENHANCEMENTS[enhancement](query),
                }
            ],
        )
    enhanced_query = response.choices[0].message.content
    if enhanced_query is None:
        raise RuntimeError("LLM returned an empty response")
    enhanced_query = enhanced_query.strip()
    print(f"Enhanced query ({enhancement}): '{query}' -> '{enhanced_query}'")

    return enhanced_query
