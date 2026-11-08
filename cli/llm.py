import os
from functools import cache

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

API_KEY_ENV = "OPENROUTER_API_KEY"
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openrouter/free"

@cache
def get_client() -> OpenAI:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"{API_KEY_ENV} environment variable not set")
    return OpenAI(base_url=BASE_URL, api_key=api_key)

def spell_correct_query(query: str) -> str:
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": spell_prompt(query),
            }
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned an empty response")

    return content.strip()

def spell_prompt(query: str) -> str:
    return f"""
Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""