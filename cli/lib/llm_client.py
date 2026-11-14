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
