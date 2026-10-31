from nltk.stem import PorterStemmer

from .search_utils import load_movies, DATA_PATH_STOPWORDS, Movie
import string as string


DEFAULT_SEARCH_LIMIT = 5
stemmer = PorterStemmer()

def process(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.lower()

def load_stopwords() -> list[str]:
    with DATA_PATH_STOPWORDS.open("r") as f:
        stopwords = [process(line) for line in f]
    return stopwords

STOPWORDS = load_stopwords()

def tokenize(text: str) -> list[str]:
    return [stemmer.stem(token) for token in text.split() if token not in STOPWORDS]

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    return any(
        query_token in title_token
        for query_token in query_tokens
        for title_token in title_tokens
    )

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    if limit <= 0:
        return []

    query_tokens = [token for token in tokenize(process(query))]
    for token in query_tokens:
        if token in STOPWORDS:
            query_tokens.remove(token)

    movies = load_movies()
    results: list[Movie] = []
    for movie in movies.values():
        title_tokens = [token for token in tokenize(process(movie["title"]))]
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) == limit:
                break

    return results