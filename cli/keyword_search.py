from search_utils import load_movies, DATA_PATH_STOPWORDS, Movie
import string as string


DEFAULT_SEARCH_LIMIT = 5

def load_stopwords() -> list[str]:
    with DATA_PATH_STOPWORDS.open("r") as f:
        stopwords = [line.strip() for line in f]
    return stopwords


STOPWORDS = load_stopwords()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    if limit <= 0:
        return []

    query = process(query)
    query_tokens = [token for token in tokenize(process(query)) if token not in STOPWORDS]
    for token in query_tokens:
        if token in STOPWORDS:
            query_tokens.remove(token)

    movies = load_movies()
    results: list[Movie] = []
    for movie in movies.values():
        title_tokens = [token for token in tokenize(process(movie["title"])) if token not in STOPWORDS]
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) == limit:
                break

    return results

def process(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.lower()

def tokenize(text: str) -> list[str]:
    return text.split()

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    return any(
        query_token in title_token
        for query_token in query_tokens
        for title_token in title_tokens
    )