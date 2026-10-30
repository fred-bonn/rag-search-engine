from search_utils import load_movies, load_stopwords, Movie
import string as string

DEFAULT_SEARCH_LIMIT = 5

class SearchContext:
    instance = None
    stopwords: list[str]

    def __new__(cls):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
            cls.instance.stopwords = [process(word) for word in load_stopwords()]
        return cls.instance

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    if limit <= 0:
        return []

    search_context = SearchContext()

    query = process(query)
    query_tokens = [token for token in tokenize(process(query)) if token not in search_context.stopwords]
    for token in query_tokens:
        if token in search_context.stopwords:
            query_tokens.remove(token)

    movies = load_movies()
    results: list[Movie] = []
    for movie in movies.values():
        title_tokens = [token for token in tokenize(process(movie["title"])) if token not in search_context.stopwords]
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