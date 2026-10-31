import string, pickle
from nltk.stem import PorterStemmer
from pathlib import Path

from .search_utils import load_movies, DATA_PATH_STOPWORDS, PROJECT_ROOT, Movie


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

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, []))

    def build(self) -> None:
        movies = load_movies()
        for doc_id, movie in movies.items():
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie["title"]} {movie["description"]}")

    def save(self) -> None:
        cache_dir = PROJECT_ROOT / "cache"
        cache_dir.mkdir(exist_ok=True)

        index_path = cache_dir / "index.pkl"
        docmap_path = cache_dir / "docmap.pkl"

        with index_path.open("wb") as f:
            pickle.dump(self.index, f)
        with docmap_path.open("wb") as f:
            pickle.dump(self.docmap, f)

def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()
    print(f"First document for token 'merida': = {index.get_documents('merida')}")


def tokenize(text: str) -> list[str]:
    text = process(text)
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

    query_tokens = [token for token in tokenize(query)]
    for token in query_tokens:
        if token in STOPWORDS:
            query_tokens.remove(token)

    movies = load_movies()
    results: list[Movie] = []
    for movie in movies.values():
        title_tokens = [token for token in tokenize(movie["title"])]
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) == limit:
                break

    return results