import pickle, string
from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    STOPWORDS_PATH,
    DEFAULT_SEARCH_LIMIT,
    Movie,
    load_movies
)


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
        CACHE_DIR.mkdir(exist_ok=True)

        index_path = CACHE_DIR / "index.pkl"
        docmap_path = CACHE_DIR / "docmap.pkl"

        with index_path.open("wb") as f:
            pickle.dump(self.index, f)
        with docmap_path.open("wb") as f:
            pickle.dump(self.docmap, f)

    def load(self) -> None:
        index_path = CACHE_DIR / "index.pkl"
        docmap_path = CACHE_DIR / "docmap.pkl"

        if not index_path.exists() or not docmap_path.exists():
            raise FileNotFoundError("Cache files not found.")

        with index_path.open("rb") as f:
            self.index = pickle.load(f)
        with docmap_path.open("rb") as f:
            self.docmap = pickle.load(f)

        return

def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

def search_command(query: str, index: InvertedIndex, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    if limit <= 0:
        return []

    query_tokens = [token for token in tokenize(query)]
    for token in query_tokens:
        if token in STOPWORDS:
            query_tokens.remove(token)

    results: list[Movie] = []
    for token in query_tokens:
        for doc_id in index.get_documents(token):
            movie = index.docmap[doc_id]
            if movie not in results:
                results.append(movie)
                if len(results) == limit:
                    break

    return results

def process(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.lower()

def load_stopwords() -> list[str]:
    with STOPWORDS_PATH.open("r") as f:
        stopwords = [process(line) for line in f]
    return stopwords

STOPWORDS = load_stopwords()

def tokenize(text: str) -> list[str]:
    text = process(text)
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in text.split() if token not in STOPWORDS]