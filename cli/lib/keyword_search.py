import pickle, string, math
from collections import Counter
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
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1

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
        term_frequencies_path = CACHE_DIR / "term_frequencies.pkl"

        with index_path.open("wb") as f:
            pickle.dump(self.index, f)
        with docmap_path.open("wb") as f:
            pickle.dump(self.docmap, f)
        with term_frequencies_path.open("wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        index_path = CACHE_DIR / "index.pkl"
        docmap_path = CACHE_DIR / "docmap.pkl"
        term_frequencies_path = CACHE_DIR / "term_frequencies.pkl"

        paths = [index_path, docmap_path, term_frequencies_path]

        if not all(path.exists() for path in paths):
            raise FileNotFoundError("Cache files not found. Build first.")

        with index_path.open("rb") as f:
            self.index = pickle.load(f)
        with docmap_path.open("rb") as f:
            self.docmap = pickle.load(f)
        with term_frequencies_path.open("rb") as f:
            self.term_frequencies = pickle.load(f)

        return

    def get_term_frequency(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def get_inverse_document_frequency(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_term_frequency(doc_id, term)
        idf = self.get_inverse_document_frequency(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log(
            ((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5)) + 1
        )

def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    index = InvertedIndex()
    index.load()

    if limit <= 0:
        return []

    query_tokens = [token for token in tokenize(query)]
    for token in query_tokens:
        if token in STOPWORDS:
            query_tokens.remove(token)

    seen: set[int] = set()
    results: list[Movie] = []
    for token in query_tokens:
        for doc_id in index.get_documents(token):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc = index.docmap[doc_id]
            results.append(doc)
            if len(results) >= limit:
                return results

    return results

def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    return index.get_term_frequency(doc_id, tokenize_term(term))

def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_inverse_document_frequency(tokenize_term(term))

def tf_idf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_tf_idf(doc_id, tokenize_term(term))

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(tokenize_term(term))

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

def tokenize_term(term: str) -> str:
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError("Term must tokenize to exactly one token.")
    return tokens[0]