import pickle, string, math
from collections import Counter
from nltk.stem import PorterStemmer
from pathlib import Path

from .search_utils import (
    CACHE_DIR,
    STOPWORDS_PATH,
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    Movie,
    KeywordSearchResult,
    ScoredSearchResult,
    load_movies
)


def process(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def load_stopwords() -> list[str]:
    with STOPWORDS_PATH.open("r") as f:
        stopwords = [process(line).strip() for line in f]
    return stopwords

STOPWORDS: list[str] = load_stopwords()

def tokenize(text: str) -> list[str]:
    text = process(text)
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in text.split() if token not in STOPWORDS]

def tokenize_term(term: str) -> str:
    tokens = tokenize(term)
    if len(tokens) != 1:
        raise ValueError("Term must tokenize to exactly one token.")
    return tokens[0]

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}
        self.index_path = CACHE_DIR / "index.pkl"
        self.docmap_path = CACHE_DIR / "docmap.pkl"
        self.term_frequencies_path = CACHE_DIR / "term_frequencies.pkl"
        self.doc_lengths_path = CACHE_DIR / "doc_lengths.pkl"

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def _get_cache_paths(self) -> list[Path]:
        return [
            self.index_path,
            self.docmap_path,
            self.term_frequencies_path,
            self.doc_lengths_path,
        ]

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, []))

    def build(self) -> None:
        movies = load_movies()
        for doc_id, movie in movies.items():
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie["title"]} {movie["description"]}")

    def save(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)

        with self.index_path.open("wb") as f:
            pickle.dump(self.index, f)
        with self.docmap_path.open("wb") as f:
            pickle.dump(self.docmap, f)
        with self.term_frequencies_path.open("wb") as f:
            pickle.dump(self.term_frequencies, f)
        with self.doc_lengths_path.open("wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        if not all(path.exists() for path in self._get_cache_paths()):
            raise FileNotFoundError("Cache files not found. Build first.")

        with self.index_path.open("rb") as f:
            self.index = pickle.load(f)
        with self.docmap_path.open("rb") as f:
            self.docmap = pickle.load(f)
        with self.term_frequencies_path.open("rb") as f:
            self.term_frequencies = pickle.load(f)
        with self.doc_lengths_path.open("rb") as f:
            self.doc_lengths = pickle.load(f)

        return

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get(doc_id, Counter()).get(term, 0)

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log(
            ((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5)) + 1
        )

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return tf * idf

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[KeywordSearchResult]:
        query_tokens = tokenize(query)

        seen: set[int] = set()
        results: list[KeywordSearchResult] = []
        for token in query_tokens:
            for doc_id in self.get_documents(token):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                movie = self.docmap[doc_id]
                results.append(
                    {
                        "id": movie["id"],
                        "title": movie["title"],
                        "document": movie["description"],
                    }
                )
                if len(results) >= limit:
                    return results

        return results

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[ScoredSearchResult]:
        query_tokens = tokenize(query)

        scores: dict[int, float] = {}
        for token in query_tokens:
            for doc_id in self.get_documents(token):
                scores[doc_id] = scores.get(doc_id, 0) + self.bm25(doc_id, token)

        ranked_doc_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)[:limit]
        results: list[ScoredSearchResult] = []
        for doc_id in ranked_doc_ids:
            movie = self.docmap[doc_id]
            results.append(
                {
                    "id": movie["id"],
                    "title": movie["title"],
                    "document": movie["description"],
                    "score": scores[doc_id],
                }
            )
        return results


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[KeywordSearchResult]:
    if limit <= 0:
        return []
    index = InvertedIndex()
    index.load()
    return index.search(query, limit)

def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[ScoredSearchResult]:
    if limit <= 0:
        return []
    index = InvertedIndex()
    index.load()
    return index.bm25_search(query, limit)

def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    return index.get_tf(doc_id, tokenize_term(term))

def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_idf(tokenize_term(term))

def tf_idf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_tf_idf(doc_id, tokenize_term(term))

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(tokenize_term(term))

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, tokenize_term(term), k1, b)
