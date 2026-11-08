from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (
    HybridSearchResult,
    Movie,
    RRFSearchResult,
    ScoredSearchResult,
    load_movies,
)


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)
    score_range = maximum - minimum

    if score_range == 0:
        return [1.0 for _ in scores]

    return [(number - minimum) / score_range for number in scores]

def entry(movie: Movie) -> HybridSearchResult:
    return {
        "id": movie["id"],
        "title": movie["title"],
        "document": movie["description"],
        "hybrid_score": 0.0,
        "bm25_score": 0.0,
        "semantic_score": 0.0,
    }

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_entry(movie: Movie) -> RRFSearchResult:
    return {
        "id": movie["id"],
        "title": movie["title"],
        "document": movie["description"],
        "rrf_score": 0.0,
        "bm25_rank": None,
        "semantic_rank": None,
    }

def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)

class HybridSearch:
    def __init__(self, documents: list[Movie] | None = None) -> None:
        self.documents = documents
        self.semantic_search: ChunkedSemanticSearch | None = None
        self.inverted_index: InvertedIndex | None = None

    def _ensure_loaded(self) -> None:
        if self.semantic_search is not None and self.inverted_index is not None:
            return

        documents = self.documents
        if documents is None:
            documents = list(load_movies().values())
            self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.inverted_index = InvertedIndex()
        if not all(path.exists() for path in self.inverted_index._get_cache_paths()):
            raise FileNotFoundError("Keyword Search cache files not found. Build first.")
        self.inverted_index.load()

    def _bm25_search(self, query: str, limit: int) -> list[ScoredSearchResult]:
        assert self.inverted_index is not None
        return self.inverted_index.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[HybridSearchResult]:
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be a floating point number between 0 and 1")
        if limit <= 0:
            return []

        self._ensure_loaded()
        assert self.semantic_search is not None and self.documents is not None

        documents_by_id = {document["id"]: document for document in self.documents}

        bm25_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)

        bm25_scores = normalize_command([result["score"] for result in bm25_results])
        semantic_scores = normalize_command([result["score"] for result in semantic_results])

        combined: dict[int, HybridSearchResult] = {}
        for result, score in zip(bm25_results, bm25_scores):
            movie = documents_by_id[result["id"]]
            combined.setdefault(movie["id"], entry(movie))["bm25_score"] = score

        for result, score in zip(semantic_results, semantic_scores):
            movie = documents_by_id[result["id"]]
            combined.setdefault(movie["id"], entry(movie))["semantic_score"] = score

        for result in combined.values():
            result["hybrid_score"] = hybrid_score(result["bm25_score"], result["semantic_score"], alpha)

        ranked = sorted(combined.values(), key=lambda result: result["hybrid_score"], reverse=True)
        return ranked[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[RRFSearchResult]:
        if k < 0:
            raise ValueError("The k value in RRF search must not be negative")
        if limit <= 0:
            return []

        self._ensure_loaded()
        assert self.semantic_search is not None and self.documents is not None

        documents_by_id = {document["id"]: document for document in self.documents}

        bm25_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)

        combined: dict[int, RRFSearchResult] = {}
        for rank, result in enumerate(bm25_results, start=1):
            movie = documents_by_id[result["id"]]
            combined_result = combined.setdefault(movie["id"], rrf_entry(movie))
            combined_result["bm25_rank"] = rank
            combined_result["rrf_score"] += rrf_score(rank, k)

        for rank, result in enumerate(semantic_results, start=1):
            movie = documents_by_id[result["id"]]
            combined_result = combined.setdefault(movie["id"], rrf_entry(movie))
            combined_result["semantic_rank"] = rank
            combined_result["rrf_score"] += rrf_score(rank, k)

        ranked = sorted(combined.values(), key=lambda result: result["rrf_score"], reverse=True)
        return ranked[:limit]

def normalize_command(numbers: list[float]) -> list[float]:
    if not numbers:
        return []
    return normalize(numbers)

def weighted_search_command(query: str, alpha: float, limit: int) -> list[HybridSearchResult]:
    hybrid_search = HybridSearch()
    return hybrid_search.weighted_search(query, alpha, limit)

def rrf_search_command(query: str, k: int, limit: int) -> list[RRFSearchResult]:
    hybrid_search = HybridSearch()
    return hybrid_search.rrf_search(query, k, limit)
