from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (
    HybridSearchResult,
    Movie,
    ScoredSearchResult,
    load_movies,
)


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents

        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.inverted_index = InvertedIndex()
        if not all(path.exists() for path in self.inverted_index._get_cache_paths()):
            raise FileNotFoundError("Keyword Search cache files not found. Build first.")
        self.inverted_index.load()

    def _bm25_search(self, query: str, limit: int) -> list[ScoredSearchResult]:
        return self.inverted_index.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[HybridSearchResult]:
        if limit <= 0:
            return []

        documents_by_id = {document["id"]: document for document in self.documents}

        bm25_results = self._bm25_search(query, limit*500)
        semantic_results = self.semantic_search.search_chunks(query, limit*500)

        bm25_scores = normalize_command([result["score"] for result in bm25_results])
        semantic_scores = normalize_command([result["score"] for result in semantic_results])

        combined: dict[int, HybridSearchResult] = {}
        for result, score in zip(bm25_results, bm25_scores):
            movie = documents_by_id[result["id"]]
            combined[movie["id"]] = {
                "id": movie["id"],
                "title": movie["title"],
                "document": movie["description"][:100],
                "hybrid_score": 0.0,
                "bm25_score": score,
                "semantic_score": 0.0,
            }

        for result, score in zip(semantic_results, semantic_scores):
            movie = documents_by_id[result["id"]]
            if movie["id"] in combined:
                combined[movie["id"]]["semantic_score"] = score
            else:
                combined[movie["id"]] = {
                    "id": movie["id"],
                    "title": movie["title"],
                    "document": movie["description"][:100],
                    "hybrid_score": 0.0,
                    "bm25_score": 0.0,
                    "semantic_score": score,
                }

        for result in combined.values():
            result["hybrid_score"] = (
                alpha * result["bm25_score"] + (1 - alpha) * result["semantic_score"]
            )

        ranked = sorted(combined.values(), key=lambda result: result["hybrid_score"], reverse=True)
        return ranked[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_command(numbers: list[float]) -> list[float]:
    if not numbers:
        return []

    minimum = min(numbers)
    maximum = max(numbers)
    score_range = maximum - minimum

    if score_range == 0:
        return [1.0 for _ in numbers]

    return [(number - minimum) / score_range for number in numbers]

def weighted_search_command(query: str, alpha: float, limit: int) -> list[HybridSearchResult]:
    if limit <= 0:
        return []

    hybrid_search = HybridSearch(list(load_movies().values()))
    return hybrid_search.weighted_search(query, alpha, limit)
