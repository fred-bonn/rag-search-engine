import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    Movie,
    load_movies,
)


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings: np.ndarray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}
        self.embeddings_path = CACHE_DIR / "movie_embeddings.npy"

    def __populate(self, documents: list[Movie]) -> None:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

    def build_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.__populate(documents)

        texts = [f"{document["title"]}: {document["description"]}" for document in documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        CACHE_DIR.mkdir(exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.__populate(documents)

        if self.embeddings_path.exists():
            embeddings = np.load(self.embeddings_path)
            if len(embeddings) == len(documents):
                self.embeddings = embeddings
                return embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[tuple[Movie, float]]:
        if limit <= 0:
            return []

        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call load_or_create_embeddings first.")

        query_embedding = self.generate_embedding(query)
        results = [
            (document, cosine_similarity(query_embedding, embedding))
            for document, embedding in zip(self.documents, self.embeddings)
        ]
        results.sort(key=lambda result: result[1], reverse=True)

        return results[:limit]

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text:
            raise ValueError("Input text cannot be empty.")
        if text.isspace():
            raise ValueError("Input text cannot be only whitespace.")
        embedding = self.model.encode([text])
        return embedding[0]

def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded:        {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text:                            {text}")
    print(f"First 3 dimensions of embedding: {embedding[:3]}")
    print(f"Dimensions:                      {embedding.shape[0]}")

def embed_query_text(query: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Text:                            {query}")
    print(f"First 3 dimensions of embedding: {embedding[:3]}")
    print(f"Dimensions:                      {embedding.shape[0]}")

def verify_embeddings() -> None:
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(list(movies.values()))
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    semantic_search_engine = SemanticSearch()
    movies = load_movies()
    semantic_search_engine.load_or_create_embeddings(list(movies.values()))
    results = semantic_search_engine.search(query, limit)

    for rank, (movie, _) in enumerate(results, start=1):
        print(f"{rank}: {movie["title"]}: {movie["description"]}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)