import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_CHUNK_SIZE_SEMANTIC,
    DEFAULT_OVERLAP_SEMANTIC,
    MetaData,
    ChunkScore,
    Movie,
    SemanticSearchResult,
    ChunkedSemanticSearchResult,
    load_movies,
)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2, out=None)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

class SemanticSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}
        self.embeddings_path = CACHE_DIR / "movie_embeddings.npy"

    def _populate(self, documents: list[Movie]) -> None:
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document

    def build_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self._populate(documents)

        texts = [f"{document["title"]}: {document["description"]}" for document in documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)

        CACHE_DIR.mkdir(exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self._populate(documents)

        if self.embeddings_path.exists():
            embeddings = np.load(self.embeddings_path)
            if len(embeddings) == len(documents):
                self.embeddings = embeddings
                return embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[SemanticSearchResult]:
        if limit <= 0:
            return []

        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call load_or_create_embeddings first.")

        query_embedding = self.generate_embedding(query)
        results: list[SemanticSearchResult] = [
            {
                "id": document["id"],
                "title": document["title"],
                "document": document["description"],
                "score": cosine_similarity(query_embedding, embedding),
            }
            for document, embedding in zip(self.documents, self.embeddings)
        ]
        results.sort(key=lambda result: result["score"], reverse=True)

        return results[:limit]

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text:
            raise ValueError("Input text cannot be empty.")
        if text.isspace():
            raise ValueError("Input text cannot be only whitespace.")
        embedding = self.model.encode([text])
        return embedding[0]

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[MetaData] = []
        self.chunk_embeddings_path = CACHE_DIR / "chunk_embeddings.npy"
        self.chunk_metadata_path = CACHE_DIR / "chunk_metadata.json"

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self._populate(documents)

        chunks: list[str] = []
        metadata: list[MetaData] = []

        for movie_idx, document in enumerate(documents):
            if not document["description"]:
                continue
            description_chunks = semantic_chunking_command(
                document["description"], DEFAULT_CHUNK_SIZE_SEMANTIC, DEFAULT_OVERLAP_SEMANTIC
            )
            for chunk_idx, chunk in enumerate(description_chunks):
                metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(description_chunks),
                    }
                )
                chunks.append(chunk)

        self.chunk_embeddings = self.model.encode(chunks)
        self.chunk_metadata = metadata

        CACHE_DIR.mkdir(exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": metadata, "total_chunks": len(chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self._populate(documents)

        if self.chunk_embeddings_path.exists() and self.chunk_metadata_path.exists():
            embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path) as f:
                metadata = json.load(f)
            if len(embeddings) == metadata["total_chunks"]:
                self.chunk_embeddings = embeddings
                self.chunk_metadata = metadata["chunks"]
                return embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int) -> list[ChunkedSemanticSearchResult]:
        if limit <= 0:
            return []

        if self.chunk_embeddings is None or not self.chunk_metadata:
            raise ValueError("No chunk embeddings loaded. Call load_or_create_chunk_embeddings first.")
        if self.documents is None:
            raise ValueError("No documents loaded. Call load_or_create_chunk_embeddings first.")

        query_embedding = self.generate_embedding(query)

        chunk_scores: list[ChunkScore] = []
        for meta, embedding in zip(self.chunk_metadata, self.chunk_embeddings):
            similarity = cosine_similarity(query_embedding, embedding)
            chunk_scores.append(
                {
                    "chunk_idx": meta["chunk_idx"],
                    "movie_idx": meta["movie_idx"],
                    "score": similarity,
                }
            )

        best_scores: dict[int, float] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            if movie_idx not in best_scores or score > best_scores[movie_idx]:
                best_scores[movie_idx] = score
        filtered_scores = sorted(best_scores.items(), key=lambda item: item[1], reverse=True)[:limit]

        results: list[ChunkedSemanticSearchResult] = []
        for movie_idx, score in filtered_scores:
            movie = self.documents[movie_idx]
            results.append(
                {
                    "id": movie["id"],
                    "title": movie["title"],
                    "document": movie["description"][:100],
                    "score": score,
                }
            )

        return results


def verify_model_command() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded:        {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text_command(text: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text:                            {text}")
    print(f"First 3 dimensions of embedding: {embedding[:3]}")
    print(f"Dimensions:                      {embedding.shape[0]}")

def embed_query_text_command(query: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Text:                            {query}")
    print(f"First 3 dimensions of embedding: {embedding[:3]}")
    print(f"Dimensions:                      {embedding.shape[0]}")

def verify_embeddings_command() -> None:
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(list(movies.values()))
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_chunks_command() -> None:
    chunked_search = ChunkedSemanticSearch()
    movies = load_movies()
    embeddings = chunked_search.load_or_create_chunk_embeddings(list(movies.values()))
    print(f"Generated {len(embeddings)} chunked embeddings")

def semantic_search_command(query: str, limit: int) -> None:
    semantic_search_engine = SemanticSearch()
    movies = load_movies()
    semantic_search_engine.load_or_create_embeddings(list(movies.values()))
    results = semantic_search_engine.search(query, limit)

    for rank, result in enumerate(results, start=1):
        print(f"{rank}: {result["title"]}: {result["document"]}")

def search_chunked_command(query: str, limit: int) -> None:
    chunked_search_engine = ChunkedSemanticSearch()
    movies = load_movies()
    chunked_search_engine.load_or_create_chunk_embeddings(list(movies.values()))
    results = chunked_search_engine.search_chunks(query, limit)

    for i, result in enumerate(results, start=1):
        print(f"\n{i}. {result["title"]} (score: {result["score"]:.4f})")
        print(f"   {result["document"]}...")

def fixed_size_chunking_command(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be a positive integer.")
    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be a non-negative integer less than chunk size.")

    words = text.split()

    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks

def semantic_chunking_command(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    if max_chunk_size <= 0:
        raise ValueError("Max chunk size must be a positive integer.")
    if not 0 <= overlap < max_chunk_size:
        raise ValueError("Overlap must be a non-negative integer less than max chunk size.")

    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        stripped_chunk = " ".join(chunk_sentences).strip()
        if stripped_chunk:
            chunks.append(stripped_chunk)
        i += max_chunk_size - overlap

    return chunks
