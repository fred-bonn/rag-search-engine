import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import (
    CACHE_DIR,
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

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text:
            raise ValueError("Input text cannot be empty.")
        if text.isspace():
            raise ValueError("Input text cannot be only whitespace.")
        embedding = self.model.encode([text])
        return embedding[0]

def verify_model() -> bool:
    try:
        semantic_search = SemanticSearch()
        print(f"Model loaded: {semantic_search.model}")
        print(f"Max sequence length: {semantic_search.model.max_seq_length}")
        return True
    except Exception as e:
        print(f"Error initializing SemanticSearch model: {e}")
    return False

def embed_text(text: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text:                            {text}")
    print(f"First 3 dimensions of embedding: {embedding[:3]}")
    print(f"Dimensions:                      {embedding.shape[0]}")

def verify_embeddings() -> None:
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(list(movies.values()))
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")