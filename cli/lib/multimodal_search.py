from typing import cast

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from .search_utils import DEFAULT_SEARCH_LIMIT, ImageSearchResult, Movie, load_movies
from .semantic_search import cosine_similarity


class MultimodalSearch:
    def __init__(self, documents: list[Movie] | None = None, model_name: str = "clip-ViT-B-32") -> None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        if documents is None:
            documents = list(load_movies().values())
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        embedding = cast(np.ndarray, self.model.encode([image]))
        return embedding[0]

    def search_with_image(self, image_path: str, limit: int = 5) -> list[ImageSearchResult]:
        image_embedding = self.embed_image(image_path)

        results: list[ImageSearchResult] = []
        for document, text_embedding in zip(self.documents, self.text_embeddings):
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "document": document["description"],
                    "score": cosine_similarity(image_embedding, text_embedding),
                }
            )

        results.sort(key=lambda result: result["score"], reverse=True)
        return results[:limit]


def verify_image_embedding(image_path: str) -> np.ndarray:
    multimodal_search = MultimodalSearch()
    return multimodal_search.embed_image(image_path)


def image_search(image_path: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[ImageSearchResult]:
    multimodal_search = MultimodalSearch()
    return multimodal_search.search_with_image(image_path, limit)
