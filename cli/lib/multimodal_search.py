from typing import cast

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self.model: SentenceTransformer = SentenceTransformer(model_name)

    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path)
        embedding = cast(np.ndarray, self.model.encode([image]))
        return embedding[0]


def verify_image_embedding(image_path: str) -> np.ndarray:
    multimodal_search = MultimodalSearch()
    return multimodal_search.embed_image(image_path)
