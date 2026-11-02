import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

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
    print(f"Text: {text}")
    print(f"First 3 dimensions of embedding: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")