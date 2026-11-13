import json
from pathlib import Path
from typing import NotRequired, TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
GOLDEN_DATASET_PATH = PROJECT_ROOT / "data" / "golden_dataset.json"
CACHE_DIR = PROJECT_ROOT / "cache"
LOG_DIR = PROJECT_ROOT / "logs"
SEARCH_LOG_PATH = LOG_DIR / "search.jsonl"
SEARCH_LOG_SCHEMA_VERSION = 1

DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_SIZE_SEMANTIC = 4
DEFAULT_OVERLAP_CLI = 0
DEFAULT_OVERLAP_SEMANTIC = 1

BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_ALPHA = 0.5
DEFAULT_RRF_K = 60

RERANK_LIMIT_MULTIPLIER = 5
RERANK_DELAY_SECONDS = 3

SEMANTIC_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"

class Movie(TypedDict):
    id: int
    title: str
    description: str

class MetaData(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int

class ChunkScore(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float

class KeywordSearchResult(TypedDict):
    id: int
    title: str
    document: str
    rerank_score: NotRequired[int | None]
    cross_encoder_score: NotRequired[float | None]

class ScoredSearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    rerank_score: NotRequired[int | None]
    cross_encoder_score: NotRequired[float | None]

class HybridSearchResult(TypedDict):
    id: int
    title: str
    document: str
    hybrid_score: float
    bm25_score: float
    semantic_score: float
    rerank_score: NotRequired[int | None]
    cross_encoder_score: NotRequired[float | None]

class RRFSearchResult(TypedDict):
    id: int
    title: str
    document: str
    rrf_score: float
    bm25_rank: int | None
    semantic_rank: int | None
    rerank_score: NotRequired[int | None]
    cross_encoder_score: NotRequired[float | None]

class TestCase(TypedDict):
    query: str
    relevant_docs: list[str]

class EvaluationMetrics(TypedDict):
    precision: float
    recall: float
    f1: float

class LoggedResult(TypedDict):
    rank: int
    id: int
    title: str
    score: NotRequired[float]
    hybrid_score: NotRequired[float]
    bm25_score: NotRequired[float]
    semantic_score: NotRequired[float]
    rrf_score: NotRequired[float]
    bm25_rank: NotRequired[int | None]
    semantic_rank: NotRequired[int | None]
    rerank_score: NotRequired[int | None]
    cross_encoder_score: NotRequired[float | None]

def load_movies() -> dict[int, Movie]:
    with DATA_PATH.open("r") as f:
        raw_data = json.load(f)["movies"]
        movies = {
            int(m["id"]): Movie(
                id=int(m["id"]),
                title=m["title"],
                description=m["description"],
            )
            for m in raw_data
        }
    return movies