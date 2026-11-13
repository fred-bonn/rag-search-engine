import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .search_utils import (
    SEARCH_LOG_PATH,
    SEARCH_LOG_SCHEMA_VERSION,
    LoggedResult,
)


RESULT_SCORE_FIELDS = (
    "score",
    "hybrid_score",
    "bm25_score",
    "semantic_score",
    "rrf_score",
    "bm25_rank",
    "semantic_rank",
    "rerank_score",
    "cross_encoder_score",
)


def new_search_id() -> str:
    return uuid4().hex


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    return str(value)


def _append(event: dict[str, Any]) -> None:
    try:
        SEARCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SEARCH_LOG_PATH.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
    except OSError as e:
        print(f"Warning: could not write search log: {e}", file=sys.stderr)


def _base_event(event: str, search_id: str) -> dict[str, Any]:
    return {
        "schema_version": SEARCH_LOG_SCHEMA_VERSION,
        "timestamp": _timestamp(),
        "event": event,
        "search_id": search_id,
    }


def _logged_result(result: Mapping[str, Any], rank: int) -> LoggedResult:
    logged = LoggedResult(rank=rank, id=result["id"], title=result["title"])
    for field in RESULT_SCORE_FIELDS:
        if field in result:
            logged[field] = result[field]
    return logged


def _logged_results(results: Sequence[Mapping[str, Any]]) -> list[LoggedResult]:
    return [_logged_result(result, rank) for rank, result in enumerate(results, start=1)]


def log_search(
    *,
    search_id: str,
    query: str,
    engine: str,
    search_type: str,
    limit: int,
    results: Sequence[Mapping[str, Any]],
    enhancement: str | None = None,
    enhanced_query: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> None:
    event = _base_event("search", search_id)
    event.update(
        {
            "query": query,
            "engine": engine,
            "search_type": search_type,
            "limit": limit,
        }
    )
    if enhancement is not None:
        event["enhancement"] = enhancement
    if enhanced_query is not None:
        event["enhanced_query"] = enhanced_query
    if params:
        event["params"] = dict(params)
    event["results"] = _logged_results(results)
    _append(event)


def log_rerank(
    *,
    search_id: str,
    rerank_method: str,
    results: Sequence[Mapping[str, Any]],
) -> None:
    event = _base_event("rerank", search_id)
    event["rerank_method"] = rerank_method
    event["results"] = _logged_results(results)
    _append(event)


def log_error(
    *,
    search_id: str,
    query: str,
    engine: str,
    search_type: str,
    error: Exception,
) -> None:
    event = _base_event("error", search_id)
    event.update(
        {
            "query": query,
            "engine": engine,
            "search_type": search_type,
            "error": str(error),
            "error_type": type(error).__name__,
        }
    )
    _append(event)
