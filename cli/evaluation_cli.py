import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

from lib.hybrid_search import HybridSearch
from lib.search_utils import (
    DEFAULT_RRF_K,
    RRFSearchResult,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DATASET_PATH = PROJECT_ROOT / "data" / "golden_dataset.json"


class TestCase(TypedDict):
    query: str
    relevant_docs: list[str]


def load_golden_dataset() -> list[TestCase]:
    try:
        with GOLDEN_DATASET_PATH.open("r") as f:
            raw_data = json.load(f)["test_cases"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Golden dataset not found: {GOLDEN_DATASET_PATH}") from None
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Invalid golden dataset at {GOLDEN_DATASET_PATH}: {e}") from None

    test_cases = [
        TestCase(
            query=case["query"],
            relevant_docs=case["relevant_docs"],
        )
        for case in raw_data
    ]
    return test_cases


def run_rrf_searches(test_cases: list[TestCase], limit: int) -> list[tuple[TestCase, list[RRFSearchResult]]]:
    hybrid_search = HybridSearch()
    return [
        (test_case, hybrid_search.rrf_search(test_case["query"], DEFAULT_RRF_K, limit))
        for test_case in test_cases
    ]


def precision_at_k(relevant_docs: list[str], results: list[RRFSearchResult]) -> float:
    if not results:
        return 0.0

    relevant = set(relevant_docs)
    retrieved = [result["title"] for result in results]
    relevant_retrieved = sum(1 for title in retrieved if title in relevant)
    return relevant_retrieved / len(retrieved)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    try:
        test_cases = load_golden_dataset()
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1

    print(f"k={limit}")
    print()

    for test_case, results in run_rrf_searches(test_cases, limit):
        retrieved = [result["title"] for result in results]
        precision = precision_at_k(test_case["relevant_docs"], results)
        print(f"- Query: {test_case['query']}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(test_case['relevant_docs'])}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
