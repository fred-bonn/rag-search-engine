import pytest

import keyword_search


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Grandview, U.S.A.", "grandview usa"),
        ("Kaakha..Kaakha: The Police", "kaakhakaakha the police"),
        ("  Mixed CASE!  ", "  mixed case  "),
        ("", ""),
        ("!@#$%^&*()", ""),
        ("Café—NOIR", "café—noir"),
    ],
)
def test_process(text: str, expected: str) -> None:
    assert keyword_search.process(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("one two", ["one", "two"]),
        ("  spaced   words  ", ["spaced", "words"]),
        ("line one\nline two\tend", ["line", "one", "line", "two", "end"]),
        ("", []),
        (" \n\t ", []),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert keyword_search.tokenize(text) == expected


@pytest.mark.parametrize(
    ("query_tokens", "title_tokens", "expected"),
    [
        (["the"], ["theater"], True),
        (["view", "grand"], ["grandview"], True),
        (["absent"], ["grandview"], False),
        ([], ["grandview"], False),
        (["grandview"], [], False),
        ([], [], False),
    ],
)
def test_has_matching_token(
    query_tokens: list[str], title_tokens: list[str], expected: bool
) -> None:
    assert keyword_search.has_matching_token(query_tokens, title_tokens) is expected


@pytest.mark.parametrize(
    ("query", "limit", "expected_ids"),
    [
        ("the the", 5, [1, 2, 3, 4, 5]),
        ("theater", 5, [1, 2, 4, 6]),
        ("absent", 5, []),
        ("", 5, []),
        ("the", 0, []),
        ("the", -1, []),
        ("the", 2, [1, 2]),
        ("the", None, [1, 2, 3, 4, 5]),
    ],
)
def test_search_command(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    limit: int | None,
    expected_ids: list[int],
) -> None:
    movies = {
        1: {"id": 1, "title": "The Theater", "description": ""},
        2: {"id": 2, "title": "Another Theater", "description": ""},
        3: {"id": 3, "title": "The Avengers", "description": ""},
        4: {"id": 4, "title": "Theater Four", "description": ""},
        5: {"id": 5, "title": "The Fifth", "description": ""},
        6: {"id": 6, "title": "Theater Six", "description": ""},
    }
    monkeypatch.setattr(keyword_search, "load_movies", lambda: movies)
    monkeypatch.setattr(keyword_search, "load_stopwords", lambda: [])
    monkeypatch.setattr(keyword_search.SearchContext, "instance", None)

    if limit is None:
        results = keyword_search.search_command(query)
    else:
        results = keyword_search.search_command(query, limit)

    result_ids = [movie["id"] for movie in results]
    assert result_ids == expected_ids
    assert len(result_ids) == len(set(result_ids))
