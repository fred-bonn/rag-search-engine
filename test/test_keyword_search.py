import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))

from keyword_search import has_matching_token, process, search_command, tokenize


class KeywordSearchHelperTests(unittest.TestCase):
    def test_process(self) -> None:
        cases = [
            ("Grandview, U.S.A.", "grandview usa"),
            ("Kaakha..Kaakha: The Police", "kaakhakaakha the police"),
            ("  Mixed CASE!  ", "  mixed case  "),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(process(text), expected)

    def test_tokenize(self) -> None:
        cases = [
            ("one two", ["one", "two"]),
            ("  spaced   words  ", ["spaced", "words"]),
            ("", []),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(tokenize(text), expected)

    def test_has_matching_token(self) -> None:
        cases = [
            (["the"], ["theater"], True),
            (["view", "grand"], ["grandview"], True),
            (["absent"], ["grandview"], False),
            ([], ["grandview"], False),
            (["grandview"], [], False),
        ]
        for query_tokens, title_tokens, expected in cases:
            with self.subTest(query_tokens=query_tokens, title_tokens=title_tokens):
                self.assertEqual(
                    has_matching_token(query_tokens, title_tokens), expected
                )

    def test_search_command_returns_each_movie_once(self) -> None:
        movies = {
            1: {"id": 1, "title": "The Theater", "description": ""},
            2: {"id": 2, "title": "Another Theater", "description": ""},
        }
        with patch("keyword_search.load_movies", return_value=movies):
            results = search_command("the the")

        self.assertEqual([movie["id"] for movie in results], [1, 2])


if __name__ == "__main__":
    unittest.main()
