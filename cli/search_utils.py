import json
from typing import TypedDict


DATA_FILE_PATH = "data/movies.json"


class Movie(TypedDict):
    id: int
    title: str
    description: str


def title_match(movie: Movie, query: str) -> bool:
    return query.lower() in movie["title"].lower()

def load_data() -> dict[int, Movie]:
    with open(DATA_FILE_PATH, "r") as f:
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