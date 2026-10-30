import json
from pathlib import Path
from typing import TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"


class Movie(TypedDict):
    id: int
    title: str
    description: str

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