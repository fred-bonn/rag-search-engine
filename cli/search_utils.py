import json
from pathlib import Path
from typing import TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
TEST_DATA_PATH = PROJECT_ROOT / "test" / "movies.json"

class Movie(TypedDict):
    id: int
    title: str
    description: str

def load_movies(testing: bool = False) -> dict[int, Movie]:
    data_path = TEST_DATA_PATH if testing else DATA_PATH
    with data_path.open("r") as f:
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