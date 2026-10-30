from search_utils import load_movies, Movie
import string as string

DEFAULT_SEARCH_LIMIT = 5

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    query = process(query)
    movies = load_movies()

    results: list[Movie] = []
    for movie in movies.values():
        title = process(movie["title"])
        if query in title:
            results.append(movie)
            if len(results) == limit:
                break

    return results

def process(input: str) -> str:
    input = input.lower()
    translator = str.maketrans("", "", string.punctuation)
    input = input.translate(translator)
    return input.lower()