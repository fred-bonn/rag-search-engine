from search_utils import load_movies, Movie

DEFAULT_SEARCH_LIMIT = 5

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    movies = load_movies()
    results: list[Movie] = []
    for movie in movies.values():
        if query.lower() in movie["title"].lower():
            results.append(movie)
            if len(results) == limit:
                break
    return results