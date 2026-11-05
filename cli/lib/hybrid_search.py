
def normalize_command(numbers: list[float]) -> list[float]:
    if not numbers:
        return []

    minimum = min(numbers)
    maximum = max(numbers)
    score_range = maximum - minimum

    if score_range == 0:
        return [1.0 for _ in numbers]

    return [(number - minimum) / score_range for number in numbers]