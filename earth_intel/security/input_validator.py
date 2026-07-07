MAX_QUERY_LENGTH = 5000


class InvalidInputException(Exception):
    pass


def validate_input(query: str):

    if not query:
        raise InvalidInputException(
            "Empty query."
        )

    if not query.strip():
        raise InvalidInputException(
            "Blank query."
        )

    if len(query) > MAX_QUERY_LENGTH:
        raise InvalidInputException(
            "Query too long."
        )

    return query.strip()