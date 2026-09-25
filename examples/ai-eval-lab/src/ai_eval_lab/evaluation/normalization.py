def normalize_claim(claim: str) -> str:
    """Lower-case, collapse whitespace and drop a trailing full stop.

    Two claims that differ only in those ways are the same claim.
    """
    return " ".join(claim.lower().split()).rstrip(".")
