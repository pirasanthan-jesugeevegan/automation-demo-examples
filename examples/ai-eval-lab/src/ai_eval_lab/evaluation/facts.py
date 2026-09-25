"""Pull the checkable facts (who, how much, when) out of a sentence."""

import re

_ENTITY = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_NUMBER = re.compile(r"£?\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

# Capitalised only because they open a sentence, so they are not entities.
_SENTENCE_STARTERS = frozenset({"a", "an", "and", "in", "on", "the", "this", "that", "it"})


def extract_entities(text: str) -> set[str]:
    entities: set[str] = set()

    for match in _ENTITY.findall(text):
        words = match.split()

        while words and words[0].lower() in _SENTENCE_STARTERS:
            words.pop(0)

        if words and len(" ".join(words)) > 1:
            entities.add(" ".join(words))

    return entities


def extract_years(text: str) -> set[str]:
    return set(_YEAR.findall(text))


def extract_numbers(text: str) -> set[str]:
    """Amounts and counts. Years are excluded, they are checked separately."""
    years = extract_years(text)

    return {
        " ".join(number.split()) for number in _NUMBER.findall(text.lower()) if number not in years
    }
