from ai_eval_lab.evaluation.facts import (
    extract_entities,
    extract_numbers,
    extract_years,
)


def test_extract_entities() -> None:
    text = "Acme Holdings received a penalty."

    assert "Acme Holdings" in extract_entities(text)


def test_extract_numbers() -> None:
    text = "Acme Holdings received a £2 million penalty."

    assert "£2 million" in extract_numbers(text)


def test_extract_years() -> None:
    text = "The regulator announced the penalty in 2024."

    assert "2024" in extract_years(text)


def test_sentence_opening_words_are_not_entities() -> None:
    text = "In 2024, the regulator fined Acme Holdings. The Financial News reported it."

    entities = extract_entities(text)

    assert "In" not in entities
    assert "The Financial News" not in entities
    assert {"Acme Holdings", "Financial News"} <= entities


def test_years_are_not_counted_as_numbers() -> None:
    text = "A £2 million penalty in 2024 and 50 offices."

    assert extract_numbers(text) == {"£2 million", "50"}
    assert extract_years(text) == {"2024"}
