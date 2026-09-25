from ai_eval_lab.evaluation.groundedness import evaluate_groundedness
from ai_eval_lab.models.investigation import (
    Finding,
    SourceDocument,
)


def create_source() -> SourceDocument:
    return SourceDocument(
        id="source-001",
        title="Regulatory Decision",
        source_type="regulator",
        content=("In 2024, the regulator announced a £2 million penalty against Acme Holdings."),
    )


def test_supported_claim_has_high_groundedness() -> None:
    finding = Finding(
        claim=("Acme Holdings received a £2 million penalty in 2024."),
        source_ids=["source-001"],
        confidence=0.95,
    )

    result = evaluate_groundedness(
        finding,
        [create_source()],
    )

    assert result.grounded is True
    assert result.score >= 0.6
    assert result.supporting_source_ids == ["source-001"]


def test_claim_without_valid_source_fails() -> None:
    finding = Finding(
        claim="Acme Holdings received a £2 million penalty.",
        source_ids=["source-999"],
        confidence=0.95,
    )

    result = evaluate_groundedness(
        finding,
        [create_source()],
    )

    assert result.grounded is False
    assert result.score == 0.0
    assert result.supporting_source_ids == []


def test_unrelated_claim_has_low_groundedness() -> None:
    finding = Finding(
        claim="Acme Holdings opened 50 new offices in Canada.",
        source_ids=["source-001"],
        confidence=0.95,
    )

    result = evaluate_groundedness(
        finding,
        [create_source()],
    )

    assert result.grounded is False


def test_materially_different_amount_should_not_be_considered_grounded() -> None:
    finding = Finding(
        claim=("Acme Holdings received a £20 million penalty in 2024."),
        source_ids=["source-001"],
        confidence=0.95,
    )

    result = evaluate_groundedness(
        finding,
        [create_source()],
    )

    assert result.grounded is False


def test_claim_sharing_nothing_with_its_source_is_ungrounded_not_an_error() -> None:
    finding = Finding(
        claim="Zebras migrate seasonally.",
        source_ids=["source-001"],
        confidence=0.9,
    )

    result = evaluate_groundedness(finding, [create_source()])

    assert result.grounded is False
    assert result.score == 0.0
    assert result.explanation


def test_claim_made_only_of_stop_words_is_ungrounded_not_an_error() -> None:
    finding = Finding(claim="The", source_ids=["source-001"], confidence=0.9)

    result = evaluate_groundedness(finding, [create_source()])

    assert result.grounded is False
    assert result.explanation


def test_wrong_year_is_not_grounded() -> None:
    finding = Finding(
        claim="Acme Holdings received a £2 million penalty in 2019.",
        source_ids=["source-001"],
        confidence=0.95,
    )

    result = evaluate_groundedness(finding, [create_source()])

    assert result.grounded is False
    assert "date mismatch" in result.explanation
