from ai_eval_lab.evaluation.deterministic import evaluate
from ai_eval_lab.models.evaluation_case import (
    EvaluationCase,
    ExpectedFinding,
)
from ai_eval_lab.models.investigation import (
    Entity,
    Finding,
    Investigation,
    SourceDocument,
)


def create_source() -> SourceDocument:
    return SourceDocument(
        id="source-001",
        title="Regulatory Decision",
        source_type="regulator",
        content=("Acme Holdings received a £2 million penalty in 2024."),
    )


def create_expected_case() -> EvaluationCase:
    return EvaluationCase(
        id="CASE-001",
        description="Straightforward regulatory finding",
        entity=Entity(
            name="Acme Holdings",
            entity_type="company",
        ),
        expected_findings=[
            ExpectedFinding(
                claim=("Acme Holdings received a £2 million regulatory penalty in 2024."),
                source_ids=["source-001"],
            )
        ],
    )


def test_correct_investigation_passes() -> None:
    expected = create_expected_case()

    actual = Investigation(
        entity=expected.entity,
        sources=[create_source()],
        findings=[
            Finding(
                claim=("Acme Holdings received a £2 million regulatory penalty in 2024."),
                source_ids=["source-001"],
                confidence=0.95,
            )
        ],
    )

    result = evaluate(expected, actual)

    assert result.passed is True
    assert result.metrics.source_validity == 1.0
    assert result.metrics.finding_coverage == 1.0
    assert result.metrics.unexpected_finding_rate == 0.0
    assert result.errors == []
    assert result.invalid_source_ids == []


def test_incorrect_claim_fails() -> None:
    expected = create_expected_case()

    actual = Investigation(
        entity=expected.entity,
        sources=[create_source()],
        findings=[
            Finding(
                claim=("Acme Holdings received a £20 million regulatory penalty in 2024."),
                source_ids=["source-001"],
                confidence=0.95,
            )
        ],
    )

    result = evaluate(expected, actual)

    assert result.passed is False
    assert result.metrics.finding_coverage == 0.0
    assert result.metrics.unexpected_finding_rate == 1.0


def test_invalid_source_reference_fails() -> None:
    expected = create_expected_case()

    actual = Investigation(
        entity=expected.entity,
        sources=[create_source()],
        findings=[
            Finding(
                claim=("Acme Holdings received a £2 million regulatory penalty in 2024."),
                source_ids=["source-999"],
                confidence=0.95,
            )
        ],
    )

    result = evaluate(expected, actual)

    assert result.passed is False
    assert result.metrics.source_validity == 0.0
    assert any("invalid source references" in error for error in result.errors)
    assert result.invalid_source_ids == ["source-999"]


def test_unexpected_finding_fails() -> None:
    expected = create_expected_case()

    actual = Investigation(
        entity=expected.entity,
        sources=[create_source()],
        findings=[
            Finding(
                claim=("Acme Holdings received a £2 million regulatory penalty in 2024."),
                source_ids=["source-001"],
                confidence=0.95,
            ),
            Finding(
                claim="Acme Holdings committed fraud.",
                source_ids=["source-001"],
                confidence=0.90,
            ),
        ],
    )

    result = evaluate(expected, actual)

    assert result.passed is False
    assert result.metrics.finding_coverage == 1.0
    assert result.metrics.unexpected_finding_rate == 0.5
