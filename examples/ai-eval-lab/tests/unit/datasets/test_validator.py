from ai_eval_lab.datasets.validator import validate_source_references
from ai_eval_lab.models.evaluation_case import (
    EvaluationCase,
    ExpectedFinding,
)
from ai_eval_lab.models.investigation import (
    Entity,
    SourceDocument,
)


def test_all_source_references_are_valid() -> None:
    cases = [
        EvaluationCase(
            id="CASE-001",
            description="Test case",
            entity=Entity(
                name="Acme Holdings",
                entity_type="company",
            ),
            expected_findings=[
                ExpectedFinding(
                    claim="Acme received a penalty.",
                    source_ids=["source-001"],
                )
            ],
        )
    ]

    sources = [
        SourceDocument(
            id="source-001",
            title="Regulatory Decision",
            source_type="regulator",
            content="Acme received a penalty.",
        )
    ]

    errors = validate_source_references(cases, sources)

    assert errors == []


def test_unknown_source_is_reported() -> None:
    cases = [
        EvaluationCase(
            id="CASE-001",
            description="Test case",
            entity=Entity(
                name="Acme Holdings",
                entity_type="company",
            ),
            expected_findings=[
                ExpectedFinding(
                    claim="Acme received a penalty.",
                    source_ids=["source-999"],
                )
            ],
        )
    ]

    sources = [
        SourceDocument(
            id="source-001",
            title="Regulatory Decision",
            source_type="regulator",
            content="Acme received a penalty.",
        )
    ]

    errors = validate_source_references(cases, sources)

    assert errors == ["CASE-001: unknown source 'source-999'"]
