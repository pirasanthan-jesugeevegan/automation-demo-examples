import pytest
from pydantic import ValidationError

from ai_eval_lab.models.evaluation_case import (
    EvaluationCase,
    ExpectedFinding,
)
from ai_eval_lab.models.investigation import Entity


def test_valid_evaluation_case_can_be_created() -> None:
    case = EvaluationCase(
        id="CASE-001",
        description="Straightforward regulatory finding",
        entity=Entity(name="Acme Holdings", entity_type="company"),
        expected_findings=[
            ExpectedFinding(
                claim="Acme Holdings received a £2 million regulatory penalty in 2024.",
                source_ids=["source-001"],
            )
        ],
    )

    assert case.id == "CASE-001"
    assert case.entity.name == "Acme Holdings"
    assert len(case.expected_findings) == 1


def test_evaluation_case_requires_an_id() -> None:
    with pytest.raises(ValidationError):
        EvaluationCase(
            id="",
            description="Test case",
            entity=Entity(name="Acme Holdings", entity_type="company"),
            expected_findings=[],
        )
