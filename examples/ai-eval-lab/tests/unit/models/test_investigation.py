import pytest
from pydantic import ValidationError

from ai_eval_lab.models.investigation import (
    Entity,
    Finding,
    Investigation,
    SourceDocument,
)


def test_valid_investigation_can_be_created() -> None:
    entity = Entity(
        name="Acme Holdings",
        entity_type="company",
    )

    source = SourceDocument(
        id="source-001",
        title="Regulatory Decision",
        source_type="regulator",
        content="Acme Holdings received a £2 million penalty.",
    )

    finding = Finding(
        claim="Acme Holdings received a £2 million penalty.",
        source_ids=["source-001"],
        confidence=0.95,
    )

    investigation = Investigation(
        entity=entity,
        sources=[source],
        findings=[finding],
    )

    assert investigation.entity.name == "Acme Holdings"
    assert len(investigation.sources) == 1
    assert len(investigation.findings) == 1


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1],
)
def test_confidence_must_be_between_zero_and_one(confidence: float) -> None:
    with pytest.raises(ValidationError):
        Finding(
            claim="Example claim",
            source_ids=["source-001"],
            confidence=confidence,
        )


def test_entity_requires_a_name() -> None:
    with pytest.raises(ValidationError):
        Entity(
            name="",
            entity_type="company",
        )
