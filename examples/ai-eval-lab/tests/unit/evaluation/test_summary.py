from ai_eval_lab.evaluation.summary import (
    evaluate_investigation_groundedness,
)
from ai_eval_lab.models.investigation import (
    Entity,
    Finding,
    Investigation,
    SourceDocument,
)


def test_investigation_groundedness_summary() -> None:
    source = SourceDocument(
        id="source-001",
        title="Regulatory Decision",
        source_type="regulator",
        content=("In 2024, the regulator announced a £2 million penalty against Acme Holdings."),
    )

    investigation = Investigation(
        entity=Entity(
            name="Acme Holdings",
            entity_type="company",
        ),
        sources=[source],
        findings=[
            Finding(
                claim=("Acme Holdings received a £2 million penalty in 2024."),
                source_ids=["source-001"],
                confidence=0.95,
            ),
            Finding(
                claim="Acme Holdings opened 50 offices in Canada.",
                source_ids=["source-001"],
                confidence=0.90,
            ),
        ],
    )

    result = evaluate_investigation_groundedness(
        investigation,
    )

    assert result.total_findings == 2
    assert result.grounded_findings == 1
    assert result.groundedness == 0.5
