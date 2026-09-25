from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    source_validity: float = Field(ge=0.0, le=1.0)
    finding_coverage: float = Field(ge=0.0, le=1.0)
    unexpected_finding_rate: float = Field(ge=0.0, le=1.0)


class EvaluationResult(BaseModel):
    case_id: str
    passed: bool
    metrics: EvaluationMetrics
    errors: list[str]
    # Cited ids that are not among the sources the AI reported using: fabricated citations.
    invalid_source_ids: list[str] = []
