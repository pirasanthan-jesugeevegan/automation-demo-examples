from pydantic import BaseModel, Field


class GroundednessResult(BaseModel):
    grounded: bool
    score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(min_length=1)
    supporting_source_ids: list[str]


class GroundednessSummary(BaseModel):
    total_findings: int = Field(ge=0)
    grounded_findings: int = Field(ge=0)
    groundedness: float = Field(ge=0.0, le=1.0)
