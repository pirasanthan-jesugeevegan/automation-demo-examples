from pydantic import BaseModel, Field

from ai_eval_lab.models.investigation import Entity


class ExpectedFinding(BaseModel):
    claim: str = Field(min_length=1)
    source_ids: list[str]


class EvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    entity: Entity
    expected_findings: list[ExpectedFinding]
