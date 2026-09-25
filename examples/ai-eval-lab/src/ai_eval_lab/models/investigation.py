from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(min_length=1)
    entity_type: str


class SourceDocument(BaseModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_type: str
    content: str = Field(min_length=1)


class Finding(BaseModel):
    claim: str = Field(min_length=1)
    source_ids: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class Investigation(BaseModel):
    entity: Entity
    sources: list[SourceDocument]
    findings: list[Finding]
