from typing import Protocol

from ai_eval_lab.models.groundedness import GroundednessResult
from ai_eval_lab.models.investigation import Finding, SourceDocument


class Judge(Protocol):
    """Decides whether a finding is supported by the sources it cites."""

    def evaluate(
        self,
        finding: Finding,
        sources: list[SourceDocument],
    ) -> GroundednessResult: ...
