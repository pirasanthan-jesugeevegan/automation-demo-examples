from ai_eval_lab.evaluation.groundedness import evaluate_groundedness
from ai_eval_lab.models.groundedness import GroundednessResult
from ai_eval_lab.models.investigation import Finding, SourceDocument


class HeuristicJudge:
    """The offline word-overlap check, behind the Judge interface so it can be validated too."""

    def evaluate(
        self,
        finding: Finding,
        sources: list[SourceDocument],
    ) -> GroundednessResult:
        return evaluate_groundedness(finding, sources)
