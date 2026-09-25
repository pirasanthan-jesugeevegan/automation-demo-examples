"""Run a system under test over the golden cases and score what it produces."""

from typing import Protocol

from pydantic import BaseModel

from ai_eval_lab.evaluation.deterministic import evaluate
from ai_eval_lab.evaluation.groundedness import evaluate_groundedness
from ai_eval_lab.evaluation.judge import Judge
from ai_eval_lab.evaluation.summary import evaluate_investigation_groundedness
from ai_eval_lab.models.evaluation_case import EvaluationCase
from ai_eval_lab.models.evaluation_result import EvaluationResult
from ai_eval_lab.models.groundedness import GroundednessSummary
from ai_eval_lab.models.investigation import Investigation, SourceDocument


class Investigator(Protocol):
    """The system under test: given a case and the available sources, produce findings."""

    def investigate(self, case: EvaluationCase, sources: list[SourceDocument]) -> Investigation: ...


class CaseReport(BaseModel):
    result: EvaluationResult
    groundedness: GroundednessSummary
    # Findings where the offline check and the LLM judge disagree. Empty without a judge.
    judge_disagreements: list[str] = []


class EvaluationReport(BaseModel):
    cases: list[CaseReport]

    @property
    def passed_cases(self) -> int:
        return sum(case.result.passed for case in self.cases)

    @property
    def invalid_citations(self) -> int:
        """Fabricated citations across all cases."""
        return sum(len(case.result.invalid_source_ids) for case in self.cases)

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / len(self.cases) if self.cases else 0.0


def _judge_disagreements(investigation: Investigation, judge: Judge) -> list[str]:
    disagreements: list[str] = []

    for finding in investigation.findings:
        offline = evaluate_groundedness(finding, investigation.sources)
        verdict = judge.evaluate(finding, investigation.sources)

        if offline.grounded != verdict.grounded:
            disagreements.append(
                f"'{finding.claim}': offline check says grounded={offline.grounded}, "
                f"judge says grounded={verdict.grounded} ({verdict.explanation})"
            )

    return disagreements


def run_evaluation(
    cases: list[EvaluationCase],
    sources: list[SourceDocument],
    investigator: Investigator,
    judge: Judge | None = None,
) -> EvaluationReport:
    if not cases:
        raise ValueError("There are no cases to evaluate, so the result would prove nothing.")

    reports: list[CaseReport] = []

    for case in cases:
        investigation = investigator.investigate(case, sources)

        reports.append(
            CaseReport(
                result=evaluate(case, investigation),
                groundedness=evaluate_investigation_groundedness(investigation),
                judge_disagreements=_judge_disagreements(investigation, judge) if judge else [],
            )
        )

    return EvaluationReport(cases=reports)
