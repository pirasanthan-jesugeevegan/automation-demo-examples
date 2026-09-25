"""Run the same evaluation several times.

A live model can answer differently each time, so one run says little. Repeating shows the worst
case, the average, and which cases are flaky (right on some runs, wrong on others).
"""

from pydantic import BaseModel

from ai_eval_lab.evaluation.judge import Judge
from ai_eval_lab.evaluation.runner import EvaluationReport, Investigator, run_evaluation
from ai_eval_lab.models.evaluation_case import EvaluationCase
from ai_eval_lab.models.investigation import SourceDocument


class RepeatedReport(BaseModel):
    runs: list[EvaluationReport]

    @property
    def pass_rates(self) -> list[float]:
        return [run.pass_rate for run in self.runs]

    @property
    def worst_pass_rate(self) -> float:
        return min(self.pass_rates)

    @property
    def mean_pass_rate(self) -> float:
        return sum(self.pass_rates) / len(self.runs)

    @property
    def worst_invalid_citations(self) -> int:
        return max(run.invalid_citations for run in self.runs)

    def case_pass_counts(self) -> dict[str, int]:
        """For each case, on how many runs it passed."""
        counts: dict[str, int] = {}
        for run in self.runs:
            for case in run.cases:
                counts[case.result.case_id] = (
                    counts.get(case.result.case_id, 0) + case.result.passed
                )
        return counts

    def flaky_case_ids(self) -> list[str]:
        """Cases that passed on some runs but not all."""
        total = len(self.runs)
        return [case_id for case_id, count in self.case_pass_counts().items() if 0 < count < total]


def run_repeated(
    cases: list[EvaluationCase],
    sources: list[SourceDocument],
    investigator: Investigator,
    runs: int,
    judge: Judge | None = None,
) -> RepeatedReport:
    if runs < 1:
        raise ValueError("Need at least one run.")

    return RepeatedReport(
        runs=[run_evaluation(cases, sources, investigator, judge) for _ in range(runs)]
    )
