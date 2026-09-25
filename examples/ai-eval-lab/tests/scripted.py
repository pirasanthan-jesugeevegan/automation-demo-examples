from ai_eval_lab.investigators.recorded import RecordedInvestigator
from ai_eval_lab.models.evaluation_case import EvaluationCase
from ai_eval_lab.models.investigation import Investigation, SourceDocument


class ScriptedInvestigator:
    """Behaves like one recording on each successive run, as a live model that varies would."""

    def __init__(self, script: list[RecordedInvestigator], cases_per_run: int) -> None:
        self.script = script
        self.cases_per_run = cases_per_run
        self.calls = 0

    def investigate(self, case: EvaluationCase, sources: list[SourceDocument]) -> Investigation:
        run = self.calls // self.cases_per_run
        self.calls += 1
        return self.script[run].investigate(case, sources)
