"""Replays saved outputs of the system under test, so an evaluation needs no model and no key."""

from pathlib import Path

from ai_eval_lab.datasets.loader import load_json
from ai_eval_lab.models.evaluation_case import EvaluationCase
from ai_eval_lab.models.investigation import Finding, Investigation, SourceDocument


class RecordedInvestigator:
    def __init__(self, recording: dict[str, list[Finding]]) -> None:
        self.recording = recording

    @classmethod
    def from_file(cls, path: Path) -> "RecordedInvestigator":
        raw = load_json(path)
        return cls(
            {
                case_id: [Finding.model_validate(f) for f in findings]
                for case_id, findings in raw.items()
            }
        )

    def investigate(self, case: EvaluationCase, sources: list[SourceDocument]) -> Investigation:
        if case.id not in self.recording:
            raise LookupError(f"No recorded output for {case.id}.")

        return Investigation(
            entity=case.entity,
            sources=sources,
            findings=self.recording[case.id],
        )
