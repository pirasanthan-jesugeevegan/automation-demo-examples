"""Check a judge against human labels.

An evaluation is only as good as its judge. This measures how often a judge agrees with a person
who read the same sources, and which way it errs. Treat "grounded" as the positive class:

- a false pass lets an unsupported claim through. This is the dangerous error.
- a false fail rejects a claim that was supported. It wastes review time.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from ai_eval_lab.datasets.loader import load_json
from ai_eval_lab.evaluation.judge import Judge
from ai_eval_lab.models.investigation import Finding, SourceDocument


class LabelledFinding(BaseModel):
    id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    source_ids: list[str]
    grounded: bool
    why: str = Field(min_length=1)


class JudgeValidationReport(BaseModel):
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    disagreements: list[str]

    @property
    def total(self) -> int:
        return (
            self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        )

    @property
    def agreement(self) -> float:
        return (self.true_positives + self.true_negatives) / self.total if self.total else 0.0

    @property
    def false_pass_rate(self) -> float:
        """Share of unsupported claims the judge let through."""
        unsupported = self.false_positives + self.true_negatives
        return self.false_positives / unsupported if unsupported else 0.0

    @property
    def false_fail_rate(self) -> float:
        """Share of supported claims the judge rejected."""
        supported = self.false_negatives + self.true_positives
        return self.false_negatives / supported if supported else 0.0

    @property
    def cohens_kappa(self) -> float:
        """Agreement beyond chance: 1 is perfect, 0 is no better than guessing."""
        n = self.total
        if not n:
            return 0.0

        observed = (self.true_positives + self.true_negatives) / n
        judged_yes = self.true_positives + self.false_positives
        labelled_yes = self.true_positives + self.false_negatives
        judged_no = self.false_negatives + self.true_negatives
        labelled_no = self.false_positives + self.true_negatives
        expected = (judged_yes * labelled_yes + judged_no * labelled_no) / (n * n)

        if expected == 1.0:
            return 1.0 if observed == 1.0 else 0.0
        return (observed - expected) / (1 - expected)


def load_labels(path: Path) -> list[LabelledFinding]:
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError("Judge labels must be a list.")

    return [LabelledFinding.model_validate(item) for item in data]


def validate_labels(labels: list[LabelledFinding], sources: list[SourceDocument]) -> list[str]:
    """Problems with the label set itself: repeated ids, unknown sources, one-sided labels."""
    known = {source.id for source in sources}
    problems: list[str] = []

    ids = [label.id for label in labels]
    problems.extend(
        f"{i}: id used more than once" for i in sorted({i for i in ids if ids.count(i) > 1})
    )

    for label in labels:
        problems.extend(
            f"{label.id}: unknown source '{sid}'" for sid in label.source_ids if sid not in known
        )

    if not any(label.grounded for label in labels) or all(label.grounded for label in labels):
        problems.append("Labels need both grounded and ungrounded examples.")

    return problems


def validate_judge(
    judge: Judge,
    labels: list[LabelledFinding],
    sources: list[SourceDocument],
) -> JudgeValidationReport:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    disagreements: list[str] = []

    for label in labels:
        finding = Finding(claim=label.claim, source_ids=label.source_ids, confidence=1.0)
        verdict = judge.evaluate(finding, sources)

        if verdict.grounded and label.grounded:
            counts["tp"] += 1
        elif verdict.grounded:
            counts["fp"] += 1
        elif label.grounded:
            counts["fn"] += 1
        else:
            counts["tn"] += 1

        if verdict.grounded != label.grounded:
            kind = "false pass" if verdict.grounded else "false fail"
            disagreements.append(f"{label.id} ({kind}): {label.claim} [{label.why}]")

    return JudgeValidationReport(
        true_positives=counts["tp"],
        false_positives=counts["fp"],
        true_negatives=counts["tn"],
        false_negatives=counts["fn"],
        disagreements=disagreements,
    )
