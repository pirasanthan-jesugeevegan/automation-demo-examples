from pathlib import Path

import pytest

from ai_eval_lab.datasets.loader import load_sources
from ai_eval_lab.evaluation.heuristic_judge import HeuristicJudge
from ai_eval_lab.evaluation.judge_validation import (
    JudgeValidationReport,
    LabelledFinding,
    load_labels,
    validate_judge,
    validate_labels,
)
from ai_eval_lab.models.groundedness import GroundednessResult
from ai_eval_lab.models.investigation import Finding, SourceDocument
from tests.fakes import FakeJudge


def report(tp: int, fp: int, tn: int, fn: int) -> JudgeValidationReport:
    return JudgeValidationReport(
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        disagreements=[],
    )


def label(label_id: str, grounded: bool, source_ids: list[str] | None = None) -> LabelledFinding:
    return LabelledFinding(
        id=label_id,
        claim=f"Claim {label_id}",
        source_ids=source_ids or ["source-001"],
        grounded=grounded,
        why="Because.",
    )


class TestMetrics:
    def test_rates_and_kappa_are_computed_from_the_confusion_matrix(self) -> None:
        r = report(tp=3, fp=1, tn=4, fn=2)

        assert r.total == 10
        assert r.agreement == 0.7
        assert r.false_pass_rate == pytest.approx(1 / 5)  # 1 of 5 unsupported claims let through
        assert r.false_fail_rate == pytest.approx(2 / 5)  # 2 of 5 supported claims rejected
        assert r.cohens_kappa == pytest.approx(0.4)  # (0.7 - 0.5) / (1 - 0.5)

    def test_a_perfect_judge_has_kappa_one(self) -> None:
        assert report(tp=4, fp=0, tn=5, fn=0).cohens_kappa == 1.0

    def test_when_everything_is_one_class_and_the_judge_agrees_kappa_is_one(self) -> None:
        # Chance agreement is 1 here, so the usual formula would divide by zero.
        assert report(tp=5, fp=0, tn=0, fn=0).cohens_kappa == 1.0

    def test_a_judge_no_better_than_chance_has_kappa_zero(self) -> None:
        # Says "grounded" for everything, on a set that is half grounded.
        assert report(tp=5, fp=5, tn=0, fn=0).cohens_kappa == 0.0

    def test_an_empty_report_is_all_zeros_not_a_crash(self) -> None:
        r = report(0, 0, 0, 0)

        assert (r.agreement, r.false_pass_rate, r.false_fail_rate, r.cohens_kappa) == (0, 0, 0, 0)


class TestValidateJudge:
    SOURCE = SourceDocument(id="source-001", title="T", source_type="regulator", content="Text.")

    def test_a_judge_that_passes_everything_is_caught_letting_bad_claims_through(self) -> None:
        verdict = GroundednessResult(
            grounded=True, score=1.0, explanation="Yes.", supporting_source_ids=[]
        )
        labels = [label("A", True), label("B", False), label("C", False)]

        result = validate_judge(FakeJudge(verdict), labels, [self.SOURCE])

        assert (result.true_positives, result.false_positives) == (1, 2)
        assert result.false_pass_rate == 1.0
        assert [d.split(" ")[0] for d in result.disagreements] == ["B", "C"]
        assert "false pass" in result.disagreements[0]

    def test_a_judge_that_rejects_everything_is_caught_dropping_good_claims(self) -> None:
        verdict = GroundednessResult(
            grounded=False, score=0.0, explanation="No.", supporting_source_ids=[]
        )

        result = validate_judge(FakeJudge(verdict), [label("A", True)], [self.SOURCE])

        assert result.false_fail_rate == 1.0
        assert "false fail" in result.disagreements[0]

    def test_the_judge_is_given_the_claim_and_its_cited_sources(self) -> None:
        seen: list[Finding] = []

        class Recorder:
            def evaluate(
                self, finding: Finding, sources: list[SourceDocument]
            ) -> GroundednessResult:
                seen.append(finding)
                return GroundednessResult(
                    grounded=True, score=1.0, explanation="Yes.", supporting_source_ids=[]
                )

        validate_judge(Recorder(), [label("A", True, ["source-001"])], [self.SOURCE])

        assert seen[0].claim == "Claim A"
        assert seen[0].source_ids == ["source-001"]


class TestValidateLabels:
    SOURCES = [SourceDocument(id="source-001", title="T", source_type="x", content="Text.")]

    def test_a_good_label_set_has_no_problems(self) -> None:
        assert validate_labels([label("A", True), label("B", False)], self.SOURCES) == []

    def test_repeated_ids_are_reported(self) -> None:
        problems = validate_labels([label("A", True), label("A", False)], self.SOURCES)

        assert "A: id used more than once" in problems

    def test_an_unknown_source_is_reported(self) -> None:
        labels = [label("A", True, ["source-404"]), label("B", False)]

        assert "A: unknown source 'source-404'" in validate_labels(labels, self.SOURCES)

    def test_one_sided_labels_are_reported_because_they_cannot_measure_both_errors(self) -> None:
        problems = validate_labels([label("A", True), label("B", True)], self.SOURCES)

        assert any("both grounded and ungrounded" in p for p in problems)


class TestLoadLabels:
    def test_a_file_that_is_not_a_list_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "labels.json"
        path.write_text('{"id": "LABEL-01"}', encoding="utf-8")

        with pytest.raises(ValueError, match="must be a list"):
            load_labels(path)


class TestTheLabelFile:
    def test_it_is_consistent_with_the_sources(self, datasets_dir: Path) -> None:
        labels = load_labels(datasets_dir / "judge_labels" / "labels.json")
        sources = load_sources(datasets_dir / "sources")

        assert validate_labels(labels, sources) == []

    def test_it_is_balanced_so_neither_error_can_hide(self, datasets_dir: Path) -> None:
        labels = load_labels(datasets_dir / "judge_labels" / "labels.json")

        assert sum(item.grounded for item in labels) == len(labels) // 2

    def test_the_heuristic_scores_as_measured(self, datasets_dir: Path) -> None:
        """Pins what the offline check gets right and wrong today, so a change to it is a decision.

        Its four errors are the limits the README describes: it cannot see negation, does not
        know "CEO" means Chief Executive Officer, cannot tell a notice from a fine, and cannot
        verify a claim that sources disagree. If you improve the heuristic, update these numbers.
        """
        labels = load_labels(datasets_dir / "judge_labels" / "labels.json")
        sources = load_sources(datasets_dir / "sources")

        result = validate_judge(HeuristicJudge(), labels, sources)

        assert (result.true_positives, result.false_positives) == (7, 2)
        assert (result.false_negatives, result.true_negatives) == (2, 7)
        assert [d.split(" ")[0] for d in result.disagreements] == [
            "LABEL-05",
            "LABEL-08",
            "LABEL-13",
            "LABEL-16",
        ]
