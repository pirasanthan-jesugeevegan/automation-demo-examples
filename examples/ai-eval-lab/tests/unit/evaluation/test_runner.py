import pytest

from ai_eval_lab.evaluation.runner import run_evaluation
from ai_eval_lab.models.evaluation_case import EvaluationCase, ExpectedFinding
from ai_eval_lab.models.groundedness import GroundednessResult
from ai_eval_lab.models.investigation import (
    Entity,
    Finding,
    Investigation,
    SourceDocument,
)
from tests.fakes import FakeJudge

SOURCE = SourceDocument(
    id="source-001",
    title="Regulatory Decision",
    source_type="regulator",
    content="In 2024, the regulator announced a £2 million penalty against Acme Holdings.",
)
CLAIM = "Acme Holdings received a £2 million penalty in 2024."


def make_case(case_id: str) -> EvaluationCase:
    return EvaluationCase(
        id=case_id,
        description="Regulatory penalty",
        entity=Entity(name="Acme Holdings", entity_type="company"),
        expected_findings=[ExpectedFinding(claim=CLAIM, source_ids=["source-001"])],
    )


class StubInvestigator:
    """Produces the expected finding for every case except the ones told to get it wrong."""

    def __init__(self, wrong_for: frozenset[str] = frozenset()) -> None:
        self.wrong_for = wrong_for

    def investigate(self, case: EvaluationCase, sources: list[SourceDocument]) -> Investigation:
        claim = (
            "Acme Holdings received a £20 million penalty in 2024."
            if case.id in self.wrong_for
            else CLAIM
        )
        finding = Finding(claim=claim, source_ids=["source-001"], confidence=0.9)
        return Investigation(entity=case.entity, sources=sources, findings=[finding])


def test_a_correct_investigator_passes_every_case() -> None:
    cases = [make_case("CASE-001"), make_case("CASE-002")]

    report = run_evaluation(cases, [SOURCE], StubInvestigator())

    assert report.pass_rate == 1.0
    assert report.passed_cases == 2


def test_a_wrong_answer_fails_only_its_own_case() -> None:
    cases = [make_case("CASE-001"), make_case("CASE-002")]

    report = run_evaluation(cases, [SOURCE], StubInvestigator(wrong_for=frozenset({"CASE-001"})))

    assert [c.result.case_id for c in report.cases if not c.result.passed] == ["CASE-001"]
    assert report.pass_rate == 0.5


def test_groundedness_is_reported_for_every_case() -> None:
    report = run_evaluation([make_case("CASE-001")], [SOURCE], StubInvestigator())

    assert report.cases[0].groundedness.total_findings == 1
    assert report.cases[0].groundedness.grounded_findings == 1


def test_an_empty_dataset_is_an_error_not_a_pass() -> None:
    with pytest.raises(ValueError, match="no cases"):
        run_evaluation([], [SOURCE], StubInvestigator())


def test_judge_disagreements_are_reported_when_a_judge_is_given() -> None:
    verdict = GroundednessResult(
        grounded=True,
        score=0.9,
        explanation="Paraphrase of the source.",
        supporting_source_ids=["source-001"],
    )

    report = run_evaluation(
        [make_case("CASE-001")],
        [SOURCE],
        StubInvestigator(wrong_for=frozenset({"CASE-001"})),
        judge=FakeJudge(verdict),
    )

    (disagreement,) = report.cases[0].judge_disagreements
    assert "£20 million" in disagreement
    assert "judge says grounded=True" in disagreement


def test_no_judge_means_no_disagreements() -> None:
    report = run_evaluation([make_case("CASE-001")], [SOURCE], StubInvestigator())

    assert report.cases[0].judge_disagreements == []
