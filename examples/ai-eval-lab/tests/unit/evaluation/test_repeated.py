from pathlib import Path

import pytest

from ai_eval_lab.datasets.loader import load_evaluation_cases, load_sources
from ai_eval_lab.evaluation.repeated import run_repeated
from ai_eval_lab.investigators.recorded import RecordedInvestigator
from tests.scripted import ScriptedInvestigator


def recordings(datasets_dir: Path) -> tuple[RecordedInvestigator, RecordedInvestigator]:
    recorded = datasets_dir / "recorded"
    return (
        RecordedInvestigator.from_file(recorded / "baseline.json"),
        RecordedInvestigator.from_file(recorded / "regression.json"),
    )


def test_a_stable_system_has_identical_runs_and_no_flaky_cases(datasets_dir: Path) -> None:
    cases = load_evaluation_cases(datasets_dir / "golden" / "cases.json")
    sources = load_sources(datasets_dir / "sources")
    baseline, _ = recordings(datasets_dir)

    report = run_repeated(cases, sources, baseline, runs=3)

    assert report.pass_rates == [1.0, 1.0, 1.0]
    assert report.worst_pass_rate == 1.0
    assert report.flaky_case_ids() == []


def test_a_varying_system_shows_its_worst_run_its_mean_and_its_flaky_cases(
    datasets_dir: Path,
) -> None:
    cases = load_evaluation_cases(datasets_dir / "golden" / "cases.json")
    sources = load_sources(datasets_dir / "sources")
    baseline, regression = recordings(datasets_dir)
    # Good, bad, good. The regression passes only CASE-003.
    system = ScriptedInvestigator([baseline, regression, baseline], cases_per_run=len(cases))

    report = run_repeated(cases, sources, system, runs=3)

    assert report.pass_rates == [1.0, 0.2, 1.0]
    assert report.worst_pass_rate == 0.2
    assert report.mean_pass_rate == pytest.approx(2.2 / 3)
    assert report.worst_invalid_citations == 1
    assert report.case_pass_counts() == {
        "CASE-001": 2,
        "CASE-002": 2,
        "CASE-003": 3,
        "CASE-004": 2,
        "CASE-005": 2,
    }
    # CASE-003 passes every time, so it is stable. The others are right only some of the time.
    assert report.flaky_case_ids() == ["CASE-001", "CASE-002", "CASE-004", "CASE-005"]


def test_zero_runs_is_an_error(datasets_dir: Path) -> None:
    baseline, _ = recordings(datasets_dir)

    with pytest.raises(ValueError, match="at least one run"):
        run_repeated([], [], baseline, runs=0)
