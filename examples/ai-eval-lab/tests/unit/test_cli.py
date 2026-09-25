import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ai_eval_lab.cli import EXIT_BAD_INPUT, EXIT_BELOW_GATE, EXIT_OK, main
from ai_eval_lab.models.groundedness import GroundednessResult
from tests.fakes import FakeJudge


def run(datasets_dir: Path, *args: str) -> int:
    return main(["--datasets", str(datasets_dir), *args])


def test_the_healthy_baseline_passes_the_gate(
    datasets_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(datasets_dir, "--recording", "baseline") == EXIT_OK
    assert "5/5 cases passed" in capsys.readouterr().out


def test_the_seeded_regression_is_caught_case_by_case(
    datasets_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model upgrade that hallucinates an amount, mixes up two people, invents a source
    and returns an outdated answer must fail the gate, and say which cases and why."""
    assert run(datasets_dir, "--recording", "regression") == EXIT_BELOW_GATE

    output = capsys.readouterr().out
    for failing in ("CASE-001  FAIL", "CASE-002  FAIL", "CASE-004  FAIL", "CASE-005  FAIL"):
        assert failing in output
    assert "CASE-003  PASS" in output
    assert "invalid source references: source-404" in output
    assert "1/5 cases passed" in output


def test_the_gate_threshold_is_inclusive(datasets_dir: Path) -> None:
    # The regression passes 1 of 5 cases, exactly 20%.
    assert run(datasets_dir, "--recording", "regression", "--min-pass-rate", "0.2") == EXIT_OK
    assert (
        run(datasets_dir, "--recording", "regression", "--min-pass-rate", "0.21") == EXIT_BELOW_GATE
    )


def test_an_unknown_recording_is_bad_input_not_a_failed_gate(
    datasets_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(datasets_dir, "--recording", "nope") == EXIT_BAD_INPUT
    assert "Cannot load inputs" in capsys.readouterr().err


def test_a_missing_dataset_directory_is_bad_input(tmp_path: Path) -> None:
    assert run(tmp_path / "missing", "--recording", "baseline") == EXIT_BAD_INPUT


def test_a_golden_case_citing_an_unknown_source_is_bad_input(
    datasets_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "datasets"
    shutil.copytree(datasets_dir, broken)
    cases_path = broken / "golden" / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    cases[0]["expected_findings"][0]["source_ids"] = ["source-999"]
    cases_path.write_text(json.dumps(cases), encoding="utf-8")

    assert run(broken, "--recording", "baseline") == EXIT_BAD_INPUT
    assert "source-999" in capsys.readouterr().err


def test_judge_disagreements_are_printed_but_do_not_change_the_gate(
    datasets_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verdict = GroundednessResult(
        grounded=True,
        score=0.9,
        explanation="Supported.",
        supporting_source_ids=[],
    )
    monkeypatch.setattr(
        "ai_eval_lab.evaluation.anthropic_judge.AnthropicJudge", lambda: FakeJudge(verdict)
    )

    code = run(datasets_dir, "--recording", "baseline", "--judge", "anthropic")

    assert code == EXIT_OK
    assert "judge disagrees" in capsys.readouterr().out


def test_an_empty_golden_dataset_is_bad_input_not_a_pass(
    datasets_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "datasets"
    shutil.copytree(datasets_dir, empty)
    (empty / "golden" / "cases.json").write_text("[]", encoding="utf-8")

    assert run(empty, "--recording", "baseline") == EXIT_BAD_INPUT
    assert "no cases" in capsys.readouterr().err


def test_the_module_entry_point_sets_the_exit_code() -> None:
    project = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "-m", "ai_eval_lab", "--recording", "regression"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == EXIT_BELOW_GATE
    assert "1/5 cases passed" in result.stdout


def test_asking_for_the_judge_without_an_api_key_is_a_clear_error(
    datasets_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert run(datasets_dir, "--recording", "baseline", "--judge", "anthropic") == EXIT_BAD_INPUT
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_a_fabricated_citation_fails_the_gate_even_when_the_pass_rate_is_met(
    datasets_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(
        datasets_dir,
        "--recording",
        "regression",
        "--min-pass-rate",
        "0.2",
        "--max-invalid-citations",
        "0",
    )

    assert code == EXIT_BELOW_GATE
    output = capsys.readouterr().out
    assert "1 fabricated citation(s)" in output
    assert "pass rate" not in output.split("Gate: FAILED")[1]


def test_the_healthy_baseline_has_no_fabricated_citations(datasets_dir: Path) -> None:
    assert run(datasets_dir, "--recording", "baseline", "--max-invalid-citations", "0") == EXIT_OK


def test_the_report_file_holds_the_gate_result_and_every_case(
    datasets_dir: Path, tmp_path: Path
) -> None:
    report_path = tmp_path / "report.json"

    code = run(datasets_dir, "--recording", "regression", "--report", str(report_path))

    assert code == EXIT_BELOW_GATE
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["recording"] == "regression"
    assert report["gate_passed"] is False
    assert report["passed_cases"] == 1
    assert report["invalid_citations"] == 1
    assert [case["result"]["case_id"] for case in report["cases"]] == [
        "CASE-001",
        "CASE-002",
        "CASE-003",
        "CASE-004",
        "CASE-005",
    ]
    assert report["cases"][3]["result"]["invalid_source_ids"] == ["source-404"]
