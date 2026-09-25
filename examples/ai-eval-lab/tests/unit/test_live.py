import json
from pathlib import Path

import pytest

from ai_eval_lab import live
from ai_eval_lab.cli import EXIT_BAD_INPUT, EXIT_BELOW_GATE, EXIT_OK
from ai_eval_lab.investigators.recorded import RecordedInvestigator
from tests.scripted import ScriptedInvestigator


def run(datasets_dir: Path, *args: str) -> int:
    return live.main(["--datasets", str(datasets_dir), *args])


def use_investigator(monkeypatch: pytest.MonkeyPatch, investigator: object) -> None:
    monkeypatch.setattr(live, "AnthropicInvestigator", lambda: investigator)


def test_a_stable_live_system_passes_the_gate(
    datasets_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = RecordedInvestigator.from_file(datasets_dir / "recorded" / "baseline.json")
    use_investigator(monkeypatch, baseline)

    assert run(datasets_dir, "--runs", "2") == EXIT_OK

    output = capsys.readouterr().out
    assert "CASE-001  passed 2/2 runs" in output
    assert "Gate: PASSED" in output


def test_one_bad_run_fails_the_gate_and_the_flaky_cases_are_named(
    datasets_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    recorded = datasets_dir / "recorded"
    baseline = RecordedInvestigator.from_file(recorded / "baseline.json")
    regression = RecordedInvestigator.from_file(recorded / "regression.json")
    use_investigator(monkeypatch, ScriptedInvestigator([baseline, regression, baseline], 5))

    assert run(datasets_dir, "--runs", "3", "--min-pass-rate", "0.9") == EXIT_BELOW_GATE

    output = capsys.readouterr().out
    assert "CASE-001  passed 2/3 runs  FLAKY" in output
    assert "CASE-003  passed 3/3 runs\n" in output
    assert "pass rate per run: 100%, 20%, 100% (mean 73%)" in output
    assert "the gate applies to the worst run" in output


def test_the_report_file_holds_every_run(
    datasets_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    baseline = RecordedInvestigator.from_file(datasets_dir / "recorded" / "baseline.json")
    use_investigator(monkeypatch, baseline)
    report_path = tmp_path / "live.json"

    run(datasets_dir, "--runs", "2", "--report", str(report_path))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["runs"] == 2
    assert report["pass_rates"] == [1.0, 1.0]
    assert report["flaky_cases"] == []
    assert len(report["reports"]) == 2


def test_running_live_without_an_api_key_is_a_clear_error(
    datasets_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert run(datasets_dir) == EXIT_BAD_INPUT
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_zero_runs_is_rejected_by_the_parser(datasets_dir: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        run(datasets_dir, "--runs", "0")

    assert exit_info.value.code == 2
