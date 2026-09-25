import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ai_eval_lab import judge_check
from ai_eval_lab.cli import EXIT_BAD_INPUT, EXIT_BELOW_GATE, EXIT_OK
from ai_eval_lab.models.groundedness import GroundednessResult
from tests.fakes import FakeJudge

PROJECT = Path(__file__).resolve().parents[2]


def run(datasets_dir: Path, *args: str) -> int:
    return judge_check.main(["--datasets", str(datasets_dir), *args])


def test_the_offline_heuristic_is_validated_by_default(
    datasets_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(datasets_dir) == EXIT_OK

    output = capsys.readouterr().out
    assert "agreement        78%" in output
    assert "false-pass rate  22%" in output
    assert "LABEL-05 (false pass)" in output


def test_a_minimum_agreement_can_fail_the_run(
    datasets_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(datasets_dir, "--min-agreement", "0.9") == EXIT_BELOW_GATE
    assert "agreement 78% is below 90%" in capsys.readouterr().out


def test_a_false_pass_limit_can_fail_the_run(datasets_dir: Path) -> None:
    assert run(datasets_dir, "--max-false-pass-rate", "0.1") == EXIT_BELOW_GATE
    assert run(datasets_dir, "--max-false-pass-rate", "0.25") == EXIT_OK


def test_the_report_file_has_the_confusion_matrix(datasets_dir: Path, tmp_path: Path) -> None:
    report_path = tmp_path / "judge.json"

    run(datasets_dir, "--report", str(report_path))

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["judge"] == "heuristic"
    assert report["confusion"] == {
        "true_positives": 7,
        "false_positives": 2,
        "true_negatives": 7,
        "false_negatives": 2,
    }
    assert len(report["disagreements"]) == 4


def test_a_live_judge_is_validated_through_the_same_path(
    datasets_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    always_yes = GroundednessResult(
        grounded=True, score=1.0, explanation="Yes.", supporting_source_ids=[]
    )
    monkeypatch.setattr(
        "ai_eval_lab.evaluation.anthropic_judge.AnthropicJudge", lambda: FakeJudge(always_yes)
    )

    code = run(datasets_dir, "--judge", "anthropic", "--max-false-pass-rate", "0.5")

    assert code == EXIT_BELOW_GATE
    assert "false-pass rate  100%" in capsys.readouterr().out


def test_asking_for_the_live_judge_without_a_key_is_a_clear_error(
    datasets_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert run(datasets_dir, "--judge", "anthropic") == EXIT_BAD_INPUT
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_inconsistent_labels_are_bad_input(
    datasets_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "datasets"
    shutil.copytree(datasets_dir, broken)
    path = broken / "judge_labels" / "labels.json"
    labels = json.loads(path.read_text(encoding="utf-8"))
    labels[0]["source_ids"] = ["source-404"]
    path.write_text(json.dumps(labels), encoding="utf-8")

    assert run(broken) == EXIT_BAD_INPUT
    assert "source-404" in capsys.readouterr().err


@pytest.mark.parametrize("module", ["ai_eval_lab.judge_check", "ai_eval_lab.live"])
def test_each_module_entry_point_actually_runs(module: str) -> None:
    """`python -m` must call main(). A module that only defines it exits 0 and prints nothing."""
    env = {"PATH": "/usr/bin:/bin", "ANTHROPIC_API_KEY": ""}

    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.stdout or result.stderr, f"{module} produced no output"
    if module.endswith("live"):
        assert result.returncode == EXIT_BAD_INPUT
        assert "ANTHROPIC_API_KEY" in result.stderr
    else:
        assert result.returncode == EXIT_OK
        assert "agreement" in result.stdout
