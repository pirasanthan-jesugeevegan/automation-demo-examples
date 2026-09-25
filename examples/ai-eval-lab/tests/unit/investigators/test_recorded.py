from pathlib import Path

import pytest

from ai_eval_lab.datasets.loader import load_evaluation_cases, load_sources
from ai_eval_lab.investigators.recorded import RecordedInvestigator


def test_replays_the_recorded_findings_for_a_case(datasets_dir: Path) -> None:
    cases = load_evaluation_cases(datasets_dir / "golden" / "cases.json")
    sources = load_sources(datasets_dir / "sources")
    investigator = RecordedInvestigator.from_file(datasets_dir / "recorded" / "baseline.json")

    investigation = investigator.investigate(cases[0], sources)

    assert investigation.entity == cases[0].entity
    assert investigation.sources == sources
    assert [f.source_ids for f in investigation.findings] == [["source-001"]]


def test_a_case_with_no_recording_is_an_error(datasets_dir: Path) -> None:
    cases = load_evaluation_cases(datasets_dir / "golden" / "cases.json")
    investigator = RecordedInvestigator({})

    with pytest.raises(LookupError, match="CASE-001"):
        investigator.investigate(cases[0], [])


def test_every_recording_covers_every_golden_case(datasets_dir: Path) -> None:
    case_ids = {c.id for c in load_evaluation_cases(datasets_dir / "golden" / "cases.json")}

    for recording in (datasets_dir / "recorded").glob("*.json"):
        investigator = RecordedInvestigator.from_file(recording)
        assert set(investigator.recording) == case_ids, recording.name
