"""The golden dataset is test data for the evaluator, so it needs its own tests."""

from pathlib import Path

from ai_eval_lab.datasets.loader import load_evaluation_cases, load_sources
from ai_eval_lab.datasets.validator import validate_source_references


def test_every_expected_finding_cites_a_source_that_exists(datasets_dir: Path) -> None:
    cases = load_evaluation_cases(datasets_dir / "golden" / "cases.json")
    sources = load_sources(datasets_dir / "sources")

    assert validate_source_references(cases, sources) == []


def test_case_ids_are_unique(datasets_dir: Path) -> None:
    cases = load_evaluation_cases(datasets_dir / "golden" / "cases.json")
    ids = [case.id for case in cases]

    assert len(ids) == len(set(ids))


def test_source_ids_match_their_file_names(datasets_dir: Path) -> None:
    file_stems = sorted(path.stem for path in (datasets_dir / "sources").glob("*.json"))

    assert [source.id for source in load_sources(datasets_dir / "sources")] == file_stems
