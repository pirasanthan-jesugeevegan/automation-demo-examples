from pathlib import Path

import pytest

from ai_eval_lab.datasets.loader import (
    load_evaluation_cases,
    load_json,
    load_sources,
)


def test_load_json_reads_a_json_file(tmp_path: Path) -> None:
    file_path = tmp_path / "example.json"
    file_path.write_text('{"name": "Acme Holdings"}', encoding="utf-8")

    assert load_json(file_path) == {"name": "Acme Holdings"}


def test_load_evaluation_cases(datasets_dir: Path) -> None:
    cases = load_evaluation_cases(datasets_dir / "golden" / "cases.json")

    assert len(cases) == 5
    assert cases[0].id == "CASE-001"
    assert cases[0].entity.name == "Acme Holdings"


def test_load_sources_reads_every_source_file(datasets_dir: Path) -> None:
    sources = load_sources(datasets_dir / "sources")

    assert [source.id for source in sources] == [
        "source-001",
        "source-002",
        "source-003",
        "source-004",
        "source-005",
        "source-006",
    ]


def test_a_dataset_that_is_not_a_list_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text('{"id": "CASE-001"}', encoding="utf-8")

    with pytest.raises(ValueError, match="list of cases"):
        load_evaluation_cases(path)
