import json
from pathlib import Path
from typing import Any

from ai_eval_lab.models.evaluation_case import EvaluationCase
from ai_eval_lab.models.investigation import SourceDocument


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    data = load_json(path)

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must contain a list of cases.")

    return [EvaluationCase.model_validate(case) for case in data]


def load_sources(directory: Path) -> list[SourceDocument]:
    """Every `*.json` file in the directory is one source document."""
    return [
        SourceDocument.model_validate(load_json(path)) for path in sorted(directory.glob("*.json"))
    ]
