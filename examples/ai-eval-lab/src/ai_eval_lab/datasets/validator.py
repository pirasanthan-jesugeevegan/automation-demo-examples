from ai_eval_lab.models.evaluation_case import EvaluationCase
from ai_eval_lab.models.investigation import SourceDocument


def validate_source_references(
    cases: list[EvaluationCase],
    sources: list[SourceDocument],
) -> list[str]:
    source_ids = {source.id for source in sources}

    errors: list[str] = []

    for case in cases:
        for finding in case.expected_findings:
            for source_id in finding.source_ids:
                if source_id not in source_ids:
                    errors.append(f"{case.id}: unknown source '{source_id}'")

    return errors
