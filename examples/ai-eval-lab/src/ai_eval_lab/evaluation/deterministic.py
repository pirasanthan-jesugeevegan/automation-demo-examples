"""Score an AI investigation against a golden case, with no model calls."""

from ai_eval_lab.evaluation.normalization import normalize_claim
from ai_eval_lab.models.evaluation_case import EvaluationCase
from ai_eval_lab.models.evaluation_result import (
    EvaluationMetrics,
    EvaluationResult,
)
from ai_eval_lab.models.investigation import Finding, Investigation


def _invalid_source_ids(actual: Investigation) -> list[str]:
    available = {source.id for source in actual.sources}
    cited = {sid for finding in actual.findings for sid in finding.source_ids}

    return sorted(cited - available)


def _match_expected_findings(
    expected: EvaluationCase,
    actual: Investigation,
) -> tuple[int, list[str]]:
    """How many expected findings were produced with their sources, and why the rest were not."""
    produced = {normalize_claim(finding.claim): finding for finding in actual.findings}
    matched = 0
    errors: list[str] = []

    for expected_finding in expected.expected_findings:
        finding = produced.get(normalize_claim(expected_finding.claim))

        if finding is None:
            errors.append(
                f"{expected.id}: expected finding was not produced: '{expected_finding.claim}'"
            )
        elif not set(expected_finding.source_ids) <= set(finding.source_ids):
            errors.append(f"{expected.id}: finding has incorrect source references.")
        else:
            matched += 1

    return matched, errors


def _unexpected_findings(expected: EvaluationCase, actual: Investigation) -> list[Finding]:
    expected_claims = {normalize_claim(f.claim) for f in expected.expected_findings}

    return [f for f in actual.findings if normalize_claim(f.claim) not in expected_claims]


def evaluate(expected: EvaluationCase, actual: Investigation) -> EvaluationResult:
    errors: list[str] = []

    invalid_ids = _invalid_source_ids(actual)
    if invalid_ids:
        errors.append("AI output contains invalid source references: " + ", ".join(invalid_ids))

    matched, match_errors = _match_expected_findings(expected, actual)
    errors.extend(match_errors)

    unexpected = _unexpected_findings(expected, actual)
    errors.extend(f"{expected.id}: unexpected finding: '{f.claim}'" for f in unexpected)

    expected_count = len(expected.expected_findings)
    metrics = EvaluationMetrics(
        source_validity=0.0 if invalid_ids else 1.0,
        finding_coverage=matched / expected_count if expected_count else 1.0,
        unexpected_finding_rate=len(unexpected) / len(actual.findings) if actual.findings else 0.0,
    )

    return EvaluationResult(
        case_id=expected.id,
        passed=not errors,
        metrics=metrics,
        errors=errors,
        invalid_source_ids=invalid_ids,
    )
