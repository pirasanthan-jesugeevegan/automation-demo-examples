from ai_eval_lab.evaluation.groundedness import evaluate_groundedness
from ai_eval_lab.models.groundedness import GroundednessSummary
from ai_eval_lab.models.investigation import Investigation


def evaluate_investigation_groundedness(
    investigation: Investigation,
) -> GroundednessSummary:
    results = [
        evaluate_groundedness(
            finding,
            investigation.sources,
        )
        for finding in investigation.findings
    ]

    total_findings = len(results)

    grounded_findings = sum(result.grounded for result in results)

    groundedness = grounded_findings / total_findings if total_findings else 1.0

    return GroundednessSummary(
        total_findings=total_findings,
        grounded_findings=grounded_findings,
        groundedness=groundedness,
    )
