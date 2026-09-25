"""Command line entry point: `python -m ai_eval_lab --recording baseline`."""

import argparse
import json
import sys
from pathlib import Path

from ai_eval_lab.datasets.loader import load_evaluation_cases, load_sources
from ai_eval_lab.datasets.validator import validate_source_references
from ai_eval_lab.evaluation.gate import Gate, check_gate
from ai_eval_lab.evaluation.judge import Judge
from ai_eval_lab.evaluation.runner import EvaluationReport, run_evaluation
from ai_eval_lab.investigators.recorded import RecordedInvestigator
from ai_eval_lab.models.evaluation_case import EvaluationCase
from ai_eval_lab.models.investigation import SourceDocument

EXIT_OK = 0
EXIT_BELOW_GATE = 1
EXIT_BAD_INPUT = 2


def load_dataset(datasets_dir: Path) -> tuple[list[EvaluationCase], list[SourceDocument]]:
    """Cases and sources, checked against each other. Raises OSError or ValueError."""
    cases = load_evaluation_cases(datasets_dir / "golden" / "cases.json")
    sources = load_sources(datasets_dir / "sources")

    problems = validate_source_references(cases, sources)
    if problems:
        raise ValueError("Golden dataset is inconsistent:\n  " + "\n  ".join(problems))

    return cases, sources


def make_judge(name: str | None) -> Judge | None:
    """The judge asked for on the command line. Raises RuntimeError if it cannot be built."""
    if name != "anthropic":
        return None

    from ai_eval_lab.evaluation.anthropic_judge import AnthropicJudge

    return AnthropicJudge()


def add_gate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=1.0,
        help="Fraction of cases that must pass (0 to 1). Default 1.0.",
    )
    parser.add_argument(
        "--max-invalid-citations",
        type=int,
        help="Fail if the AI cites more than this many sources that do not exist. Off by default.",
    )


def gate_from(args: argparse.Namespace) -> Gate:
    return Gate(min_pass_rate=args.min_pass_rate, max_invalid_citations=args.max_invalid_citations)


def print_case_lines(report: EvaluationReport) -> None:
    for case in report.cases:
        result = case.result
        status = "PASS" if result.passed else "FAIL"
        grounded = f"{case.groundedness.grounded_findings}/{case.groundedness.total_findings}"
        print(f"{result.case_id}  {status}  heuristic groundedness {grounded}")

        for error in result.errors:
            print(f"    {error}")
        for line in case.judge_disagreements:
            print(f"    judge disagrees: {line}")


def print_summary(passed: int, total: int, pass_rate: float, failures: list[str]) -> None:
    print(f"\n{passed}/{total} cases passed ({pass_rate:.0%})")

    if not failures:
        print("Gate: PASSED")
        return

    print("Gate: FAILED")
    for failure in failures:
        print(f"  - {failure}")


def write_report(path: Path, recording: str, report: EvaluationReport, failures: list[str]) -> None:
    payload = {
        "recording": recording,
        "pass_rate": report.pass_rate,
        "passed_cases": report.passed_cases,
        "total_cases": len(report.cases),
        "invalid_citations": report.invalid_citations,
        "gate_passed": not failures,
        "gate_failures": failures,
        "cases": [case.model_dump(mode="json") for case in report.cases],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai_eval_lab",
        description="Score an AI investigator against the golden cases and gate on the result.",
    )
    parser.add_argument("--recording", required=True, help="Name of a file in datasets/recorded.")
    parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    add_gate_arguments(parser)
    parser.add_argument(
        "--judge",
        choices=["anthropic"],
        help="Also ask Claude for a verdict on every finding (needs ANTHROPIC_API_KEY).",
    )
    parser.add_argument("--report", type=Path, help="Write the full result as JSON to this path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        cases, sources = load_dataset(args.datasets)
        investigator = RecordedInvestigator.from_file(
            args.datasets / "recorded" / f"{args.recording}.json"
        )
    except (OSError, ValueError) as error:
        print(f"Cannot load inputs: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT

    try:
        judge = make_judge(args.judge)
    except RuntimeError as error:
        print(f"Cannot create the judge: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT

    try:
        report = run_evaluation(cases, sources, investigator, judge)
    except (LookupError, ValueError) as error:
        print(f"Cannot evaluate: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT

    failures = check_gate(report.pass_rate, report.invalid_citations, gate_from(args))

    print_case_lines(report)
    print_summary(report.passed_cases, len(report.cases), report.pass_rate, failures)

    if args.report:
        write_report(args.report, args.recording, report, failures)

    return EXIT_BELOW_GATE if failures else EXIT_OK
