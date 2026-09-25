"""Evaluate a live model: `python -m ai_eval_lab.live --runs 5`. Needs ANTHROPIC_API_KEY.

Each run makes one model call per golden case (plus one per finding with --judge), so cost grows
with --runs. The gate applies to the worst run, not the average.
"""

import argparse
import json
import sys
from pathlib import Path

from ai_eval_lab.cli import (
    EXIT_BAD_INPUT,
    EXIT_BELOW_GATE,
    EXIT_OK,
    add_gate_arguments,
    gate_from,
    load_dataset,
    make_judge,
    print_summary,
)
from ai_eval_lab.evaluation.gate import check_gate
from ai_eval_lab.evaluation.repeated import RepeatedReport, run_repeated
from ai_eval_lab.investigators.anthropic import AnthropicInvestigator


def _runs(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai_eval_lab.live",
        description="Run the golden cases on a live model several times; gate the worst run.",
    )
    parser.add_argument(
        "--runs", type=_runs, default=3, help="How many times to repeat. Default 3."
    )
    parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    add_gate_arguments(parser)
    parser.add_argument("--judge", choices=["anthropic"])
    parser.add_argument("--report", type=Path, help="Write every run as JSON to this path.")
    return parser.parse_args(argv)


def _print_runs(report: RepeatedReport) -> None:
    total = len(report.runs)
    counts = report.case_pass_counts()
    flaky = set(report.flaky_case_ids())

    for case_id, passed in counts.items():
        note = "  FLAKY" if case_id in flaky else ""
        print(f"{case_id}  passed {passed}/{total} runs{note}")

    rates = ", ".join(f"{rate:.0%}" for rate in report.pass_rates)
    print(f"\npass rate per run: {rates} (mean {report.mean_pass_rate:.0%})")


def _write_report(path: Path, report: RepeatedReport, failures: list[str]) -> None:
    payload = {
        "runs": len(report.runs),
        "pass_rates": report.pass_rates,
        "mean_pass_rate": report.mean_pass_rate,
        "worst_pass_rate": report.worst_pass_rate,
        "worst_invalid_citations": report.worst_invalid_citations,
        "flaky_cases": report.flaky_case_ids(),
        "gate_passed": not failures,
        "gate_failures": failures,
        "reports": [run.model_dump(mode="json") for run in report.runs],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        cases, sources = load_dataset(args.datasets)
        investigator = AnthropicInvestigator()
        judge = make_judge(args.judge)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Cannot start: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT

    report = run_repeated(cases, sources, investigator, args.runs, judge)

    failures = check_gate(report.worst_pass_rate, report.worst_invalid_citations, gate_from(args))

    _print_runs(report)
    worst = min(report.runs, key=lambda run: run.pass_rate)
    print_summary(worst.passed_cases, len(worst.cases), worst.pass_rate, failures)
    print("(the gate applies to the worst run)")

    if args.report:
        _write_report(args.report, report, failures)

    return EXIT_BELOW_GATE if failures else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
