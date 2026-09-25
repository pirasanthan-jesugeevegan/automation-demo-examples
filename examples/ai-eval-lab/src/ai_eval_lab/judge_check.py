"""Validate a judge against human labels: `python -m ai_eval_lab.judge_check`.

Defaults to the offline heuristic. With `--judge anthropic` (needs ANTHROPIC_API_KEY) it validates
Claude the same way, one model call per labelled claim, so the two can be compared.
"""

import argparse
import json
import sys
from pathlib import Path

from ai_eval_lab.cli import EXIT_BAD_INPUT, EXIT_BELOW_GATE, EXIT_OK
from ai_eval_lab.datasets.loader import load_sources
from ai_eval_lab.evaluation.heuristic_judge import HeuristicJudge
from ai_eval_lab.evaluation.judge import Judge
from ai_eval_lab.evaluation.judge_validation import (
    JudgeValidationReport,
    load_labels,
    validate_judge,
    validate_labels,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai_eval_lab.judge_check",
        description="Measure how often a judge agrees with human labels, and which way it errs.",
    )
    parser.add_argument("--datasets", type=Path, default=Path("datasets"))
    parser.add_argument("--judge", choices=["heuristic", "anthropic"], default="heuristic")
    parser.add_argument("--min-agreement", type=float, help="Fail below this agreement (0 to 1).")
    parser.add_argument(
        "--max-false-pass-rate",
        type=float,
        help="Fail if more than this share of unsupported claims are let through (0 to 1).",
    )
    parser.add_argument("--report", type=Path, help="Write the result as JSON to this path.")
    return parser.parse_args(argv)


def _build_judge(name: str) -> Judge:
    if name == "heuristic":
        return HeuristicJudge()

    from ai_eval_lab.evaluation.anthropic_judge import AnthropicJudge

    return AnthropicJudge()


def _gate_failures(report: JudgeValidationReport, args: argparse.Namespace) -> list[str]:
    failures: list[str] = []

    if args.min_agreement is not None and report.agreement < args.min_agreement:
        failures.append(f"agreement {report.agreement:.0%} is below {args.min_agreement:.0%}")

    if args.max_false_pass_rate is not None and report.false_pass_rate > args.max_false_pass_rate:
        failures.append(
            f"false-pass rate {report.false_pass_rate:.0%} is above {args.max_false_pass_rate:.0%}"
        )

    return failures


def _print_report(name: str, report: JudgeValidationReport) -> None:
    print(f"Judge: {name}, {report.total} labelled claims\n")
    print("                    human: grounded   human: not grounded")
    print(
        f"judge: grounded     {report.true_positives:>13}   {report.false_positives:>19}"
        "   <- false passes"
    )
    print(f"judge: not grounded {report.false_negatives:>13}   {report.true_negatives:>19}\n")
    print(f"agreement        {report.agreement:.0%}")
    print(f"Cohen's kappa    {report.cohens_kappa:.2f}  (agreement beyond chance)")
    print(f"false-pass rate  {report.false_pass_rate:.0%}  (unsupported claims let through)")
    print(f"false-fail rate  {report.false_fail_rate:.0%}  (supported claims rejected)")

    if report.disagreements:
        print("\nWhere the judge and the human disagree:")
        for line in report.disagreements:
            print(f"  {line}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        sources = load_sources(args.datasets / "sources")
        labels = load_labels(args.datasets / "judge_labels" / "labels.json")
        problems = validate_labels(labels, sources)
        if problems:
            raise ValueError("Judge labels are inconsistent:\n  " + "\n  ".join(problems))
        judge = _build_judge(args.judge)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Cannot start: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT

    report = validate_judge(judge, labels, sources)
    failures = _gate_failures(report, args)

    _print_report(args.judge, report)
    for failure in failures:
        print(f"\nGate FAILED: {failure}")

    if args.report:
        payload = {
            "judge": args.judge,
            "total": report.total,
            "agreement": report.agreement,
            "cohens_kappa": report.cohens_kappa,
            "false_pass_rate": report.false_pass_rate,
            "false_fail_rate": report.false_fail_rate,
            "confusion": {
                "true_positives": report.true_positives,
                "false_positives": report.false_positives,
                "true_negatives": report.true_negatives,
                "false_negatives": report.false_negatives,
            },
            "disagreements": report.disagreements,
            "gate_failures": failures,
        }
        args.report.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    return EXIT_BELOW_GATE if failures else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
