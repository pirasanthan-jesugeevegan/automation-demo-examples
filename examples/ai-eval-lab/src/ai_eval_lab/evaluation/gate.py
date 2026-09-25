"""Pass or fail an evaluation run, and say why."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Gate:
    min_pass_rate: float = 1.0
    # A fabricated citation is a different kind of failure from a missing finding: in an
    # investigation it is a hard stop. Off unless set, so a threshold is a deliberate choice.
    max_invalid_citations: int | None = None


def check_gate(pass_rate: float, invalid_citations: int, gate: Gate) -> list[str]:
    """Reasons the run failed the gate. Empty means it passed."""
    failures: list[str] = []

    if pass_rate < gate.min_pass_rate:
        failures.append(f"pass rate {pass_rate:.0%} is below the {gate.min_pass_rate:.0%} minimum")

    if gate.max_invalid_citations is not None and invalid_citations > gate.max_invalid_citations:
        failures.append(
            f"{invalid_citations} fabricated citation(s), "
            f"at most {gate.max_invalid_citations} allowed"
        )

    return failures
