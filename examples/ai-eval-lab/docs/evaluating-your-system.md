# Evaluating your own system

The runner needs one thing from you: something that, given a case and the available sources, returns
an `Investigation` (the findings, each with the ids of the sources it cites).

## 1. Implement `Investigator`

```python
from ai_eval_lab.models.evaluation_case import EvaluationCase
from ai_eval_lab.models.investigation import Finding, Investigation, SourceDocument


class MySystem:
    def investigate(self, case: EvaluationCase, sources: list[SourceDocument]) -> Investigation:
        findings = call_my_pipeline(case.entity, sources)  # returns claim, source ids, confidence
        return Investigation(entity=case.entity, sources=sources, findings=findings)
```

Report in `Investigation.sources` the sources your system actually had. A finding that cites an id
outside that list counts as a fabricated citation.

## 2. Run it

```python
from ai_eval_lab.cli import load_dataset
from ai_eval_lab.evaluation.gate import Gate, check_gate
from ai_eval_lab.evaluation.repeated import run_repeated
from pathlib import Path

cases, sources = load_dataset(Path("datasets"))
report = run_repeated(cases, sources, MySystem(), runs=5)

failures = check_gate(
    report.worst_pass_rate,
    report.worst_invalid_citations,
    Gate(min_pass_rate=0.8, max_invalid_citations=0),
)
print(report.pass_rates, report.flaky_case_ids(), failures)
```

For a single deterministic run use `run_evaluation` from `ai_eval_lab.evaluation.runner`.

## 3. Or use the built-in live system

```bash
ANTHROPIC_API_KEY=... uv run python -m ai_eval_lab.live --runs 5 --min-pass-rate 0.8 \
  --max-invalid-citations 0 --report live.json
```

| Flag | Meaning |
| --- | --- |
| `--runs N` | Repeat the whole evaluation N times. Default 3. |
| `--min-pass-rate` | Fraction of cases that must pass on the worst run. Default 1.0. |
| `--max-invalid-citations` | Fail if any run cites more sources than this that do not exist. Off unless set. |
| `--judge anthropic` | Also ask Claude about every finding and list where it disagrees with the offline check. |
| `--report PATH` | Write every run as JSON. |

Exit codes: `0` passed the gate, `1` below the gate, `2` bad input (missing key, unreadable or
inconsistent dataset).

**Cost.** One model call per case per run, plus one per finding with `--judge`. Five cases times five
runs is 25 investigator calls.

**Models.** `ANTHROPIC_INVESTIGATOR_MODEL` (default `claude-haiku-4-5-20251001`) is the system under
test. `ANTHROPIC_MODEL` (default `claude-sonnet-5`) is the judge. Keep them different: a model tends to
rate its own writing generously.

## Choosing a gate

- **Minimum pass rate** catches general decline. On live runs a value below 1.0 leaves room for a
  model that is right most of the time.
- **Fabricated citations** should usually be zero. A missing finding is a gap; an invented source is
  a lie, and one is enough to distrust the output.
- The gate applies to the **worst** run, not the mean, so one bad run in five is not averaged away.
  The mean is printed beside it so you can see the difference.

## Before trusting the judge

Run `python -m ai_eval_lab.judge_check --judge anthropic` first. If its false-pass rate is high, a
green evaluation is not evidence of a good system.
