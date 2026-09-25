# ai-eval-lab: testing an AI that cites its sources

An AI investigator reads documents about a company or person and reports findings, each with
citations. It can go wrong in ways a normal test suite will not notice: it states the wrong amount,
cites the wrong document, invents a source, or gives an answer that was true last year. And when the
model is upgraded, "it still runs" says nothing about whether it got worse.

This project evaluates that kind of system, and evaluates the evaluator too:

- **Golden cases** score what the AI produced, and a **gate** fails the build when it gets worse.
- A **seeded regression** proves the gate catches real defects.
- A **judge validation** measures how far to trust the LLM judge, against human labels.
- **Repeated live runs** show the worst run and which cases are flaky, because a model varies.

## Try it, no API key needed

```bash
uv sync --extra dev
uv run python -m ai_eval_lab --recording baseline     # a healthy system: passes, exit 0
uv run python -m ai_eval_lab --recording regression   # a worse model version: fails, exit 1
uv run python -m ai_eval_lab.judge_check              # how far can the offline check be trusted?
```

`baseline` and `regression` are recorded outputs. The regression has four deliberate defects, and the
gate names each one:

```
CASE-001  FAIL  heuristic groundedness 0/1
    CASE-001: expected finding was not produced: 'Acme Holdings received a £2 million regulatory penalty in 2024.'
    CASE-001: unexpected finding: 'Acme Holdings received a £20 million regulatory penalty in 2024.'
CASE-002  FAIL  heuristic groundedness 0/1
    CASE-002: finding has incorrect source references.
CASE-003  PASS  heuristic groundedness 0/1
CASE-004  FAIL  heuristic groundedness 0/1
    AI output contains invalid source references: source-404
    CASE-004: unexpected finding: 'Acme Holdings was fined £5 million in 2022.'
CASE-005  FAIL  heuristic groundedness 0/1
    CASE-005: expected finding was not produced: 'Bob Jones is the current CEO of Acme Holdings.'
    CASE-005: unexpected finding: 'John Smith is the current CEO of Acme Holdings.'

1/5 cases passed (20%)
Gate: FAILED
  - pass rate 20% is below the 100% minimum
  - 1 fabricated citation(s), at most 0 allowed
```

A hallucinated amount, a namesake's notice pinned on the wrong person, an invented source and an
outdated answer. CI runs both recordings: the baseline must pass and the regression must fail. If the
regression ever passes, the evaluation itself is broken and the build says so.

`heuristic groundedness 1/1` is the offline check's count of findings it judges supported. It is
information, not part of the gate, and it is naive on purpose (see "What it cannot do").

## Can the judge be trusted?

`judge_check` runs a judge over 18 claims that a person labelled by reading the sources, nine
supported and nine not, chosen to be hard: paraphrase, an abbreviation, negation, a wrong year, a
namesake, sources that disagree.

```
Judge: heuristic, 18 labelled claims

                    human: grounded   human: not grounded
judge: grounded                 7                     2   <- false passes
judge: not grounded             2                     7

agreement        78%
Cohen's kappa    0.56  (agreement beyond chance)
false-pass rate  22%  (unsupported claims let through)
false-fail rate  22%  (supported claims rejected)
```

The offline check's four mistakes are the ones you would predict from how it works: it cannot see
negation, does not know that "CEO" means Chief Executive Officer, cannot tell a notice from a fine,
and cannot verify a claim that sources disagree. That is why an LLM judge exists. To measure it the
same way, run `--judge anthropic` (needs `ANTHROPIC_API_KEY`), then compare the two tables. The
false-pass rate is the number to watch, because it is the one that lets bad claims into a report.

## Evaluate your own system

Implement one method and the runner, gate and reports work unchanged:

```python
class Investigator(Protocol):
    def investigate(self, case: EvaluationCase, sources: list[SourceDocument]) -> Investigation: ...
```

`investigators/anthropic.py` is a working example that asks Claude. To run it against the golden cases
several times and gate the worst run:

```bash
ANTHROPIC_API_KEY=... uv run python -m ai_eval_lab.live --runs 5 --min-pass-rate 0.8 \
  --max-invalid-citations 0 --report live.json
```

A model can answer differently each time, so the output shows per-case pass counts and marks cases
that pass on some runs and fail on others as `FLAKY`. The investigator and the judge default to
different models: a model rates its own writing generously. See
[docs/evaluating-your-system.md](docs/evaluating-your-system.md).

## What is in it

| Piece | What it does |
| --- | --- |
| `datasets/golden` | Five cases chosen to be hard: a clear finding, two people with one name, sources that disagree, no evidence at all, and an answer that has gone out of date. |
| `datasets/judge_labels` | The human-labelled claims used to validate a judge. |
| `evaluation/deterministic.py` | Scores an investigation with no model: are the cited sources real, were the expected findings produced with the right sources, did it add findings nobody asked for. |
| `evaluation/groundedness.py` | A fast offline check that a claim is supported by its sources. |
| `evaluation/anthropic_judge.py` | Asks Claude for a verdict. Structured output guarantees the reply fits the schema, and source text is fenced as untrusted data, so a hostile document cannot talk the judge into a pass. |
| `evaluation/gate.py` | Pass or fail, with reasons: a minimum pass rate, and an optional zero-tolerance limit on fabricated citations. |
| `evaluation/repeated.py`, `live.py` | Repeat a live run, report worst and mean, flag flaky cases. |
| `evaluation/judge_validation.py`, `judge_check.py` | Agreement, kappa, and false-pass and false-fail rates against human labels. |
| `investigators/` | `recorded` replays saved outputs for the demo and CI. `anthropic` is a live system under test. |

Every command can write its result as JSON with `--report`, for a dashboard or a trend. The dataset
formats and how to add a case are in [docs/dataset.md](docs/dataset.md).

## What it cannot do

- **The offline groundedness check is naive on purpose.** See the table above for what it gets wrong.
  The pass/fail gate uses the deterministic score, not this heuristic.
- **Findings are matched by exact text** after normalising case, spacing and a trailing full stop. A
  correct paraphrase counts as a miss. The next step would be to match by meaning with the judge.
- **Eighteen labels and five cases show the method, not statistics.** A real evaluation needs many
  more, drawn from real failures, and labels checked by more than one person.
- **The recordings are hand-written stand-ins** for a system's output.
- **The live path is tested, but not against the real API in CI.** The tests run the real SDK against
  a stubbed transport. The first run with your own key is the first time real model output flows
  through it, so expect to tune the investigator prompt.

## How it started

The first version was written while learning how to evaluate LLM output. Reviewing it found real
bugs, each now pinned by a test that fails on the original code:

- Groundedness crashed when a claim shared no words with its source, or was only stop words.
- A capitalised sentence opener such as "In" counted as an entity, and a year counted as an amount,
  so a wrong year was penalised twice.
- The test suite did not collect (an `IndentationError`), and a test only passed from one directory.
- Two identical result models, and a `Judge` interface that the fake and the real judge did not both
  satisfy.
- The judge parsed model output with `json.loads`, which fails on a fenced reply, and put untrusted
  source text straight into the prompt.
- The default model was pinned to an older ID, and an empty `ANTHROPIC_MODEL` from `.env.example`
  overrode it.

## Development

```bash
uv run pytest          # 119 tests
uv run ruff check .
uv run ruff format --check .
uv run mypy            # strict, source and tests
```

Python 3.12 or newer, managed with [uv](https://docs.astral.sh/uv/).
