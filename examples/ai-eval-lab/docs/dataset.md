# Datasets

Everything is JSON under `datasets/`. Each command validates the files it reads (golden cases only
cite sources that exist, labels too, recordings cover every case) and exits with code 2 if they are
inconsistent.

## `sources/source-NNN.json`

One document the AI may cite. The file name must match the id.

```json
{
  "id": "source-001",
  "title": "Regulatory Decision Against Acme Holdings",
  "source_type": "regulator",
  "content": "In 2024, the regulator announced a £2 million penalty against Acme Holdings."
}
```

## `golden/cases.json`

A list of cases. Each says what a correct investigation of the entity contains.

```json
{
  "id": "CASE-001",
  "description": "Straightforward regulatory finding",
  "entity": { "name": "Acme Holdings", "entity_type": "company" },
  "expected_findings": [
    {
      "claim": "Acme Holdings received a £2 million regulatory penalty in 2024.",
      "source_ids": ["source-001"]
    }
  ]
}
```

An empty `expected_findings` means the correct answer is to report nothing.

## `recorded/<name>.json`

Saved output of a system under test, used by `--recording <name>`. It maps each case id to the findings
the system produced. Every recording must cover every case.

```json
{ "CASE-001": [{ "claim": "…", "source_ids": ["source-001"], "confidence": 0.94 }] }
```

## `judge_labels/labels.json`

Claims a person labelled by reading the cited sources. `grounded` is true only if the cited sources
state the claim.

```json
{
  "id": "LABEL-03",
  "claim": "Acme Holdings received a £20 million regulatory penalty in 2024.",
  "source_ids": ["source-001"],
  "grounded": false,
  "why": "The amount is ten times the source's £2 million."
}
```

Keep the set balanced. Only supported claims cannot show false passes, and only unsupported ones
cannot show false fails.

## Adding a case

1. Add any new source documents.
2. Add the case to `golden/cases.json`, with the claims a correct system reports and the sources that
   state them.
3. Add the correct output to `recorded/baseline.json`.
4. Run `uv run pytest`. The dataset tests fail if a case cites an unknown source or a recording is
   missing a case.

Cases that come from real failures are worth more than ones invented to be tricky.
