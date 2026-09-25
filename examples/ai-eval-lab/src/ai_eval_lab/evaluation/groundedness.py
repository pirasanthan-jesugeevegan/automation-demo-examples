"""A fast, offline check of whether a finding is supported by the sources it cites.

It measures word overlap, then penalises a mismatched entity, amount or year. It cannot
understand meaning: a claim that negates its source ("was not fined") still scores well, and
a claim that summarises several sources ("the sources disagree") scores badly. That is what
the LLM judge is for; this check is the cheap first pass.
"""

import re

from ai_eval_lab.evaluation.facts import (
    extract_entities,
    extract_numbers,
    extract_years,
)
from ai_eval_lab.models.groundedness import GroundednessResult
from ai_eval_lab.models.investigation import Finding, SourceDocument

GROUNDED_THRESHOLD = 0.6

# Multipliers applied to the overlap score when a check fails. A wrong amount is the most
# damaging error in an investigation, so it is penalised hardest.
ENTITY_MISMATCH_PENALTY = 0.5
NUMBER_MISMATCH_PENALTY = 0.3
DATE_MISMATCH_PENALTY = 0.5

_STOP_WORDS = frozenset(
    {"a", "an", "and", "the", "in", "of", "to", "was", "were", "is", "are", "that", "on", "for"}
)

NO_VALID_SOURCE = "The finding does not cite any valid sources."
NO_OVERLAP = "The cited sources share no content with the claim."


def _content_words(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z0-9£]+\b", text.lower())

    return {word for word in words if word not in _STOP_WORDS}


def _score_against_source(claim: str, source: SourceDocument) -> tuple[float, str]:
    claim_words = _content_words(claim)

    if not claim_words:
        return 0.0, NO_OVERLAP

    overlap = claim_words & _content_words(source.content)
    score = len(overlap) / len(claim_words)

    problems: list[str] = []

    if not extract_entities(claim) <= extract_entities(source.content):
        score *= ENTITY_MISMATCH_PENALTY
        problems.append("entity mismatch")

    if not extract_numbers(claim) <= extract_numbers(source.content):
        score *= NUMBER_MISMATCH_PENALTY
        problems.append("number mismatch")

    if not extract_years(claim) <= extract_years(source.content):
        score *= DATE_MISMATCH_PENALTY
        problems.append("date mismatch")

    if not problems:
        return score, "lexical, entity, number and date checks passed"

    return score, ", ".join(problems)


def evaluate_groundedness(
    finding: Finding,
    sources: list[SourceDocument],
) -> GroundednessResult:
    sources_by_id = {source.id: source for source in sources}
    cited = [sources_by_id[sid] for sid in finding.source_ids if sid in sources_by_id]

    if not cited:
        return GroundednessResult(
            grounded=False,
            score=0.0,
            explanation=NO_VALID_SOURCE,
            supporting_source_ids=[],
        )

    # max() keeps the first of equal scores, so the order the finding cites them in breaks ties.
    scored = [(_score_against_source(finding.claim, source), source.id) for source in cited]
    (best_score, explanation), best_source_id = max(scored, key=lambda item: item[0][0])

    grounded = best_score >= GROUNDED_THRESHOLD

    return GroundednessResult(
        grounded=grounded,
        score=best_score,
        explanation=explanation,
        supporting_source_ids=[best_source_id] if grounded else [],
    )
