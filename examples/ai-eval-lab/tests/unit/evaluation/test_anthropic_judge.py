import pytest

from ai_eval_lab.config import DEFAULT_MODEL
from ai_eval_lab.evaluation.anthropic_judge import AnthropicJudge, JudgeError
from ai_eval_lab.models.groundedness import GroundednessResult
from ai_eval_lab.models.investigation import Finding, SourceDocument
from tests.fakes import FakeAnthropic

VERDICT = GroundednessResult(
    grounded=True,
    score=0.95,
    explanation="The source supports the claim.",
    supporting_source_ids=["source-001"],
)


FINDING = Finding(
    claim="Acme Holdings received a £2 million penalty.",
    source_ids=["source-001"],
    confidence=0.95,
)
SOURCE = SourceDocument(
    id="source-001",
    title="Regulatory Decision",
    source_type="regulator",
    content="Acme Holdings received a £2 million penalty.",
)


@pytest.fixture(autouse=True)
def default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)


def test_returns_the_structured_verdict() -> None:
    result = AnthropicJudge(client=FakeAnthropic(VERDICT).as_client()).evaluate(FINDING, [SOURCE])

    assert result == VERDICT


def test_asks_for_the_verdict_schema_with_the_configured_model() -> None:
    client = FakeAnthropic(VERDICT)

    AnthropicJudge(client=client.as_client()).evaluate(FINDING, [SOURCE])

    (call,) = client.messages.calls
    assert call["model"] == DEFAULT_MODEL
    assert call["output_format"] is GroundednessResult
    assert FINDING.claim in call["messages"][0]["content"]


def test_a_reply_with_no_verdict_is_an_error_not_a_silent_pass() -> None:
    judge = AnthropicJudge(client=FakeAnthropic(None).as_client())

    with pytest.raises(JudgeError, match="no verdict"):
        judge.evaluate(FINDING, [SOURCE])


def test_building_a_judge_without_a_client_needs_an_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicJudge()
