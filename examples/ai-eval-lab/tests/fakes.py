from types import SimpleNamespace
from typing import Any, cast

from anthropic import Anthropic
from pydantic import BaseModel

from ai_eval_lab.models.groundedness import GroundednessResult
from ai_eval_lab.models.investigation import Finding, SourceDocument


class FakeJudge:
    """Returns a fixed verdict, so tests never call a model."""

    def __init__(self, result: GroundednessResult) -> None:
        self.result = result

    def evaluate(
        self,
        finding: Finding,  # noqa: ARG002 - part of the Judge interface
        sources: list[SourceDocument],  # noqa: ARG002
    ) -> GroundednessResult:
        return self.result


class FakeMessages:
    def __init__(self, parsed_output: BaseModel | None) -> None:
        self.parsed_output = parsed_output
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output=self.parsed_output, stop_reason="end_turn")


class FakeAnthropic:
    """Stands in for the SDK client's `messages.parse`, and records what it was asked."""

    def __init__(self, parsed_output: BaseModel | None) -> None:
        self.messages = FakeMessages(parsed_output)

    def as_client(self) -> Anthropic:
        return cast(Anthropic, self)
