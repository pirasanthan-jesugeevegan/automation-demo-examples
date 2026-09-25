from anthropic import Anthropic

from ai_eval_lab.config import get_anthropic_api_key, get_anthropic_model
from ai_eval_lab.evaluation.judge_prompt import DEFAULT_RUBRIC, build_groundedness_prompt
from ai_eval_lab.models.groundedness import GroundednessResult
from ai_eval_lab.models.investigation import Finding, SourceDocument

MAX_TOKENS = 1024


class JudgeError(RuntimeError):
    """The model did not return a usable verdict."""


class AnthropicJudge:
    """Asks Claude for a verdict. Structured output guarantees the reply matches the schema."""

    def __init__(
        self,
        client: Anthropic | None = None,
        rubric: str = DEFAULT_RUBRIC,
    ) -> None:
        self.client = client or Anthropic(api_key=get_anthropic_api_key())
        self.model = get_anthropic_model()
        self.rubric = rubric

    def evaluate(
        self,
        finding: Finding,
        sources: list[SourceDocument],
    ) -> GroundednessResult:
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": build_groundedness_prompt(finding, sources, self.rubric),
                }
            ],
            output_format=GroundednessResult,
        )

        if response.parsed_output is None:
            raise JudgeError(
                f"The judge returned no verdict (stop reason: {response.stop_reason})."
            )

        return response.parsed_output
