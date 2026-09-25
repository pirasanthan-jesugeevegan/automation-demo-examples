"""Runs the judge through the real Anthropic SDK against a stubbed HTTP transport.

The fake client in test_anthropic_judge.py proves the judge's logic. This proves the pieces fit
the SDK: the schema built from GroundednessResult is accepted, and a real response body parses.
"""

import json
from typing import Any

import httpx
from anthropic import Anthropic

from ai_eval_lab.evaluation.anthropic_judge import AnthropicJudge
from ai_eval_lab.models.investigation import Finding, SourceDocument

VERDICT_JSON = {
    "grounded": False,
    "score": 0.2,
    "explanation": "The source gives 2 million, the claim says 20 million.",
    "supporting_source_ids": [],
}


def stub_client(requests: list[dict[str, Any]]) -> Anthropic:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": [{"type": "text", "text": json.dumps(VERDICT_JSON)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 120, "output_tokens": 40},
            },
        )

    return Anthropic(
        api_key="not-a-real-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_judge_works_end_to_end_through_the_sdk() -> None:
    requests: list[dict[str, Any]] = []
    finding = Finding(
        claim="Acme Holdings received a £20 million penalty.",
        source_ids=["source-001"],
        confidence=0.9,
    )
    source = SourceDocument(
        id="source-001",
        title="Regulatory Decision",
        source_type="regulator",
        content="The regulator announced a £2 million penalty against Acme Holdings.",
    )

    result = AnthropicJudge(client=stub_client(requests)).evaluate(finding, [source])

    assert result.grounded is False
    assert result.score == 0.2

    (body,) = requests
    schema = body["output_config"]["format"]["schema"]
    assert {"grounded", "score", "explanation", "supporting_source_ids"} <= set(
        schema["properties"]
    )
    assert finding.claim in body["messages"][0]["content"]
