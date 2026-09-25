import pytest

from ai_eval_lab.config import (
    DEFAULT_INVESTIGATOR_MODEL,
    DEFAULT_MODEL,
    get_anthropic_api_key,
    get_anthropic_model,
    get_investigator_model,
)


def test_model_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    assert get_anthropic_model() == DEFAULT_MODEL


def test_model_defaults_when_copied_from_an_empty_env_example(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "")

    assert get_anthropic_model() == DEFAULT_MODEL


def test_model_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    assert get_anthropic_model() == "claude-haiku-4-5-20251001"


def test_missing_api_key_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_anthropic_api_key()


def test_api_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")

    assert get_anthropic_api_key() == "sk-ant-not-a-real-key"


def test_the_investigator_and_the_judge_default_to_different_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_INVESTIGATOR_MODEL", raising=False)

    assert get_investigator_model() == DEFAULT_INVESTIGATOR_MODEL
    assert get_investigator_model() != get_anthropic_model()


def test_the_investigator_model_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_INVESTIGATOR_MODEL", "claude-sonnet-5")

    assert get_investigator_model() == "claude-sonnet-5"
