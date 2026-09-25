"""Settings read from the environment. Only the live judge needs an API key."""

import os

# The judge and the system under test default to different models. A model tends to rate its own
# writing generously, so judging with the model that produced the answer flatters it.
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_INVESTIGATOR_MODEL = "claude-haiku-4-5-20251001"


def get_anthropic_api_key() -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    return api_key


def get_anthropic_model() -> str:
    # `or`, not a getenv default: copying .env.example leaves ANTHROPIC_MODEL set but empty.
    return os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL


def get_investigator_model() -> str:
    return os.getenv("ANTHROPIC_INVESTIGATOR_MODEL") or DEFAULT_INVESTIGATOR_MODEL
