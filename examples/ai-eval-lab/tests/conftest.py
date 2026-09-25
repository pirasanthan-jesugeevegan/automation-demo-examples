from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def datasets_dir() -> Path:
    """Absolute path, so tests pass from any working directory."""
    return Path(__file__).resolve().parents[1] / "datasets"
