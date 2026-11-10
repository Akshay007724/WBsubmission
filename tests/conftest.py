import pytest

from acquirer_engine import data
from acquirer_engine.profile import TargetProfile


@pytest.fixture(scope="session")
def ds():
    return data.load()


@pytest.fixture
def profile():
    return TargetProfile()


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch, tmp_path):
    """Tests must never read the developer's .env or call a paid API."""
    from acquirer_engine import config

    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "no.env")
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
